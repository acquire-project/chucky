#include "lod/reduce_csr.h"

#include "defs.limits.h"
#include "lod/lod_plan.h"
#include "util/index.ops.h"

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Per-source-element mapping, populated in the first parallel pass of build
// and scattered in the second. Private implementation detail.
struct src_map
{
  uint64_t dst_elem;
  uint64_t src_elem;
};

int
reduce_csr_alloc(struct reduce_csr* csr, uint64_t src_total, uint64_t dst_total)
{
  memset(csr, 0, sizeof(*csr));
  csr->batch_count = 1;
  csr->dst_segment_size = dst_total;
  csr->src_lod_count = src_total;

  if (src_total == 0 || dst_total == 0)
    return 0;

  csr->starts = (uint64_t*)calloc(dst_total + 1, sizeof(uint64_t));
  csr->indices = (uint64_t*)malloc(src_total * sizeof(uint64_t));
  csr->scratch_map = (struct src_map*)malloc(src_total * sizeof(struct src_map));
  csr->scratch_wpos = (uint64_t*)malloc(dst_total * sizeof(uint64_t));

  if (!csr->starts || !csr->indices || !csr->scratch_map || !csr->scratch_wpos) {
    reduce_csr_free(csr);
    return 1;
  }
  return 0;
}

void
reduce_csr_free(struct reduce_csr* csr)
{
  if (!csr)
    return;
  free(csr->starts);
  free(csr->indices);
  free(csr->scratch_map);
  free(csr->scratch_wpos);
  memset(csr, 0, sizeof(*csr));
}

int
reduce_csr_build(struct reduce_csr* csr,
                 const struct lod_plan* plan,
                 int level)
{
  const struct level_dims* src_ld = &plan->levels.level[level];
  const struct level_dims* dst_ld = &plan->levels.level[level + 1];
  uint32_t dropped_mask = src_ld->lod_mask & ~dst_ld->lod_mask;

  const uint64_t dst_total = csr->dst_segment_size;
  const uint64_t src_total = csr->src_lod_count;

  if (src_total == 0 || dst_total == 0)
    return 0;

  // Zero the counts slots of starts (starts[1..dst_total]) — we accumulate
  // histogram into them. starts[0] is set after prefix sum.
  memset(csr->starts, 0, (dst_total + 1) * sizeof(uint64_t));

  uint64_t src_lod_shape[LOD_MAX_NDIM];
  for (int k = 0; k < src_ld->lod_ndim; ++k)
    src_lod_shape[k] = src_ld->dim[src_ld->lod_to_dim[k]].size;

  uint64_t dst_lod_shape[LOD_MAX_NDIM];
  for (int k = 0; k < dst_ld->lod_ndim; ++k)
    dst_lod_shape[k] = dst_ld->dim[dst_ld->lod_to_dim[k]].size;

  struct src_map* map = csr->scratch_map;
  uint64_t* counts = csr->starts + 1;
  uint64_t lod_nelem = src_ld->lod_nelem;

  // MSVC's OpenMP front-end rejects inline loop-variable declarations
  // (for (int64_t i = 0; ...)) even with /openmp:llvm; declare before.
  // https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c3015
  {
    int64_t gi;
#pragma omp parallel for schedule(static)
    for (gi = 0; gi < (int64_t)src_total; ++gi) {
      uint64_t src_batch = (uint64_t)gi / lod_nelem;
      uint64_t src_enum = (uint64_t)gi % lod_nelem;

      uint64_t fixed_coords[LOD_MAX_NDIM];
      memset(fixed_coords, 0, sizeof(fixed_coords));
      {
        uint64_t rem = src_batch;
        for (int k = src_ld->fixed_dims_ndim - 1; k >= 0; --k) {
          fixed_coords[src_ld->fixed_dim_to_dim[k]] =
            rem % src_ld->fixed_dims_shape[k];
          rem /= src_ld->fixed_dims_shape[k];
        }
      }

      uint64_t src_coords[LOD_MAX_NDIM];
      unravel(src_ld->lod_ndim, src_lod_shape, src_enum, src_coords);

      uint64_t dst_fixed_coords[LOD_MAX_NDIM];
      memcpy(dst_fixed_coords, fixed_coords, sizeof(dst_fixed_coords));

      uint64_t dst_lod_coords[LOD_MAX_NDIM];
      int si = 0;
      for (int k = 0; k < src_ld->lod_ndim; ++k) {
        int d = src_ld->lod_to_dim[k];
        if (dropped_mask & (1u << d))
          dst_fixed_coords[d] = src_coords[k] / 2;
        else
          dst_lod_coords[si++] = src_coords[k] / 2;
      }

      uint64_t dst_morton =
        (dst_ld->lod_ndim > 0)
          ? morton_rank(dst_ld->lod_ndim, dst_lod_shape, dst_lod_coords, 0)
          : 0;

      uint64_t dst_bi = 0;
      {
        uint64_t rem = 0;
        for (int k = 0; k < dst_ld->fixed_dims_ndim; ++k) {
          rem = rem * dst_ld->fixed_dims_shape[k] +
                dst_fixed_coords[dst_ld->fixed_dim_to_dim[k]];
        }
        dst_bi = rem;
      }

      uint64_t dst_elem = dst_bi * dst_ld->lod_nelem + dst_morton;
      uint64_t src_morton =
        morton_rank(src_ld->lod_ndim, src_lod_shape, src_coords, 0);
      uint64_t src_elem = src_batch * src_ld->lod_nelem + src_morton;

      map[gi].dst_elem = dst_elem;
      map[gi].src_elem = src_elem;
#pragma omp atomic
      counts[dst_elem]++;
    }
  }

  // Prefix sum.
  csr->starts[0] = 0;
  for (uint64_t i = 0; i < dst_total; ++i)
    csr->starts[i + 1] += csr->starts[i];

  // Scatter using preallocated scratch_wpos.  Serial so ordering within each
  // bucket is deterministic (iterating gi in ascending order), which keeps
  // downstream reductions (notably float sums) reproducible across builds.
  uint64_t* write_pos = csr->scratch_wpos;
  for (uint64_t i = 0; i < dst_total; ++i)
    write_pos[i] = csr->starts[i];

  for (uint64_t i = 0; i < src_total; ++i) {
    uint64_t pos = write_pos[map[i].dst_elem]++;
    csr->indices[pos] = map[i].src_elem;
  }

  return 0;
}
