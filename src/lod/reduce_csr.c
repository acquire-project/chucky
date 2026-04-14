#include "lod/reduce_csr.h"

#include "defs.limits.h"
#include "lod/lod_plan.h"
#include "util/index.ops.h"

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

  if (!csr->starts || !csr->indices) {
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
  memset(csr, 0, sizeof(*csr));
}

// Derive dst_elem (and, if want_src != NULL, src_elem + within-dst offset) for
// one source element gi. All coordinate work happens on stack arrays sized by
// LOD_MAX_NDIM, so the compiler keeps them in registers for typical ndim.
//
// The within-dst offset is a mixed-radix encoding of parity bits across src
// LOD dims (iterated k=0..src_lod_ndim-1). For each k:
//   parity_k = src_coords[k] & 1      (which half of the src maps to dst)
//   extent_k = min(2, src_size_k - 2 * (src_coords[k] / 2))   (1 at boundary)
//   offset  += parity_k * stride
//   stride  *= extent_k
// The (parity_0..parity_{n-1}) tuple uniquely identifies the src within its
// dst's children, so the formula is a bijection onto [0, count) and requires
// no knowledge of other threads' writes.
static inline void
compute_mapping(uint64_t gi,
                const struct level_dims* src_ld,
                const struct level_dims* dst_ld,
                const uint64_t* src_lod_shape,
                const uint64_t* dst_lod_shape,
                uint32_t dropped_mask,
                uint64_t* out_dst_elem,
                uint64_t* out_src_elem,
                uint64_t* out_offset)
{
  uint64_t lod_nelem = src_ld->lod_nelem;
  uint64_t src_batch = gi / lod_nelem;
  uint64_t src_enum = gi % lod_nelem;

  uint64_t src_coords[LOD_MAX_NDIM];
  unravel(src_ld->lod_ndim, src_lod_shape, src_enum, src_coords);

  // Non-LOD fixed coords (indexed by full dim d). Dims not touched here stay 0.
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

  // Build dst_fixed (by full dim d) and dst_lod (by dst lod index) by halving
  // each src LOD coord. Dropped LOD coords become dst-fixed; kept LOD coords
  // stay LOD.
  uint64_t dst_fixed_coords[LOD_MAX_NDIM];
  memcpy(dst_fixed_coords, fixed_coords, sizeof(dst_fixed_coords));
  uint64_t dst_lod_coords[LOD_MAX_NDIM];
  int si = 0;
  for (int k = 0; k < src_ld->lod_ndim; ++k) {
    int d = src_ld->lod_to_dim[k];
    uint64_t halved = src_coords[k] / 2;
    if (dropped_mask & (1u << d))
      dst_fixed_coords[d] = halved;
    else
      dst_lod_coords[si++] = halved;
  }

  uint64_t dst_morton =
    (dst_ld->lod_ndim > 0)
      ? morton_rank(dst_ld->lod_ndim, dst_lod_shape, dst_lod_coords, 0)
      : 0;

  uint64_t dst_bi = 0;
  for (int k = 0; k < dst_ld->fixed_dims_ndim; ++k) {
    dst_bi = dst_bi * dst_ld->fixed_dims_shape[k] +
             dst_fixed_coords[dst_ld->fixed_dim_to_dim[k]];
  }

  *out_dst_elem = dst_bi * dst_ld->lod_nelem + dst_morton;

  if (!out_src_elem)
    return;

  uint64_t src_morton =
    morton_rank(src_ld->lod_ndim, src_lod_shape, src_coords, 0);
  *out_src_elem = src_batch * lod_nelem + src_morton;

  uint64_t offset = 0;
  uint64_t stride = 1;
  for (int k = 0; k < src_ld->lod_ndim; ++k) {
    uint64_t parity = src_coords[k] & 1;
    uint64_t dst_c = src_coords[k] / 2;
    uint64_t extent = src_lod_shape[k] - 2 * dst_c;
    if (extent > 2)
      extent = 2;
    offset += parity * stride;
    stride *= extent;
  }
  *out_offset = offset;
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

  uint64_t src_lod_shape[LOD_MAX_NDIM];
  for (int k = 0; k < src_ld->lod_ndim; ++k)
    src_lod_shape[k] = src_ld->dim[src_ld->lod_to_dim[k]].size;

  uint64_t dst_lod_shape[LOD_MAX_NDIM];
  for (int k = 0; k < dst_ld->lod_ndim; ++k)
    dst_lod_shape[k] = dst_ld->dim[dst_ld->lod_to_dim[k]].size;

  // Pass 1: histogram counts into starts[1..] so the in-place prefix sum below
  // turns starts[] into the exclusive bucket-base array we need for pass 2.
  memset(csr->starts, 0, (dst_total + 1) * sizeof(uint64_t));

  // MSVC's OpenMP front-end rejects inline loop-variable declarations
  // (for (int64_t i = 0; ...)) even with /openmp:llvm; declare before.
  // https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c3015
  {
    int64_t gi;
#pragma omp parallel for schedule(static)
    for (gi = 0; gi < (int64_t)src_total; ++gi) {
      uint64_t dst_elem;
      compute_mapping((uint64_t)gi,
                      src_ld,
                      dst_ld,
                      src_lod_shape,
                      dst_lod_shape,
                      dropped_mask,
                      &dst_elem,
                      NULL,
                      NULL);
#pragma omp atomic
      csr->starts[dst_elem + 1]++;
    }
  }

  // In-place exclusive prefix sum. starts[0] is already 0.
  for (uint64_t i = 0; i < dst_total; ++i)
    csr->starts[i + 1] += csr->starts[i];

  // Pass 2: deterministic scatter. Each src writes to
  // indices[starts[dst_elem] + offset] where offset is unique among src
  // elements mapping to the same dst (mixed-radix encoding of parity bits), so
  // threads never collide on the same index.
  {
    int64_t gi;
#pragma omp parallel for schedule(static)
    for (gi = 0; gi < (int64_t)src_total; ++gi) {
      uint64_t dst_elem, src_elem, offset;
      compute_mapping((uint64_t)gi,
                      src_ld,
                      dst_ld,
                      src_lod_shape,
                      dst_lod_shape,
                      dropped_mask,
                      &dst_elem,
                      &src_elem,
                      &offset);
      csr->indices[csr->starts[dst_elem] + offset] = src_elem;
    }
  }

  return 0;
}
