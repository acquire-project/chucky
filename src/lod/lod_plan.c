#include "lod/lod_plan.h"

#include "dimension.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t
clamped_extent(uint64_t shape_d, uint64_t lo, uint64_t scale)
{
  if (lo >= shape_d)
    return 0;
  uint64_t e = shape_d - lo;
  return (e < scale) ? e : scale;
}

uint64_t
morton_rank(int ndim, const uint64_t* shape, const uint64_t* coords, int depth)
{
  int p = ceil_log2(max_shape(ndim, shape));

  for (int d = 0; d < ndim; ++d) {
    int pc = coords[d] > 0 ? ceil_log2(coords[d] + 1) : 0;
    if (pc > p)
      p = pc;
  }

  int total_levels = p + depth;

  uint64_t count = 0;
  uint64_t prefix[LOD_MAX_NDIM] = { 0 };

  for (int level = 0; level < total_levels; ++level) {
    uint64_t scale = 1ull << (total_levels - 1 - level);

    int digit = 0;
    if (level < p) {
      int bit_idx = p - 1 - level;
      for (int d = 0; d < ndim; ++d)
        digit |= (int)((coords[d] >> bit_idx) & 1) << d;
    }

    uint64_t ext[LOD_MAX_NDIM][2];
    for (int d = 0; d < ndim; ++d) {
      for (int b = 0; b < 2; ++b) {
        uint64_t lo = (prefix[d] * 2 + (uint64_t)b) * scale;
        ext[d][b] = clamped_extent(shape[d], lo, scale);
      }
    }

    uint64_t free_prefix[LOD_MAX_NDIM + 1];
    free_prefix[0] = 1;
    for (int d = 0; d < ndim; ++d)
      free_prefix[d + 1] = free_prefix[d] * (ext[d][0] + ext[d][1]);

    uint64_t tight = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      int bit = (digit >> d) & 1;
      if (bit == 1)
        count += tight * ext[d][0] * free_prefix[d];
      tight *= ext[d][bit];
    }

    for (int d = 0; d < ndim; ++d)
      prefix[d] = prefix[d] * 2 + ((digit >> d) & 1);
  }

  return count;
}

uint64_t
lod_span_len(struct lod_span s)
{
  return s.end - s.beg;
}

struct lod_span
lod_spans_at(const struct lod_spans* s, uint64_t i)
{
  return (struct lod_span){
    .beg = i > 0 ? s->ends[i - 1] : 0,
    .end = s->ends[i],
  };
}

// Are all LOD dims already at 1 chunk or fewer?
// Stop generating levels when every dim fits in a single chunk.
static int
all_chunks_le_one(int lod_ndim,
                  const uint64_t* lod_shape,
                  const uint64_t* lod_chunk)
{
  for (int d = 0; d < lod_ndim; ++d) {
    uint64_t nchunks = ceildiv(lod_shape[d], lod_chunk[d]);
    if (nchunks > 1)
      return 0;
  }
  return 1;
}

// Build CSR reduce LUT for the transition from level l to level l+1.
// Flattened: batch_count=1, dst_segment_size covers the entire destination
// level (fixed_dims_count * lod_nelem), indices store absolute offsets within
// the source level. This ensures the CSR output layout matches the scatter
// kernel's ascending-d enumeration of fixed dims.
static int
build_reduce_csr(struct reduce_csr* csr, const struct lod_plan* p, int l)
{
  const struct level_dims* src_ld = &p->levels.level[l];
  const struct level_dims* dst_ld = &p->levels.level[l + 1];
  uint32_t dropped_mask = src_ld->lod_mask & ~dst_ld->lod_mask;

  uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;
  uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;

  csr->batch_count = 1;
  csr->dst_segment_size = dst_total;
  csr->src_lod_count = src_total;
  csr->starts = NULL;
  csr->indices = NULL;

  if (src_total == 0 || dst_total == 0)
    return 0;

  csr->starts = (uint64_t*)calloc(dst_total + 1, sizeof(uint64_t));
  csr->indices = (uint64_t*)malloc(src_total * sizeof(uint64_t));
  if (!csr->starts || !csr->indices) {
    free(csr->starts);
    free(csr->indices);
    csr->starts = NULL;
    csr->indices = NULL;
    return 1;
  }

  // Build src/dst LOD shapes for coordinate math.
  uint64_t src_lod_shape[LOD_MAX_NDIM];
  for (int k = 0; k < src_ld->lod_ndim; ++k)
    src_lod_shape[k] = src_ld->dim[src_ld->lod_to_dim[k]].size;

  uint64_t dst_lod_shape[LOD_MAX_NDIM];
  for (int k = 0; k < dst_ld->lod_ndim; ++k)
    dst_lod_shape[k] = dst_ld->dim[dst_ld->lod_to_dim[k]].size;

  struct src_map
  {
    uint64_t dst_elem;
    uint64_t src_elem;
  };
  struct src_map* map = (struct src_map*)malloc(src_total * sizeof(*map));
  if (!map) {
    free(csr->starts);
    free(csr->indices);
    csr->starts = NULL;
    csr->indices = NULL;
    return 1;
  }

  // Enumerate ALL source elements across all batches.
  uint64_t* counts = csr->starts + 1;
  uint64_t gi = 0;

  for (uint64_t src_batch = 0; src_batch < src_ld->fixed_dims_count;
       ++src_batch) {
    // Decompose src_batch into per-dim coords (indexed by full dim d).
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

    for (uint64_t src_enum = 0; src_enum < src_ld->lod_nelem; ++src_enum) {
      uint64_t src_coords[LOD_MAX_NDIM];
      unravel(src_ld->lod_ndim, src_lod_shape, src_enum, src_coords);

      // Halve LOD coords. Non-dropped → dst LOD, dropped → dst fixed.
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

      // Ravel dst fixed coords in ascending-d order (matches scatter layout).
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
      counts[dst_elem]++;
      gi++;
    }
  }

  // Prefix sum.
  csr->starts[0] = 0;
  for (uint64_t i = 0; i < dst_total; ++i)
    csr->starts[i + 1] += csr->starts[i];

  // Scatter into CSR indices.
  uint64_t* write_pos = (uint64_t*)malloc(dst_total * sizeof(uint64_t));
  if (!write_pos) {
    free(map);
    free(csr->starts);
    free(csr->indices);
    csr->starts = NULL;
    csr->indices = NULL;
    return 1;
  }
  for (uint64_t i = 0; i < dst_total; ++i)
    write_pos[i] = csr->starts[i];

  for (uint64_t i = 0; i < src_total; ++i)
    csr->indices[write_pos[map[i].dst_elem]++] = map[i].src_elem;

  free(write_pos);
  free(map);
  return 0;
}

int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              const uint64_t* chunk_shape,
              uint32_t lod_mask,
              int max_levels,
              int preserve_aspect_ratio)
{
  if (lod_plan_init_shapes(p,
                           ndim,
                           shape,
                           chunk_shape,
                           lod_mask,
                           max_levels,
                           preserve_aspect_ratio))
    return 1;

  int nlod = p->levels.nlod;

  // lod_nelem[k] = product of lod-projected shapes at level k.
  // Uses per-level lod_ndim/lod_to_dim to handle dimension dropping.
  for (int k = 0; k < nlod; ++k) {
    struct level_dims* ld = &p->levels.level[k];
    ld->lod_nelem = 1;
    for (int j = 0; j < ld->lod_ndim; ++j)
      ld->lod_nelem *= ld->dim[ld->lod_to_dim[j]].size;
  }

  // level_spans: cumulative total elements per level (fixed_dims * lod
  // elements). Uses per-level fixed_dims_count to handle dimension dropping.
  p->level_spans.n = (uint64_t)nlod;
  p->level_spans.ends = (uint64_t*)malloc(nlod * sizeof(uint64_t));
  if (!p->level_spans.ends)
    goto Fail;
  {
    uint64_t cumul = 0;
    for (int k = 0; k < nlod; ++k) {
      cumul +=
        p->levels.level[k].fixed_dims_count * p->levels.level[k].lod_nelem;
      p->level_spans.ends[k] = cumul;
    }
  }

  // Build CSR reduce LUTs for each level transition.
  for (int l = 0; l < nlod - 1; ++l) {
    if (build_reduce_csr(&p->reduce[l], p, l))
      goto Fail;
  }

  return 0;
Fail:
  lod_plan_free(p);
  return 1;
}

// Populate fixed-dims decomposition from lod_mask and dim sizes.
static void
level_dims_fill_fixed(struct level_dims* ld, int ndim)
{
  ld->fixed_dims_ndim = 0;
  ld->fixed_dims_count = 1;
  for (int d = 0; d < ndim; ++d) {
    if (!(ld->lod_mask & (1u << d))) {
      ld->fixed_dim_to_dim[ld->fixed_dims_ndim] = d;
      ld->fixed_dims_shape[ld->fixed_dims_ndim] = ld->dim[d].size;
      ld->fixed_dims_ndim++;
      ld->fixed_dims_count *= ld->dim[d].size;
    }
  }
}

// Is any LOD dim at ≤1 chunk?
static int
any_chunks_le_one(int lod_ndim,
                  const uint64_t* lod_shape,
                  const uint64_t* lod_chunk)
{
  for (int d = 0; d < lod_ndim; ++d) {
    uint64_t nchunks = ceildiv(lod_shape[d], lod_chunk[d]);
    if (nchunks <= 1)
      return 1;
  }
  return 0;
}

int
lod_plan_init_shapes(struct lod_plan* p,
                     int ndim,
                     const uint64_t* shape,
                     const uint64_t* chunk_shape,
                     uint32_t lod_mask,
                     int max_levels,
                     int preserve_aspect_ratio)
{
  if (ndim <= 0 || ndim > LOD_MAX_NDIM)
    return 1;
  memset(p, 0, sizeof(*p));
  p->ndim = ndim;
  p->lod_mask = lod_mask;

  for (int d = 0; d < ndim; ++d) {
    if (lod_mask & (1 << d)) {
      p->lod_to_dim[p->lod_ndim++] = d;
    } else {
      p->fixed_dim_to_dim[p->fixed_dims_ndim] = d;
      p->fixed_dims_shape[p->fixed_dims_ndim] = shape[d];
      p->fixed_dims_ndim++;
    }
  }
  p->fixed_dims_count = 1;
  for (int k = 0; k < p->fixed_dims_ndim; ++k)
    p->fixed_dims_count *= p->fixed_dims_shape[k];

  level_dims_set_shape(&p->levels.level[0], ndim, shape);

  // Per-level LOD projection: track which dims are still active (> 1 chunk).
  // Dims drop from LOD when they reach chunk_size (≤1 chunk).
  int cur_lod_ndim = p->lod_ndim;
  int cur_to_dim[LOD_MAX_NDIM];
  uint64_t cur_shape[LOD_MAX_NDIM];
  uint64_t cur_chunk[LOD_MAX_NDIM];
  for (int k = 0; k < cur_lod_ndim; ++k) {
    cur_to_dim[k] = p->lod_to_dim[k];
    cur_shape[k] = p->levels.level[0].dim[p->lod_to_dim[k]].size;
    cur_chunk[k] = chunk_shape ? chunk_shape[p->lod_to_dim[k]] : 1;
  }

  // Fill L0 per-level info.
  p->levels.level[0].lod_mask = lod_mask;
  p->levels.level[0].lod_ndim = cur_lod_ndim;
  memcpy(p->levels.level[0].lod_to_dim,
         cur_to_dim,
         sizeof(int) * (size_t)cur_lod_ndim);
  level_dims_fill_fixed(&p->levels.level[0], ndim);

  int nlod = 1;
  while (nlod < max_levels && cur_lod_ndim > 0 &&
         !all_chunks_le_one(cur_lod_ndim, cur_shape, cur_chunk)) {
    // Halve active LOD dims with clamping at chunk_size.
    uint64_t next_shape[LOD_MAX_NDIM];
    for (int k = 0; k < cur_lod_ndim; ++k) {
      uint64_t half = (cur_shape[k] + 1) / 2;
      next_shape[k] = half > cur_chunk[k] ? half : cur_chunk[k];
    }

    // Write level shapes: copy from previous, then update LOD dims.
    level_dims_copy_sizes(
      &p->levels.level[nlod], &p->levels.level[nlod - 1], ndim);
    for (int k = 0; k < cur_lod_ndim; ++k)
      p->levels.level[nlod].dim[cur_to_dim[k]].size = next_shape[k];

    // Per-level LOD info: the LOD state at this level (pre-drop).
    uint32_t level_mask = 0;
    for (int k = 0; k < cur_lod_ndim; ++k)
      level_mask |= (1u << cur_to_dim[k]);
    p->levels.level[nlod].lod_mask = level_mask;
    p->levels.level[nlod].lod_ndim = cur_lod_ndim;
    memcpy(p->levels.level[nlod].lod_to_dim,
           cur_to_dim,
           sizeof(int) * (size_t)cur_lod_ndim);
    level_dims_fill_fixed(&p->levels.level[nlod], ndim);

    ++nlod;

    // preserve_aspect_ratio: stop when any dim reaches ≤1 chunk after halving.
    // The level where the dim reached chunk_size IS included; no further
    // levels.
    if (preserve_aspect_ratio &&
        any_chunks_le_one(cur_lod_ndim, next_shape, cur_chunk))
      break;

    // Drop dims that reached ≤1 chunk for subsequent levels.
    int next_ndim = 0;
    int next_to_dim[LOD_MAX_NDIM];
    uint64_t next_chunk[LOD_MAX_NDIM];
    uint64_t next_active[LOD_MAX_NDIM];
    for (int k = 0; k < cur_lod_ndim; ++k) {
      if (ceildiv(next_shape[k], cur_chunk[k]) > 1) {
        next_to_dim[next_ndim] = cur_to_dim[k];
        next_chunk[next_ndim] = cur_chunk[k];
        next_active[next_ndim] = next_shape[k];
        next_ndim++;
      }
    }

    cur_lod_ndim = next_ndim;
    memcpy(cur_to_dim, next_to_dim, sizeof(int) * (size_t)next_ndim);
    memcpy(cur_chunk, next_chunk, sizeof(uint64_t) * (size_t)next_ndim);
    memcpy(cur_shape, next_active, sizeof(uint64_t) * (size_t)next_ndim);
  }
  p->levels.nlod = nlod;

  // chunk_size is constant across all levels (partial chunks are padded).
  for (int lv = 0; lv < nlod; ++lv) {
    for (int d = 0; d < ndim; ++d) {
      assert(!chunk_shape || chunk_shape[d] <= UINT32_MAX);
      p->levels.level[lv].dim[d].chunk_size =
        chunk_shape ? (uint32_t)chunk_shape[d] : 1;
    }
  }

  // Verify plan-level L0 convenience fields match per-level L0 fields.
  assert(p->lod_mask == p->levels.level[0].lod_mask);
  assert(p->lod_ndim == p->levels.level[0].lod_ndim);
  for (int k = 0; k < p->lod_ndim; ++k)
    assert(p->lod_to_dim[k] == p->levels.level[0].lod_to_dim[k]);
  assert(p->fixed_dims_count == p->levels.level[0].fixed_dims_count);

  return 0;
}

static void
dims_lod_params(const struct dimension* dims,
                uint8_t rank,
                uint8_t n_append,
                uint64_t* shape,
                uint64_t* chunk_shape,
                uint32_t* lod_mask)
{
  for (int d = 0; d < rank; ++d) {
    shape[d] = (dims[d].size == 0) ? dims[d].chunk_size : dims[d].size;
    chunk_shape[d] = dims[d].chunk_size;
  }
  *lod_mask = 0;
  for (int d = n_append; d < rank; ++d)
    if (dims[d].downsample)
      *lod_mask |= (1u << d);
}

static void
fill_shard_geometry(struct lod_plan* p,
                    const struct dimension* dims,
                    uint8_t rank,
                    uint8_t n_append)
{
  int nlod = p->levels.nlod;
  for (int lv = 0; lv < nlod; ++lv) {
    uint64_t cps[LOD_MAX_NDIM];
    for (int d = 0; d < rank; ++d)
      cps[d] = dims[d].chunks_per_shard;
    dim_extent_compute_shards(p->levels.level[lv].dim, rank, n_append, cps);
  }
}

int
lod_plan_init_from_dims(struct lod_plan* p,
                        const struct dimension* dims,
                        uint8_t rank,
                        int max_levels,
                        int preserve_aspect_ratio)
{
  uint64_t shape[LOD_MAX_NDIM];
  uint64_t chunk_shape[LOD_MAX_NDIM];
  uint32_t lod_mask;
  uint8_t na = dims_n_append(dims, rank);
  dims_lod_params(dims, rank, na, shape, chunk_shape, &lod_mask);
  if (lod_plan_init(p,
                    rank,
                    shape,
                    chunk_shape,
                    lod_mask,
                    max_levels,
                    preserve_aspect_ratio))
    return 1;
  fill_shard_geometry(p, dims, rank, na);
  return 0;
}

int
lod_plan_init_from_epoch_dims(struct lod_plan* p,
                              const struct dimension* dims,
                              uint8_t rank,
                              uint8_t n_append,
                              int max_levels,
                              int preserve_aspect_ratio)
{
  assert(n_append > 0 && n_append <= rank);
  uint64_t shape[LOD_MAX_NDIM];
  uint64_t chunk_shape[LOD_MAX_NDIM];
  uint32_t lod_mask;
  dims_lod_params(dims, rank, n_append, shape, chunk_shape, &lod_mask);
  for (int d = 0; d < n_append; ++d)
    shape[d] = dims[d].chunk_size;
  if (lod_plan_init(p,
                    rank,
                    shape,
                    chunk_shape,
                    lod_mask,
                    max_levels,
                    preserve_aspect_ratio))
    return 1;
  fill_shard_geometry(p, dims, rank, n_append);
  return 0;
}

void
level_dims_get_shape(const struct level_dims* ld, int ndim, uint64_t* out)
{
  for (int d = 0; d < ndim; ++d)
    out[d] = ld->dim[d].size;
}

void
level_dims_set_shape(struct level_dims* ld, int ndim, const uint64_t* shape)
{
  for (int d = 0; d < ndim; ++d)
    ld->dim[d].size = shape[d];
}

void
level_dims_copy_sizes(struct level_dims* dst,
                      const struct level_dims* src,
                      int ndim)
{
  for (int d = 0; d < ndim; ++d)
    dst->dim[d].size = src->dim[d].size;
}

uint64_t
lod_plan_lod_shape(const struct lod_plan* p, int lv, int k)
{
  return p->levels.level[lv].dim[p->levels.level[lv].lod_to_dim[k]].size;
}

void
lod_plan_fill_lod_shapes(const struct lod_plan* p, int lv, uint64_t* dst)
{
  const struct level_dims* ld = &p->levels.level[lv];
  for (int k = 0; k < ld->lod_ndim; ++k)
    dst[k] = ld->dim[ld->lod_to_dim[k]].size;
}

void
lod_plan_free(struct lod_plan* p)
{
  if (!p)
    return;
  free(p->level_spans.ends);
  for (int l = 0; l < LOD_MAX_LEVELS; ++l) {
    free(p->reduce[l].starts);
    free(p->reduce[l].indices);
  }
  memset(p, 0, sizeof(*p));
}

uint64_t
dim_extent_compute_shards(struct dim_extent* dims,
                          int ndim,
                          int n_append,
                          const uint64_t* config_cps)
{
  uint64_t shard_inner_count = 1;
  for (int d = 0; d < ndim; ++d) {
    uint64_t s = dims[d].size;
    uint32_t cs = dims[d].chunk_size;
    uint64_t cc = (s == 0) ? 1 : ceildiv(s, cs);
    dims[d].chunk_count = cc;
    uint64_t cps = config_cps[d];
    if (cps == 0)
      cps = cc;
    if (s > 0 && cc < cps)
      cps = cc;
    assert(cps <= UINT32_MAX);
    dims[d].chunks_per_shard = (uint32_t)cps;
    dims[d].shard_count = ceildiv(cc, cps);
    if (d >= n_append)
      shard_inner_count *= dims[d].shard_count;
  }
  return shard_inner_count;
}

uint64_t
dims_compute_shard_geometry(const struct dimension* dims,
                            uint8_t rank,
                            uint64_t* shard_counts,
                            uint64_t* chunks_per_shard)
{
  struct dim_extent de[LOD_MAX_NDIM];
  for (int d = 0; d < rank; ++d) {
    de[d].size = dims[d].size;
    de[d].chunk_size = (uint32_t)dims[d].chunk_size;
  }
  uint64_t cps[LOD_MAX_NDIM];
  for (int d = 0; d < rank; ++d)
    cps[d] = dims[d].chunks_per_shard;
  uint8_t na = dims_n_append(dims, rank);
  uint64_t sic = dim_extent_compute_shards(de, rank, na, cps);
  for (int d = 0; d < rank; ++d) {
    shard_counts[d] = de[d].shard_count;
    chunks_per_shard[d] = de[d].chunks_per_shard;
  }
  return sic;
}

// --- Internal dimension utilities ---

uint8_t
dims_n_append(const struct dimension* dims, uint8_t rank)
{
  uint8_t max_n = 0;
  for (uint8_t d = 0; d < rank; ++d) {
    if (dims[d].chunk_size != 1)
      break;
    max_n = d + 1;
  }
  if (max_n <= 1)
    return 1;
  // Only the rightmost append dim may be accumulator-downsampled.
  // If a dim in the prefix has downsample, it becomes the rightmost append dim.
  for (uint8_t d = 0; d < max_n; ++d) {
    if (dims[d].downsample)
      return d + 1;
  }
  return max_n;
}

int
dims_validate(const struct dimension* dims, uint8_t rank)
{
  CHECK(Fail, dims);
  CHECK(Fail, rank > 0 && rank <= HALF_MAX_RANK);

  for (int d = 0; d < rank; ++d) {
    if (dims[d].chunk_size == 0) {
      log_error("dims[%d].chunk_size must be > 0", d);
      goto Fail;
    }
  }

  for (int d = 1; d < rank; ++d) {
    if (dims[d].size == 0) {
      log_error("dims[%d].size must be > 0 (only dim 0 may be unbounded)", d);
      goto Fail;
    }
  }

  if (dims[0].size == 0 && dims[0].chunks_per_shard == 0) {
    log_error("dims[0].size=0 (unbounded) requires chunks_per_shard > 0");
    goto Fail;
  }

  {
    uint8_t na = dims_n_append(dims, rank);
    for (int d = 0; d < na; ++d) {
      if (dims[d].storage_position != d) {
        log_error("dims[%d].storage_position must be %d (append dim)", d, d);
        goto Fail;
      }
    }
    uint32_t seen = 0;
    for (int d = 0; d < rank; ++d) {
      uint8_t j = dims[d].storage_position;
      if (j >= rank || (seen & (1u << j))) {
        log_error("invalid storage_position permutation at dims[%d]", d);
        goto Fail;
      }
      seen |= (1u << j);
    }
  }

  return 0;
Fail:
  return 1;
}

void
dims_print(const struct dimension* dims, uint8_t rank)
{
  fprintf(stderr,
          "dim  name  %10s  %10s  %8s  %6s  %8s  storage  ds\n",
          "size",
          "chunk",
          "chunks",
          "cps",
          "shards");
  uint8_t na = dims_n_append(dims, rank);
  uint64_t chunk_elements = 1;
  uint64_t chunks_per_epoch = 1;
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t tc = ceildiv(dims[i].size, dims[i].chunk_size);
    uint64_t cps = dims[i].chunks_per_shard ? dims[i].chunks_per_shard : tc;
    uint64_t sc = ceildiv(tc, cps);
    fprintf(stderr,
            "%3d  %-4s  %10llu  %10llu  %8llu  %6llu  %8llu  %7d  %s\n",
            i,
            dims[i].name ? dims[i].name : "?",
            (unsigned long long)dims[i].size,
            (unsigned long long)dims[i].chunk_size,
            (unsigned long long)tc,
            (unsigned long long)cps,
            (unsigned long long)sc,
            (int)dims[i].storage_position,
            dims[i].downsample ? "Y" : ".");
    chunk_elements *= dims[i].chunk_size;
    if (i >= na)
      chunks_per_epoch *= tc;
  }
  double epoch_elements = (double)chunks_per_epoch * (double)chunk_elements;
  fprintf(stderr,
          "chunk_elements: %llu  chunks/epoch: %llu  epoch_elements: %.3g\n",
          (unsigned long long)chunk_elements,
          (unsigned long long)chunks_per_epoch,
          epoch_elements);
}
