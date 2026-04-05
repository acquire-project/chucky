#include "lod/lod_plan.h"

#include "dimension.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <assert.h>
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

static void
coords_morton_next(int ndim, int p, uint64_t* coords)
{
  for (int bit = 0; bit < p; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      uint64_t mask = 1ull << bit;
      coords[d] ^= mask;
      if (coords[d] & mask)
        return;
    }
  }
  memset(coords, 0, (size_t)ndim * sizeof(uint64_t));
  coords[0] = 1ull << p;
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

static void
lod_fill_ends(int ndim,
              const uint64_t* child_shape,
              const uint64_t* parent_shape,
              uint64_t n_parents,
              uint64_t* ends)
{
  int p = ceil_log2(max_shape(ndim, parent_shape));

  uint64_t coords[LOD_MAX_NDIM];
  uint64_t next[LOD_MAX_NDIM];
  for (uint64_t j = 0; j < n_parents; ++j) {
    unravel(ndim, parent_shape, j, coords);
    uint64_t pos = morton_rank(ndim, parent_shape, coords, 0);

    memcpy(next, coords, (size_t)ndim * sizeof(uint64_t));
    coords_morton_next(ndim, p, next);
    uint64_t val = morton_rank(ndim, child_shape, next, 1);

    ends[pos] = val;
  }
}

struct lod_span
lod_segment(const struct lod_plan* p, int level)
{
  struct lod_span next = lod_spans_at(&p->lod_levels, level + 1);
  uint64_t base = p->lod_levels.ends[0];
  return (struct lod_span){ .beg = next.beg - base, .end = next.end - base };
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

int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              const uint64_t* chunk_shape,
              uint32_t lod_mask,
              int max_levels)
{
  if (lod_plan_init_shapes(p, ndim, shape, chunk_shape, lod_mask, max_levels))
    return 1;

  int nlod = p->levels.nlod;

  // lod_nelem[k] = product of lod-projected shapes at level k.
  for (int k = 0; k < nlod; ++k) {
    p->levels.level[k].lod_nelem = 1;
    for (int d = 0; d < p->lod_ndim; ++d)
      p->levels.level[k].lod_nelem *= lod_plan_lod_shape(p, k, d);
  }

  p->lod_levels.n = (uint64_t)nlod;
  p->lod_levels.ends = (uint64_t*)malloc(nlod * sizeof(uint64_t));
  if (!p->lod_levels.ends)
    goto Fail;
  p->lod_levels.ends[0] = p->levels.level[0].lod_nelem;
  for (int k = 1; k < nlod; ++k)
    p->lod_levels.ends[k] =
      p->lod_levels.ends[k - 1] + p->levels.level[k].lod_nelem;

  p->level_spans.n = (uint64_t)nlod;
  p->level_spans.ends = (uint64_t*)malloc(nlod * sizeof(uint64_t));
  if (!p->level_spans.ends)
    goto Fail;
  for (int k = 0; k < nlod; ++k)
    p->level_spans.ends[k] = p->fixed_dims_count * p->lod_levels.ends[k];

  {
    uint64_t total_ends = p->lod_levels.ends[nlod - 1] - p->lod_levels.ends[0];
    if (total_ends > 0) {
      p->ends = (uint64_t*)malloc(total_ends * sizeof(uint64_t));
      if (!p->ends)
        goto Fail;
      for (int l = 0; l < nlod - 1; ++l) {
        uint64_t child_lod[LOD_MAX_NDIM], parent_lod[LOD_MAX_NDIM];
        lod_plan_fill_lod_shapes(p, l, child_lod);
        lod_plan_fill_lod_shapes(p, l + 1, parent_lod);
        struct lod_span seg = lod_segment(p, l);
        lod_fill_ends(p->lod_ndim,
                      child_lod,
                      parent_lod,
                      lod_span_len(seg),
                      p->ends + seg.beg);
      }
    }
  }

  return 0;
Fail:
  lod_plan_free(p);
  return 1;
}

int
lod_plan_init_shapes(struct lod_plan* p,
                     int ndim,
                     const uint64_t* shape,
                     const uint64_t* chunk_shape,
                     uint32_t lod_mask,
                     int max_levels)
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

  uint64_t lod_chunk[LOD_MAX_NDIM];
  for (int k = 0; k < p->lod_ndim; ++k)
    lod_chunk[k] = chunk_shape ? chunk_shape[p->lod_to_dim[k]] : 1;

  // Derive lod_shape for current level from sizes + lod_to_dim.
  uint64_t cur_lod[LOD_MAX_NDIM];
  for (int k = 0; k < p->lod_ndim; ++k)
    cur_lod[k] = p->levels.level[0].dim[p->lod_to_dim[k]].size;

  int nlod = 1;
  while (nlod < max_levels &&
         !all_chunks_le_one(p->lod_ndim, cur_lod, lod_chunk)) {
    uint64_t next_lod[LOD_MAX_NDIM];
    for (int k = 0; k < p->lod_ndim; ++k)
      next_lod[k] = (cur_lod[k] + 1) / 2;
    level_dims_copy_sizes(
      &p->levels.level[nlod], &p->levels.level[nlod - 1], ndim);
    for (int k = 0; k < p->lod_ndim; ++k)
      p->levels.level[nlod].dim[p->lod_to_dim[k]].size = next_lod[k];
    memcpy(cur_lod, next_lod, (size_t)p->lod_ndim * sizeof(uint64_t));
    ++nlod;
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
                        int max_levels)
{
  uint64_t shape[LOD_MAX_NDIM];
  uint64_t chunk_shape[LOD_MAX_NDIM];
  uint32_t lod_mask;
  uint8_t na = dims_n_append(dims, rank);
  dims_lod_params(dims, rank, na, shape, chunk_shape, &lod_mask);
  if (lod_plan_init(p, rank, shape, chunk_shape, lod_mask, max_levels))
    return 1;
  fill_shard_geometry(p, dims, rank, na);
  return 0;
}

int
lod_plan_init_from_epoch_dims(struct lod_plan* p,
                              const struct dimension* dims,
                              uint8_t rank,
                              uint8_t n_append,
                              int max_levels)
{
  assert(n_append > 0 && n_append <= rank);
  uint64_t shape[LOD_MAX_NDIM];
  uint64_t chunk_shape[LOD_MAX_NDIM];
  uint32_t lod_mask;
  dims_lod_params(dims, rank, n_append, shape, chunk_shape, &lod_mask);
  for (int d = 0; d < n_append; ++d)
    shape[d] = dims[d].chunk_size;
  if (lod_plan_init(p, rank, shape, chunk_shape, lod_mask, max_levels))
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
  return p->levels.level[lv].dim[p->lod_to_dim[k]].size;
}

void
lod_plan_fill_lod_shapes(const struct lod_plan* p, int lv, uint64_t* dst)
{
  for (int k = 0; k < p->lod_ndim; ++k)
    dst[k] = p->levels.level[lv].dim[p->lod_to_dim[k]].size;
}

void
lod_plan_free(struct lod_plan* p)
{
  if (!p)
    return;
  free(p->level_spans.ends);
  free(p->lod_levels.ends);
  free(p->ends);
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
