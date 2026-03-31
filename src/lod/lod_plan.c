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

  // lod_nelem[k] = product of lod_shapes[k] — the total element count at each
  // LOD level. Child levels are strictly smaller than L0, so if L0 fits in
  // uint64_t the rest do too.
  for (int k = 0; k < p->nlod; ++k) {
    p->lod_nelem[k] = 1;
    for (int d = 0; d < p->lod_ndim; ++d)
      p->lod_nelem[k] *= p->lod_shapes[k][d];
  }

  p->lod_levels.n = (uint64_t)p->nlod;
  p->lod_levels.ends = (uint64_t*)malloc(p->nlod * sizeof(uint64_t));
  if (!p->lod_levels.ends)
    goto Fail;
  p->lod_levels.ends[0] = p->lod_nelem[0];
  for (int k = 1; k < p->nlod; ++k)
    p->lod_levels.ends[k] = p->lod_levels.ends[k - 1] + p->lod_nelem[k];

  p->levels.n = (uint64_t)p->nlod;
  p->levels.ends = (uint64_t*)malloc(p->nlod * sizeof(uint64_t));
  if (!p->levels.ends)
    goto Fail;
  for (int k = 0; k < p->nlod; ++k)
    p->levels.ends[k] = p->batch_count * p->lod_levels.ends[k];

  {
    uint64_t total_ends =
      p->lod_levels.ends[p->nlod - 1] - p->lod_levels.ends[0];
    if (total_ends > 0) {
      p->ends = (uint64_t*)malloc(total_ends * sizeof(uint64_t));
      if (!p->ends)
        goto Fail;
      for (int l = 0; l < p->nlod - 1; ++l) {
        struct lod_span seg = lod_segment(p, l);
        lod_fill_ends(p->lod_ndim,
                      p->lod_shapes[l],
                      p->lod_shapes[l + 1],
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
  memset(p, 0, sizeof(*p));
  p->ndim = ndim;
  p->lod_mask = lod_mask;

  for (int d = 0; d < ndim; ++d) {
    if (lod_mask & (1 << d)) {
      p->lod_map[p->lod_ndim++] = d;
    } else {
      p->batch_map[p->batch_ndim] = d;
      p->batch_shape[p->batch_ndim] = shape[d];
      p->batch_ndim++;
    }
  }
  p->batch_count = 1;
  for (int k = 0; k < p->batch_ndim; ++k)
    p->batch_count *= p->batch_shape[k];

  memcpy(p->shapes[0], shape, (size_t)ndim * sizeof(uint64_t));
  for (int k = 0; k < p->lod_ndim; ++k)
    p->lod_shapes[0][k] = shape[p->lod_map[k]];

  uint64_t lod_chunk[LOD_MAX_NDIM];
  for (int k = 0; k < p->lod_ndim; ++k)
    lod_chunk[k] = chunk_shape ? chunk_shape[p->lod_map[k]] : 1;

  p->nlod = 1;
  while (p->nlod < max_levels &&
         !all_chunks_le_one(p->lod_ndim, p->lod_shapes[p->nlod - 1],
                            lod_chunk)) {
    for (int k = 0; k < p->lod_ndim; ++k)
      p->lod_shapes[p->nlod][k] = (p->lod_shapes[p->nlod - 1][k] + 1) / 2;
    memcpy(p->shapes[p->nlod],
           p->shapes[p->nlod - 1],
           (size_t)ndim * sizeof(uint64_t));
    for (int k = 0; k < p->lod_ndim; ++k)
      p->shapes[p->nlod][p->lod_map[k]] = p->lod_shapes[p->nlod][k];
    ++p->nlod;
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
  return lod_plan_init(p, rank, shape, chunk_shape, lod_mask, max_levels);
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
  return lod_plan_init(p, rank, shape, chunk_shape, lod_mask, max_levels);
}

void
lod_plan_free(struct lod_plan* p)
{
  if (!p)
    return;
  free(p->levels.ends);
  free(p->lod_levels.ends);
  free(p->ends);
  memset(p, 0, sizeof(*p));
}

void
shard_geometry_compute(struct shard_geometry* g,
                       uint8_t rank,
                       uint8_t n_append,
                       const uint64_t* shape,
                       const uint64_t* chunk_size,
                       const uint64_t* chunks_per_shard)
{
  assert(n_append > 0 && n_append <= rank);
  g->shard_inner_count = 1;
  for (int d = 0; d < rank; ++d) {
    g->chunk_count[d] = (shape[d] == 0) ? 1 : ceildiv(shape[d], chunk_size[d]);
    uint64_t cps = chunks_per_shard[d];
    g->chunks_per_shard[d] = (cps == 0) ? g->chunk_count[d] : cps;
    g->shard_count[d] = ceildiv(g->chunk_count[d], g->chunks_per_shard[d]);
    if (d >= n_append)
      g->shard_inner_count *= g->shard_count[d];
  }
}
