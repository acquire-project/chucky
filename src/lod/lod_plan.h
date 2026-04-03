#pragma once

#include "defs.limits.h"
#include <stdint.h>

struct dimension;

struct lod_span
{
  uint64_t beg, end;
};

struct lod_spans
{
  uint64_t* ends;
  uint64_t n;
};

// WARNING: This struct is ~49 KB. Callers that stack-allocate it should be
// aware of stack usage, especially in deeply recursive or thread-heavy code.
struct lod_plan
{
  int ndim;
  int nlod;

  // Per-level full shape (all dims). Populated by init_shapes and above.
  uint64_t shapes[LOD_MAX_LEVELS][LOD_MAX_NDIM];

  // Per-level chunk sizes, clamped to min(chunk_size, level_shape[d]).
  // Populated by init_shapes and above.
  uint32_t chunk_sizes[LOD_MAX_LEVELS][LOD_MAX_NDIM];

  // Per-level chunks_per_shard, clamped to min(config_cps, chunk_count).
  // Only populated by _from_dims / _from_epoch_dims (needs dims[d].chunks_per_shard).
  uint32_t chunks_per_shard[LOD_MAX_LEVELS][LOD_MAX_NDIM];

  uint32_t lod_mask;
  int lod_ndim;
  int lod_map[LOD_MAX_NDIM];
  int batch_ndim;
  int batch_map[LOD_MAX_NDIM];
  uint64_t batch_shape[LOD_MAX_NDIM];
  uint64_t batch_count;

  uint64_t lod_shapes[LOD_MAX_LEVELS][LOD_MAX_NDIM];
  uint64_t lod_nelem[LOD_MAX_LEVELS];

  // Heap-allocated arrays. Populated by init and above (not by init_shapes).
  struct lod_spans levels;
  struct lod_spans lod_levels;
  uint64_t* ends;
};

uint64_t
lod_span_len(struct lod_span s);

uint64_t
morton_rank(int ndim, const uint64_t* shape, const uint64_t* coords, int depth);

struct lod_span
lod_spans_at(const struct lod_spans* s, uint64_t i);

// Return the segment within the ends array for the given level.
struct lod_span
lod_segment(const struct lod_plan* p, int level);

// Initialize a plan. Returns 0 on success, non-zero on failure.
// chunk_shape: per-dimension chunk sizes (may be NULL). When provided,
// levels stop before any LOD dimension would drop below its chunk size.
// Populates everything from init_shapes plus heap-allocated arrays
// (ends, levels, lod_levels, lod_nelem).
// Does NOT populate chunks_per_shard (use _from_dims variants for that).
int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              const uint64_t* chunk_shape,
              uint32_t lod_mask,
              int max_levels);

// Compute only nlod and per-level shapes (no ends/counts/levels).
// Use when you only need the level geometry (e.g. for metadata).
// Populates: ndim, nlod, shapes, chunk_sizes, lod_mask, lod_ndim,
// lod_map, batch_ndim, batch_map, batch_shape, batch_count, lod_shapes.
int
lod_plan_init_shapes(struct lod_plan* p,
                     int ndim,
                     const uint64_t* shape,
                     const uint64_t* chunk_shape,
                     uint32_t lod_mask,
                     int max_levels);

// Compute LOD plan from dimension array. Extracts shapes, chunk shapes,
// and LOD mask (dims 1+ with downsample=1) from dimensions.
// Uses chunk_size as placeholder shape for unbounded dims (size==0).
// Populates everything from init plus chunks_per_shard.
int
lod_plan_init_from_dims(struct lod_plan* p,
                        const struct dimension* dims,
                        uint8_t rank,
                        int max_levels);

// Like _from_dims, but overrides shape[d] = dims[d].chunk_size for
// d < n_append (epoch-split). Use for the streaming path where append
// dims are split into per-epoch chunks.
// Populates everything from init plus chunks_per_shard.
int
lod_plan_init_from_epoch_dims(struct lod_plan* p,
                              const struct dimension* dims,
                              uint8_t rank,
                              uint8_t n_append,
                              int max_levels);

void
lod_plan_free(struct lod_plan* p);

// Shard geometry computed from array shape, chunk sizes, and shard config.
struct shard_geometry
{
  uint64_t chunk_count[HALF_MAX_RANK];
  uint64_t chunks_per_shard[HALF_MAX_RANK];
  uint64_t shard_count[HALF_MAX_RANK];
  uint64_t shard_inner_count; // prod(shard_count[d] for d >= n_append)
};

// Compute shard geometry from explicit shape, chunk_size, and
// chunks_per_shard arrays (each rank elements).
// chunks_per_shard[d] == 0 means all chunks along that dimension.
// n_append: number of append dims. shard_inner_count = prod(shard_count[d]
// for d >= n_append).
void
shard_geometry_compute(struct shard_geometry* g,
                       uint8_t rank,
                       uint8_t n_append,
                       const uint64_t* shape,
                       const uint64_t* chunk_size,
                       const uint64_t* chunks_per_shard);
