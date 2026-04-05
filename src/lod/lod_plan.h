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

// Per-dimension array geometry at a single LOD level.
struct dim_extent
{
  uint64_t size;
  uint32_t chunk_size;
  uint32_t chunks_per_shard;
  uint64_t chunk_count; // ceildiv(size, chunk_size)
  uint64_t shard_count; // ceildiv(chunk_count, chunks_per_shard)
};

// Per-level array geometry (one per LOD level).
struct level_dims
{
  struct dim_extent dim[LOD_MAX_NDIM];
  uint64_t lod_nelem;    // product of lod-projected sizes
  uint64_t chunk_count;  // total chunks at this level
  uint64_t chunk_offset; // cumulative offset into chunk pool
};

// Aggregate level geometry across all LOD levels.
// Passed standalone to flush/aggregate functions.
struct level_geometry
{
  int nlod;
  int enable_multiscale;
  uint64_t total_chunks;
  struct level_dims level[LOD_MAX_LEVELS];
};

struct lod_plan
{
  int ndim;
  struct level_geometry levels;

  uint32_t lod_mask;
  int lod_ndim;
  int lod_to_dim[LOD_MAX_NDIM]; // lod dim index -> full dim index

  int fixed_dims_ndim;
  int fixed_dim_to_dim[LOD_MAX_NDIM]; // fixed dim index -> full dim index
  uint64_t fixed_dims_shape[LOD_MAX_NDIM];
  uint64_t fixed_dims_count;

  // Heap-allocated arrays. Populated by init and above (not by init_shapes).
  struct lod_spans level_spans;
  struct lod_spans lod_levels;
  uint64_t* ends;
};

// Resolve chunks_per_shard and compute chunk_count + shard_count
// for each dim_extent. config_cps[d]==0 means all chunks along dim d.
// Returns shard_inner_count = prod(shard_count[d] for d >= n_append).
uint64_t
dim_extent_compute_shards(struct dim_extent* dims,
                          int ndim,
                          int n_append,
                          const uint64_t* config_cps);

// Extract sizes from level_dims into a flat array.
void
level_dims_get_shape(const struct level_dims* ld, int ndim, uint64_t* out);

// Set sizes from a flat array.
void
level_dims_set_shape(struct level_dims* ld, int ndim, const uint64_t* shape);

// Copy sizes from one level to another.
void
level_dims_copy_sizes(struct level_dims* dst,
                      const struct level_dims* src,
                      int ndim);

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
// (ends, level_spans, lod_levels).
// Does NOT populate chunks_per_shard (use _from_dims variants for that).
int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              const uint64_t* chunk_shape,
              uint32_t lod_mask,
              int max_levels);

// Compute only nlod and per-level shapes (no ends/counts/level_spans).
// Use when you only need the level geometry (e.g. for metadata).
// Populates: ndim, nlod, level[].dim[].size, level[].dim[].chunk_size,
// lod_mask, lod_ndim, lod_to_dim, fixed_dims_*.
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
// Populates everything from init plus level[].dim[].chunks_per_shard.
int
lod_plan_init_from_dims(struct lod_plan* p,
                        const struct dimension* dims,
                        uint8_t rank,
                        int max_levels);

// Like _from_dims, but overrides shape[d] = dims[d].chunk_size for
// d < n_append (epoch-split). Use for the streaming path where append
// dims are split into per-epoch chunks.
// Populates everything from init plus level[].dim[].chunks_per_shard.
int
lod_plan_init_from_epoch_dims(struct lod_plan* p,
                              const struct dimension* dims,
                              uint8_t rank,
                              uint8_t n_append,
                              int max_levels);

void
lod_plan_free(struct lod_plan* p);

// Get lod size for level lv, lod dimension k.
uint64_t
lod_plan_lod_shape(const struct lod_plan* p, int lv, int k);

// Fill dst[0..lod_ndim-1] with projected lod sizes for level lv.
void
lod_plan_fill_lod_shapes(const struct lod_plan* p, int lv, uint64_t* dst);
