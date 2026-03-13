#pragma once

#include <stdint.h>

#define LOD_MAX_NDIM 64
#define LOD_MAX_LEVELS 32

struct lod_span
{
  uint64_t beg, end;
};

struct lod_spans
{
  uint64_t* ends;
  uint64_t n;
};

struct lod_plan
{
  int ndim;
  int nlod;
  uint64_t shapes[LOD_MAX_LEVELS][LOD_MAX_NDIM];

  uint32_t lod_mask;
  int lod_ndim;
  int lod_map[LOD_MAX_NDIM];
  int batch_ndim;
  int batch_map[LOD_MAX_NDIM];
  uint64_t batch_shape[LOD_MAX_NDIM];
  uint64_t batch_count;

  uint64_t lod_shapes[LOD_MAX_LEVELS][LOD_MAX_NDIM];
  uint64_t lod_counts[LOD_MAX_LEVELS];

  struct lod_spans levels;
  struct lod_spans lod_levels;
  uint64_t* ends;
};

uint64_t
lod_span_len(struct lod_span s);

uint64_t
morton_rank(int ndim,
            const uint64_t* shape,
            const uint64_t* coords,
            int depth);

struct lod_span
lod_spans_at(const struct lod_spans* s, uint64_t i);

// Return the segment within the ends array for the given level.
struct lod_span
lod_segment(const struct lod_plan* p, int level);

// Initialize a plan. Returns 0 on success, non-zero on failure.
// tile_shape: per-dimension tile sizes (may be NULL). When provided,
// levels stop before any LOD dimension would drop below its tile size.
// exclude_dim0: when set, dim 0 is removed from lod_mask and treated as a
// batch dimension (temporal accumulation handled separately).
int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              const uint64_t* tile_shape,
              uint32_t lod_mask,
              int max_levels,
              int exclude_dim0);

// Compute only nlod and per-level shapes (no ends/counts/levels).
// Use when you only need the level geometry (e.g. for metadata).
int
lod_plan_init_shapes(struct lod_plan* p,
                     int ndim,
                     const uint64_t* shape,
                     const uint64_t* tile_shape,
                     uint32_t lod_mask,
                     int max_levels,
                     int exclude_dim0);

void
lod_plan_free(struct lod_plan* p);

