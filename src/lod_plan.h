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
  uint64_t lod_nelem[LOD_MAX_LEVELS];

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
int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              const uint64_t* chunk_shape,
              uint32_t lod_mask,
              int max_levels);

// Compute only nlod and per-level shapes (no ends/counts/levels).
// Use when you only need the level geometry (e.g. for metadata).
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
int
lod_plan_init_from_dims(struct lod_plan* p,
                        const struct dimension* dims,
                        uint8_t rank,
                        int max_levels);

void
lod_plan_free(struct lod_plan* p);
