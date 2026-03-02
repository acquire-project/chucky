// morton.util.c — shared CPU reference for LOD via compacted Morton codes.
// Intended to be #include'd (no main).

#include "index.ops.h"
#include "lod_plan.h"
#include "prelude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM LOD_MAX_NDIM
#define MAX_LOD LOD_MAX_LEVELS

// --- CPU reference functions using lod_plan ---

// Compute batch index matching the GPU convention (reverse dim iteration).
static uint64_t
plan_batch_index(const struct lod_plan* p, const uint64_t* full_coords)
{
  uint64_t idx = 0, stride = 1;
  for (int k = p->batch_ndim - 1; k >= 0; --k) {
    idx += full_coords[p->batch_map[k]] * stride;
    stride *= p->batch_shape[k];
  }
  return idx;
}

static void
plan_extract_lod(const struct lod_plan* p,
                 const uint64_t* full_coords,
                 uint64_t* lod_coords)
{
  for (int k = 0; k < p->lod_ndim; ++k)
    lod_coords[k] = full_coords[p->lod_map[k]];
}

static void
lod_scatter_cpu(const struct lod_plan* p, const float* src, float* dst)
{
  const uint64_t* full_shape = p->shapes[0];
  uint64_t n = lod_span_len(lod_spans_at(&p->levels, 0));

  uint64_t full_coords[MAX_NDIM];
  uint64_t lod_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    // Decompose in C-order (dim ndim-1 fastest) to match GPU and data layout.
    {
      uint64_t rest = i;
      for (int d = p->ndim - 1; d >= 0; --d) {
        full_coords[d] = rest % full_shape[d];
        rest /= full_shape[d];
      }
    }
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_lod(p, full_coords, lod_coords);
    uint64_t pos = morton_rank(p->lod_ndim, p->lod_shapes[0], lod_coords, 0);
    dst[b * p->lod_counts[0] + pos] = src[i];
  }
}

static void
lod_reduce_cpu(const struct lod_plan* p, float* values)
{
  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t src_ds = p->lod_counts[l];
    uint64_t dst_ds = p->lod_counts[l + 1];
    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    for (uint64_t b = 0; b < p->batch_count; ++b) {
      uint64_t src_base = src_level.beg + b * src_ds;
      uint64_t dst_base = dst_level.beg + b * dst_ds;

      for (uint64_t i = 0; i < dst_ds; ++i) {
        uint64_t start = (i > 0) ? p->ends[seg.beg + i - 1] : 0;
        uint64_t end = p->ends[seg.beg + i];
        uint64_t len = end - start;
        float sum = 0;
        for (uint64_t j = start; j < end; ++j)
          sum += values[src_base + j];
        values[dst_base + i] = sum / (float)len;
      }
    }
  }
}

static int
lod_compute(const struct lod_plan* p, const float* src, float** out_values)
{
  int ok = 0;
  *out_values = NULL;

  uint64_t total_vals = p->levels.ends[p->nlod - 1];
  float* values = (float*)malloc(total_vals * sizeof(float));
  CHECK(Error, values);
  *out_values = values;

  lod_scatter_cpu(p, src, values);
  lod_reduce_cpu(p, values);

  ok = 1;
Error:
  if (!ok) {
    free(*out_values);
    *out_values = NULL;
  }
  return ok;
}
