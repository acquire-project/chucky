// morton.util.c — shared CPU reference for LOD via compacted Morton codes.

#include "morton.util.h"

#include "index.ops.h"
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

void
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

static float
reduce_window_f32(const float* src,
                  uint64_t start,
                  uint64_t end,
                  enum lod_reduce_method method)
{
  uint64_t len = end - start;
  switch (method) {
    case lod_reduce_mean: {
      float sum = 0;
      for (uint64_t j = start; j < end; ++j)
        sum += src[j];
      return sum / (float)len;
    }
    case lod_reduce_min: {
      float best = src[start];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[j] < best)
          best = src[j];
      return best;
    }
    case lod_reduce_max: {
      float best = src[start];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[j] > best)
          best = src[j];
      return best;
    }
    case lod_reduce_median: {
      float buf[16];
      uint64_t n = (len <= 16) ? len : 16;
      for (uint64_t j = 0; j < n; ++j)
        buf[j] = src[start + j];
      for (uint64_t i = 1; i < n; ++i) {
        float key = buf[i];
        uint64_t k = i;
        while (k > 0 && buf[k - 1] > key) {
          buf[k] = buf[k - 1];
          --k;
        }
        buf[k] = key;
      }
      return buf[n / 2];
    }
    case lod_reduce_max_suppressed: {
      float top1 = src[start], top2 = src[start];
      if (len > 1) {
        float v = src[start + 1];
        if (v >= top1) {
          top2 = top1;
          top1 = v;
        } else {
          top2 = v;
        }
        for (uint64_t j = start + 2; j < end; ++j) {
          v = src[j];
          if (v >= top1) {
            top2 = top1;
            top1 = v;
          } else if (v > top2) {
            top2 = v;
          }
        }
      }
      return top2;
    }
    case lod_reduce_min_suppressed: {
      float bot1 = src[start], bot2 = src[start];
      if (len > 1) {
        float v = src[start + 1];
        if (v <= bot1) {
          bot2 = bot1;
          bot1 = v;
        } else {
          bot2 = v;
        }
        for (uint64_t j = start + 2; j < end; ++j) {
          v = src[j];
          if (v <= bot1) {
            bot2 = bot1;
            bot1 = v;
          } else if (v < bot2) {
            bot2 = v;
          }
        }
      }
      return bot2;
    }
  }
  return 0;
}

void
lod_reduce_cpu(const struct lod_plan* p,
               float* values,
               enum lod_reduce_method method)
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
        values[dst_base + i] =
          reduce_window_f32(values + src_base, start, end, method);
      }
    }
  }
}

int
lod_compute(const struct lod_plan* p,
            const float* src,
            float** out_values,
            enum lod_reduce_method method)
{
  int ok = 0;
  *out_values = NULL;

  uint64_t total_vals = p->levels.ends[p->nlod - 1];
  float* values = (float*)malloc(total_vals * sizeof(float));
  CHECK(Error, values);
  *out_values = values;

  lod_scatter_cpu(p, src, values);
  lod_reduce_cpu(p, values, method);

  ok = 1;
Error:
  if (!ok) {
    free(*out_values);
    *out_values = NULL;
  }
  return ok;
}

// --- u16 CPU reference ---

void
lod_scatter_cpu_u16(const struct lod_plan* p,
                    const uint16_t* src,
                    uint16_t* dst)
{
  const uint64_t* full_shape = p->shapes[0];
  uint64_t n = lod_span_len(lod_spans_at(&p->levels, 0));

  uint64_t full_coords[MAX_NDIM];
  uint64_t lod_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
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

uint16_t
reduce_window_u16(const uint16_t* src,
                  uint64_t start,
                  uint64_t end,
                  enum lod_reduce_method method)
{
  uint64_t len = end - start;
  switch (method) {
    case lod_reduce_mean: {
      uint32_t sum = 0;
      for (uint64_t j = start; j < end; ++j)
        sum += src[j];
      return (uint16_t)(sum / (uint32_t)len);
    }
    case lod_reduce_min: {
      uint16_t best = src[start];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[j] < best)
          best = src[j];
      return best;
    }
    case lod_reduce_max: {
      uint16_t best = src[start];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[j] > best)
          best = src[j];
      return best;
    }
    case lod_reduce_median: {
      uint16_t buf[16];
      uint64_t n = (len <= 16) ? len : 16;
      for (uint64_t j = 0; j < n; ++j)
        buf[j] = src[start + j];
      for (uint64_t i = 1; i < n; ++i) {
        uint16_t key = buf[i];
        uint64_t k = i;
        while (k > 0 && buf[k - 1] > key) {
          buf[k] = buf[k - 1];
          --k;
        }
        buf[k] = key;
      }
      return buf[n / 2];
    }
    case lod_reduce_max_suppressed: {
      uint16_t top1 = src[start], top2 = src[start];
      if (len > 1) {
        uint16_t v = src[start + 1];
        if (v >= top1) {
          top2 = top1;
          top1 = v;
        } else {
          top2 = v;
        }
        for (uint64_t j = start + 2; j < end; ++j) {
          v = src[j];
          if (v >= top1) {
            top2 = top1;
            top1 = v;
          } else if (v > top2) {
            top2 = v;
          }
        }
      }
      return top2;
    }
    case lod_reduce_min_suppressed: {
      uint16_t bot1 = src[start], bot2 = src[start];
      if (len > 1) {
        uint16_t v = src[start + 1];
        if (v <= bot1) {
          bot2 = bot1;
          bot1 = v;
        } else {
          bot2 = v;
        }
        for (uint64_t j = start + 2; j < end; ++j) {
          v = src[j];
          if (v <= bot1) {
            bot2 = bot1;
            bot1 = v;
          } else if (v < bot2) {
            bot2 = v;
          }
        }
      }
      return bot2;
    }
  }
  return 0;
}

void
lod_reduce_cpu_u16(const struct lod_plan* p,
                   uint16_t* values,
                   enum lod_reduce_method method)
{
  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t src_lod = p->lod_counts[l];
    uint64_t dst_lod = p->lod_counts[l + 1];
    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    for (uint64_t b = 0; b < p->batch_count; ++b) {
      uint64_t src_base = src_level.beg + b * src_lod;
      uint64_t dst_base = dst_level.beg + b * dst_lod;

      for (uint64_t i = 0; i < dst_lod; ++i) {
        uint64_t start = (i > 0) ? p->ends[seg.beg + i - 1] : 0;
        uint64_t end = p->ends[seg.beg + i];
        values[dst_base + i] =
          reduce_window_u16(values + src_base, start, end, method);
      }
    }
  }
}

int
lod_compute_u16(const struct lod_plan* p,
                const uint16_t* src,
                uint16_t** out_values,
                enum lod_reduce_method method)
{
  int ok = 0;
  *out_values = NULL;

  uint64_t total_vals = p->levels.ends[p->nlod - 1];
  uint16_t* values = (uint16_t*)malloc(total_vals * sizeof(uint16_t));
  CHECK(Error, values);
  *out_values = values;

  lod_scatter_cpu_u16(p, src, values);
  lod_reduce_cpu_u16(p, values, method);

  ok = 1;
Error:
  if (!ok) {
    free(*out_values);
    *out_values = NULL;
  }
  return ok;
}
