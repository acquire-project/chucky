// morton.util.c — shared CPU reference for LOD via compacted Morton codes.

#include "morton.util.h"

#include "lod/reduce_csr.h"
#include "util/index.ops.h"
#include "util/prelude.h"

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
  for (int k = p->fixed_dims_ndim - 1; k >= 0; --k) {
    idx += full_coords[p->fixed_dim_to_dim[k]] * stride;
    stride *= p->fixed_dims_shape[k];
  }
  return idx;
}

static void
plan_extract_lod(const struct lod_plan* p,
                 const uint64_t* full_coords,
                 uint64_t* lod_coords)
{
  for (int k = 0; k < p->lod_ndim; ++k)
    lod_coords[k] = full_coords[p->lod_to_dim[k]];
}

void
lod_scatter_cpu(const struct lod_plan* p, const float* src, float* dst)
{
  uint64_t full_shape[MAX_NDIM];
  level_dims_get_shape(&p->levels.level[0], p->ndim, full_shape);
  uint64_t n = lod_span_len(lod_spans_at(&p->level_spans, 0));

  uint64_t lod_shape0[MAX_NDIM];
  lod_plan_fill_lod_shapes(p, 0, lod_shape0);

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
    uint64_t pos = morton_rank(p->lod_ndim, lod_shape0, lod_coords, 0);
    dst[b * p->levels.level[0].lod_nelem + pos] = src[i];
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
  for (int l = 0; l < p->levels.nlod - 1; ++l) {
    const struct level_dims* src_ld = &p->levels.level[l];
    const struct level_dims* dst_ld = &p->levels.level[l + 1];
    uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
    uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;

    struct reduce_csr csr = { 0 };
    if (reduce_csr_alloc(&csr, src_total, dst_total))
      continue;
    if (reduce_csr_build(&csr, p, l)) {
      reduce_csr_free(&csr);
      continue;
    }

    struct lod_span src_level = lod_spans_at(&p->level_spans, l);
    struct lod_span dst_level = lod_spans_at(&p->level_spans, l + 1);

    for (uint64_t b = 0; b < csr.batch_count; ++b) {
      float* src = values + src_level.beg + b * csr.src_lod_count;
      uint64_t dst_base = dst_level.beg + b * csr.dst_segment_size;

      for (uint64_t i = 0; i < csr.dst_segment_size; ++i) {
        // Gather CSR window into a contiguous buffer, then reduce.
        uint64_t start = csr.starts[i];
        uint64_t end = csr.starts[i + 1];
        if (start >= end) {
          values[dst_base + i] = 0;
          continue;
        }
        uint64_t len = end - start;
        float buf[16];
        for (uint64_t j = 0; j < len && j < 16; ++j)
          buf[j] = src[csr.indices[start + j]];
        values[dst_base + i] =
          reduce_window_f32(buf, 0, len < 16 ? len : 16, method);
      }
    }

    reduce_csr_free(&csr);
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

  uint64_t total_vals = p->level_spans.ends[p->levels.nlod - 1];
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
  uint64_t full_shape[MAX_NDIM];
  level_dims_get_shape(&p->levels.level[0], p->ndim, full_shape);
  uint64_t n = lod_span_len(lod_spans_at(&p->level_spans, 0));

  uint64_t lod_shape0[MAX_NDIM];
  lod_plan_fill_lod_shapes(p, 0, lod_shape0);

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
    uint64_t pos = morton_rank(p->lod_ndim, lod_shape0, lod_coords, 0);
    dst[b * p->levels.level[0].lod_nelem + pos] = src[i];
  }
}

uint16_t
reduce_window_u16(const uint16_t* src,
                  const uint64_t* indices,
                  uint64_t start,
                  uint64_t end,
                  enum lod_reduce_method method)
{
  uint64_t len = end - start;
  switch (method) {
    case lod_reduce_mean: {
      uint32_t sum = 0;
      for (uint64_t j = start; j < end; ++j)
        sum += src[indices[j]];
      return (uint16_t)(sum / (uint32_t)len);
    }
    case lod_reduce_min: {
      uint16_t best = src[indices[start]];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[indices[j]] < best)
          best = src[indices[j]];
      return best;
    }
    case lod_reduce_max: {
      uint16_t best = src[indices[start]];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[indices[j]] > best)
          best = src[indices[j]];
      return best;
    }
    case lod_reduce_median: {
      uint16_t buf[16];
      uint64_t n = (len <= 16) ? len : 16;
      for (uint64_t j = 0; j < n; ++j)
        buf[j] = src[indices[start + j]];
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
      uint16_t top1 = src[indices[start]], top2 = src[indices[start]];
      if (len > 1) {
        uint16_t v = src[indices[start + 1]];
        if (v >= top1) {
          top2 = top1;
          top1 = v;
        } else {
          top2 = v;
        }
        for (uint64_t j = start + 2; j < end; ++j) {
          v = src[indices[j]];
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
      uint16_t bot1 = src[indices[start]], bot2 = src[indices[start]];
      if (len > 1) {
        uint16_t v = src[indices[start + 1]];
        if (v <= bot1) {
          bot2 = bot1;
          bot1 = v;
        } else {
          bot2 = v;
        }
        for (uint64_t j = start + 2; j < end; ++j) {
          v = src[indices[j]];
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
  for (int l = 0; l < p->levels.nlod - 1; ++l) {
    const struct level_dims* src_ld = &p->levels.level[l];
    const struct level_dims* dst_ld = &p->levels.level[l + 1];
    uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
    uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;

    struct reduce_csr csr = { 0 };
    if (reduce_csr_alloc(&csr, src_total, dst_total))
      continue;
    if (reduce_csr_build(&csr, p, l)) {
      reduce_csr_free(&csr);
      continue;
    }

    struct lod_span src_level = lod_spans_at(&p->level_spans, l);
    struct lod_span dst_level = lod_spans_at(&p->level_spans, l + 1);

    for (uint64_t b = 0; b < csr.batch_count; ++b) {
      uint16_t* src = values + src_level.beg + b * csr.src_lod_count;
      uint64_t dst_base = dst_level.beg + b * csr.dst_segment_size;

      for (uint64_t i = 0; i < csr.dst_segment_size; ++i) {
        uint64_t start = csr.starts[i];
        uint64_t end = csr.starts[i + 1];
        if (start >= end) {
          values[dst_base + i] = 0;
          continue;
        }
        values[dst_base + i] =
          reduce_window_u16(src, csr.indices, start, end, method);
      }
    }

    reduce_csr_free(&csr);
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

  uint64_t total_vals = p->level_spans.ends[p->levels.nlod - 1];
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

// --- Brute-force LOD reduce (no CSR) ---

// For each level transition, iterate all source elements, compute destination
// by halving coordinates, gather into temp buffers, reduce.
void
lod_reduce_bruteforce(const struct lod_plan* p,
                      float* values,
                      enum lod_reduce_method method)
{
  for (int l = 0; l < p->levels.nlod - 1; ++l) {
    const struct level_dims* src_ld = &p->levels.level[l];
    const struct level_dims* dst_ld = &p->levels.level[l + 1];
    struct lod_span src_span = lod_spans_at(&p->level_spans, l);
    struct lod_span dst_span = lod_spans_at(&p->level_spans, l + 1);
    uint32_t dropped_mask = src_ld->lod_mask & ~dst_ld->lod_mask;

    uint64_t dst_count = dst_ld->fixed_dims_count * dst_ld->lod_nelem;

    uint64_t src_lod_shape[MAX_NDIM];
    for (int k = 0; k < src_ld->lod_ndim; ++k)
      src_lod_shape[k] = src_ld->dim[src_ld->lod_to_dim[k]].size;

    uint64_t dst_lod_shape[MAX_NDIM];
    for (int k = 0; k < dst_ld->lod_ndim; ++k)
      dst_lod_shape[k] = dst_ld->dim[dst_ld->lod_to_dim[k]].size;

    // Gather: for each dst element, collect contributing src values.
    uint64_t* counts = (uint64_t*)calloc(dst_count, sizeof(uint64_t));
    float** bufs = (float**)calloc(dst_count, sizeof(float*));
    if (!counts || !bufs)
      goto CleanupLevel;

    // Pass 1: count contributions per destination.
    for (uint64_t src_batch = 0; src_batch < src_ld->fixed_dims_count;
         ++src_batch) {
      uint64_t fixed_coords[MAX_NDIM];
      memset(fixed_coords, 0, sizeof(fixed_coords));
      {
        uint64_t rem = src_batch;
        for (int k = src_ld->fixed_dims_ndim - 1; k >= 0; --k) {
          fixed_coords[src_ld->fixed_dim_to_dim[k]] =
            rem % src_ld->fixed_dims_shape[k];
          rem /= src_ld->fixed_dims_shape[k];
        }
      }

      for (uint64_t se = 0; se < src_ld->lod_nelem; ++se) {
        uint64_t sc[MAX_NDIM];
        unravel(src_ld->lod_ndim, src_lod_shape, se, sc);

        uint64_t dst_fc[MAX_NDIM];
        memcpy(dst_fc, fixed_coords, sizeof(dst_fc));
        uint64_t dlc[MAX_NDIM];
        int si = 0;
        for (int k = 0; k < src_ld->lod_ndim; ++k) {
          int d = src_ld->lod_to_dim[k];
          if (dropped_mask & (1u << d))
            dst_fc[d] = sc[k] / 2;
          else
            dlc[si++] = sc[k] / 2;
        }
        uint64_t dm = (dst_ld->lod_ndim > 0)
                        ? morton_rank(dst_ld->lod_ndim, dst_lod_shape, dlc, 0)
                        : 0;
        uint64_t dst_bi = 0;
        for (int k = 0; k < dst_ld->fixed_dims_ndim; ++k)
          dst_bi = dst_bi * dst_ld->fixed_dims_shape[k] +
                   dst_fc[dst_ld->fixed_dim_to_dim[k]];
        counts[dst_bi * dst_ld->lod_nelem + dm]++;
      }
    }

    // Allocate per-dst gather buffers.
    for (uint64_t i = 0; i < dst_count; ++i) {
      if (counts[i] > 0) {
        bufs[i] = (float*)malloc(counts[i] * sizeof(float));
        if (!bufs[i])
          goto CleanupLevel;
      }
    }
    memset(counts, 0, dst_count * sizeof(uint64_t));

    // Pass 2: gather values.
    for (uint64_t src_batch = 0; src_batch < src_ld->fixed_dims_count;
         ++src_batch) {
      uint64_t fixed_coords[MAX_NDIM];
      memset(fixed_coords, 0, sizeof(fixed_coords));
      {
        uint64_t rem = src_batch;
        for (int k = src_ld->fixed_dims_ndim - 1; k >= 0; --k) {
          fixed_coords[src_ld->fixed_dim_to_dim[k]] =
            rem % src_ld->fixed_dims_shape[k];
          rem /= src_ld->fixed_dims_shape[k];
        }
      }

      for (uint64_t se = 0; se < src_ld->lod_nelem; ++se) {
        uint64_t sc[MAX_NDIM];
        unravel(src_ld->lod_ndim, src_lod_shape, se, sc);

        uint64_t dst_fc[MAX_NDIM];
        memcpy(dst_fc, fixed_coords, sizeof(dst_fc));
        uint64_t dlc[MAX_NDIM];
        int si = 0;
        for (int k = 0; k < src_ld->lod_ndim; ++k) {
          int d = src_ld->lod_to_dim[k];
          if (dropped_mask & (1u << d))
            dst_fc[d] = sc[k] / 2;
          else
            dlc[si++] = sc[k] / 2;
        }
        uint64_t dm = (dst_ld->lod_ndim > 0)
                        ? morton_rank(dst_ld->lod_ndim, dst_lod_shape, dlc, 0)
                        : 0;
        uint64_t dst_bi = 0;
        for (int k = 0; k < dst_ld->fixed_dims_ndim; ++k)
          dst_bi = dst_bi * dst_ld->fixed_dims_shape[k] +
                   dst_fc[dst_ld->fixed_dim_to_dim[k]];
        uint64_t di = dst_bi * dst_ld->lod_nelem + dm;

        uint64_t src_morton =
          morton_rank(src_ld->lod_ndim, src_lod_shape, sc, 0);
        uint64_t src_pos =
          src_span.beg + src_batch * src_ld->lod_nelem + src_morton;
        bufs[di][counts[di]++] = values[src_pos];
      }
    }

    // Pass 3: reduce and write to destination.
    for (uint64_t i = 0; i < dst_count; ++i) {
      if (counts[i] > 0)
        values[dst_span.beg + i] =
          reduce_window_f32(bufs[i], 0, counts[i], method);
      else
        values[dst_span.beg + i] = 0;
    }

  CleanupLevel:
    for (uint64_t i = 0; i < dst_count; ++i)
      free(bufs[i]);
    free(bufs);
    free(counts);
  }
}

int
lod_build_host_csrs(const struct lod_plan* p, struct reduce_csr* csrs)
{
  int ncsr = p->levels.nlod - 1;
  for (int l = 0; l < ncsr; ++l) {
    const struct level_dims* src_ld = &p->levels.level[l];
    const struct level_dims* dst_ld = &p->levels.level[l + 1];
    uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
    uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;
    if (reduce_csr_alloc(&csrs[l], src_total, dst_total))
      goto Fail;
    if (reduce_csr_build(&csrs[l], p, l))
      goto Fail;
  }
  return 0;
Fail:
  lod_free_host_csrs(p, csrs);
  return 1;
}

void
lod_free_host_csrs(const struct lod_plan* p, struct reduce_csr* csrs)
{
  int ncsr = p->levels.nlod - 1;
  for (int l = 0; l < ncsr; ++l)
    reduce_csr_free(&csrs[l]);
}
