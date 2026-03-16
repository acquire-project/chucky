// GPU LOD test: compare GPU kernels against CPU reference from morton.util.c

#include "morton.util.h"

#include "lod.h"
#include "metric.cuda.h"
#include "prelude.cuda.h"
#include "prelude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM LOD_MAX_NDIM
#define MAX_LOD LOD_MAX_LEVELS

static int
upload(CUdeviceptr* d_ptr, const void* h_ptr, size_t bytes)
{
  CUresult r = cuMemAlloc(d_ptr, bytes);
  if (r != CUDA_SUCCESS)
    return 0;
  r = cuMemcpyHtoD(*d_ptr, h_ptr, bytes);
  if (r != CUDA_SUCCESS) {
    cuMemFree(*d_ptr);
    *d_ptr = 0;
    return 0;
  }
  return 1;
}

// Build gather LUT and batch offsets for lod_gather_lut, mirroring stream.c.
// Returns 1 on success, 0 on failure.
static int
setup_gather(const struct lod_plan* p,
             CUdeviceptr* d_lod_shape,
             CUdeviceptr* d_lod_strides,
             CUdeviceptr* d_gather_lut,
             CUdeviceptr* d_batch_offsets,
             CUstream stream)
{
  *d_lod_shape = 0;
  *d_lod_strides = 0;
  *d_gather_lut = 0;
  *d_batch_offsets = 0;

  uint64_t full_strides[MAX_NDIM];
  full_strides[p->ndim - 1] = 1;
  for (int d = p->ndim - 2; d >= 0; --d)
    full_strides[d] = full_strides[d + 1] * p->shapes[0][d + 1];

  if (p->lod_ndim > 0) {
    CHECK(
      Fail,
      upload(d_lod_shape, p->lod_shapes[0], p->lod_ndim * sizeof(uint64_t)));

    uint64_t lod_strides[MAX_NDIM];
    int li = p->lod_ndim - 1;
    for (int d = p->ndim - 1; d >= 0; --d) {
      if ((p->lod_mask >> d) & 1) {
        lod_strides[li] = full_strides[d];
        li--;
      }
    }
    CHECK(Fail,
          upload(d_lod_strides, lod_strides, p->lod_ndim * sizeof(uint64_t)));

    CU(Fail, cuMemAlloc(d_gather_lut, p->lod_counts[0] * sizeof(uint32_t)));
    CHECK(Fail,
          lod_build_gather_lut(*d_gather_lut,
                               *d_lod_shape,
                               *d_lod_strides,
                               p->lod_ndim,
                               p->lod_shapes[0],
                               p->lod_counts[0],
                               stream) == 0);
  } else {
    uint32_t zero = 0;
    CHECK(Fail, upload(d_gather_lut, &zero, sizeof(uint32_t)));
  }

  {
    uint32_t* batch_off = (uint32_t*)calloc(p->batch_count, sizeof(uint32_t));
    CHECK(Fail, batch_off);
    for (uint64_t bi = 0; bi < p->batch_count; ++bi) {
      uint64_t remainder = bi;
      uint64_t offset = 0;
      for (int k = p->batch_ndim - 1; k >= 0; --k) {
        uint64_t coord = remainder % p->batch_shape[k];
        remainder /= p->batch_shape[k];
        offset += coord * full_strides[p->batch_map[k]];
      }
      batch_off[bi] = (uint32_t)offset;
    }
    CHECK(
      Fail,
      upload(d_batch_offsets, batch_off, p->batch_count * sizeof(uint32_t)));
    free(batch_off);
  }

  return 1;
Fail:
  cuMemFree(*d_lod_shape);
  cuMemFree(*d_lod_strides);
  cuMemFree(*d_gather_lut);
  cuMemFree(*d_batch_offsets);
  *d_lod_shape = *d_lod_strides = *d_gather_lut = *d_batch_offsets = 0;
  return 0;
}

static void
report_metric(const struct stream_metric* m)
{
  if (m->count == 0)
    return;
  float avg_ms = m->ms / (float)m->count;
  double avg_bytes = m->total_bytes / (double)m->count;
  log_info("  %-8s %7.3f ms  %6.2f GB/s  (%d iters)",
           m->name,
           avg_ms,
           avg_bytes / ((double)avg_ms * 1e6),
           m->count);
}

struct test_lod_metrics
{
  struct stream_metric scatter;
  struct stream_metric pyramid;
  struct stream_metric total;
};

static void
test_lod_metrics_init(struct test_lod_metrics* m)
{
  *m = (struct test_lod_metrics){
    .scatter = { .name = "scatter", .best_ms = 1e30f },
    .pyramid = { .name = "pyramid", .best_ms = 1e30f },
    .total = { .name = "total", .best_ms = 1e30f },
  };
}

static void
test_lod_metrics_report(const struct test_lod_metrics* m)
{
  report_metric(&m->scatter);
  report_metric(&m->pyramid);
  report_metric(&m->total);
}

// Run LOD computation on GPU: scatter + fill_ends + reduce.
// Returns allocated host buffer with results, or NULL on failure.
static float*
lod_compute_gpu(const struct lod_plan* p,
                const float* src,
                struct test_lod_metrics* metrics,
                enum lod_reduce_method method)
{
  float* result = NULL;
  CUstream stream = NULL;
  CUevent ev_start = NULL, ev_scatter = NULL, ev_done = NULL;

  CUdeviceptr d_src = 0, d_values = 0;
  CUdeviceptr d_lod_shape = 0, d_lod_strides = 0;
  CUdeviceptr d_gather_lut = 0, d_batch_offsets = 0;
  CUdeviceptr d_ends = 0;
  CUdeviceptr d_child_shape = 0, d_parent_shape = 0;

  uint64_t n_elements = lod_span_len(lod_spans_at(&p->levels, 0));
  uint64_t total_vals = p->levels.ends[p->nlod - 1];

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev_scatter, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev_done, CU_EVENT_DEFAULT));

  CHECK(Fail, upload(&d_src, src, n_elements * sizeof(float)));
  CU(Fail, cuMemAlloc(&d_values, total_vals * sizeof(float)));

  CHECK(Fail,
        setup_gather(p,
                     &d_lod_shape,
                     &d_lod_strides,
                     &d_gather_lut,
                     &d_batch_offsets,
                     stream));

  CU(Fail, cuEventRecord(ev_start, stream));

  lod_gather_lut(d_values,
                 d_src,
                 d_gather_lut,
                 d_batch_offsets,
                 lod_dtype_f32,
                 p->lod_counts[0],
                 p->batch_count,
                 stream);

  CU(Fail, cuEventRecord(ev_scatter, stream));

  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    cuMemFree(d_child_shape);
    cuMemFree(d_parent_shape);
    d_child_shape = 0;
    d_parent_shape = 0;

    CHECK(
      Fail,
      upload(&d_child_shape, p->lod_shapes[l], p->lod_ndim * sizeof(uint64_t)));
    CHECK(Fail,
          upload(&d_parent_shape,
                 p->lod_shapes[l + 1],
                 p->lod_ndim * sizeof(uint64_t)));

    cuMemFree(d_ends);
    d_ends = 0;
    CU(Fail, cuMemAlloc(&d_ends, n_parents * sizeof(uint64_t)));

    CHECK(Fail,
          lod_fill_ends_gpu(d_ends,
                            p->lod_ndim,
                            d_child_shape,
                            d_parent_shape,
                            p->lod_shapes[l],
                            p->lod_shapes[l + 1],
                            n_parents,
                            stream) == 0);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    lod_reduce(d_values,
               d_ends,
               lod_dtype_f32,
               method,
               src_level.beg,
               dst_level.beg,
               p->lod_counts[l],
               p->lod_counts[l + 1],
               p->batch_count,
               stream);
  }

  CU(Fail, cuEventRecord(ev_done, stream));
  CU(Fail, cuStreamSynchronize(stream));

  if (metrics) {
    size_t nbytes = total_vals * sizeof(float);
    accumulate_metric_cu(&metrics->scatter, ev_start, ev_scatter, nbytes);
    accumulate_metric_cu(&metrics->pyramid, ev_scatter, ev_done, nbytes);
    accumulate_metric_cu(&metrics->total, ev_start, ev_done, nbytes);
  }

  result = (float*)malloc(total_vals * sizeof(float));
  CHECK(Fail, result);
  CU(Fail, cuMemcpyDtoH(result, d_values, total_vals * sizeof(float)));

Fail:
  cuMemFree(d_src);
  cuMemFree(d_values);
  cuMemFree(d_lod_shape);
  cuMemFree(d_lod_strides);
  cuMemFree(d_gather_lut);
  cuMemFree(d_batch_offsets);
  cuMemFree(d_ends);
  cuMemFree(d_child_shape);
  cuMemFree(d_parent_shape);
  cuEventDestroy(ev_start);
  cuEventDestroy(ev_scatter);
  cuEventDestroy(ev_done);
  cuStreamDestroy(stream);
  return result;
}

static int
test_lod_gpu_method(const char* label,
                    int ndim,
                    const uint64_t* shape,
                    uint32_t lod_mask,
                    int niter,
                    enum lod_reduce_method method)
{
  log_info("=== %s ===", label);
  int ok = 0;
  float* src = NULL;
  float* cpu_values = NULL;
  float* gpu_values = NULL;
  struct lod_plan plan = { 0 };

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);

  CHECK(Fail,
        lod_plan_init(&plan, ndim, shape, NULL, lod_mask, MAX_LOD, 0) == 0);
  log_info(
    "  lod_mask=0x%x  lod_ndim=%d  batch_ndim=%d  batch_count=%llu  nlod=%d",
    lod_mask,
    plan.lod_ndim,
    plan.batch_ndim,
    (unsigned long long)plan.batch_count,
    plan.nlod);

  CHECK(Fail, lod_compute(&plan, src, &cpu_values, method));

  struct test_lod_metrics metrics;
  test_lod_metrics_init(&metrics);
  for (int iter = 0; iter < niter; ++iter) {
    free(gpu_values);
    gpu_values = lod_compute_gpu(&plan, src, &metrics, method);
    CHECK(Fail, gpu_values);
  }
  test_lod_metrics_report(&metrics);

  {
    uint64_t total = plan.levels.ends[plan.nlod - 1];
    for (uint64_t i = 0; i < total; ++i) {
      if (fabsf(gpu_values[i] - cpu_values[i]) > 1e-5f) {
        log_error("  FAIL at i=%llu: gpu=%f cpu=%f",
               (unsigned long long)i,
               gpu_values[i],
               cpu_values[i]);
        goto Fail;
      }
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(src);
  free(cpu_values);
  free(gpu_values);
  lod_plan_free(&plan);
  return ok ? 0 : 1;
}

static int
test_lod_gpu(const char* label,
             int ndim,
             const uint64_t* shape,
             uint32_t lod_mask,
             int niter)
{
  return test_lod_gpu_method(
    label, ndim, shape, lod_mask, niter, lod_reduce_mean);
}

// Run u16 LOD computation on GPU.
static uint16_t*
lod_compute_gpu_u16(const struct lod_plan* p,
                    const uint16_t* src,
                    struct test_lod_metrics* metrics,
                    enum lod_reduce_method method)
{
  uint16_t* result = NULL;
  CUstream stream = NULL;
  CUevent ev_start = NULL, ev_scatter = NULL, ev_done = NULL;

  CUdeviceptr d_src = 0, d_values = 0;
  CUdeviceptr d_lod_shape = 0, d_lod_strides = 0;
  CUdeviceptr d_gather_lut = 0, d_batch_offsets = 0;
  CUdeviceptr d_ends = 0;
  CUdeviceptr d_child_shape = 0, d_parent_shape = 0;

  uint64_t n_elements = lod_span_len(lod_spans_at(&p->levels, 0));
  uint64_t total_vals = p->levels.ends[p->nlod - 1];

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev_scatter, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev_done, CU_EVENT_DEFAULT));

  CHECK(Fail, upload(&d_src, src, n_elements * sizeof(uint16_t)));
  CU(Fail, cuMemAlloc(&d_values, total_vals * sizeof(uint16_t)));

  CHECK(Fail,
        setup_gather(p,
                     &d_lod_shape,
                     &d_lod_strides,
                     &d_gather_lut,
                     &d_batch_offsets,
                     stream));

  CU(Fail, cuEventRecord(ev_start, stream));

  lod_gather_lut(d_values,
                 d_src,
                 d_gather_lut,
                 d_batch_offsets,
                 lod_dtype_u16,
                 p->lod_counts[0],
                 p->batch_count,
                 stream);

  CU(Fail, cuEventRecord(ev_scatter, stream));

  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    cuMemFree(d_child_shape);
    cuMemFree(d_parent_shape);
    d_child_shape = 0;
    d_parent_shape = 0;

    CHECK(
      Fail,
      upload(&d_child_shape, p->lod_shapes[l], p->lod_ndim * sizeof(uint64_t)));
    CHECK(Fail,
          upload(&d_parent_shape,
                 p->lod_shapes[l + 1],
                 p->lod_ndim * sizeof(uint64_t)));

    cuMemFree(d_ends);
    d_ends = 0;
    CU(Fail, cuMemAlloc(&d_ends, n_parents * sizeof(uint64_t)));

    CHECK(Fail,
          lod_fill_ends_gpu(d_ends,
                            p->lod_ndim,
                            d_child_shape,
                            d_parent_shape,
                            p->lod_shapes[l],
                            p->lod_shapes[l + 1],
                            n_parents,
                            stream) == 0);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    lod_reduce(d_values,
               d_ends,
               lod_dtype_u16,
               method,
               src_level.beg,
               dst_level.beg,
               p->lod_counts[l],
               p->lod_counts[l + 1],
               p->batch_count,
               stream);
  }

  CU(Fail, cuEventRecord(ev_done, stream));
  CU(Fail, cuStreamSynchronize(stream));

  if (metrics) {
    size_t nbytes = total_vals * sizeof(uint16_t);
    accumulate_metric_cu(&metrics->scatter, ev_start, ev_scatter, nbytes);
    accumulate_metric_cu(&metrics->pyramid, ev_scatter, ev_done, nbytes);
    accumulate_metric_cu(&metrics->total, ev_start, ev_done, nbytes);
  }

  result = (uint16_t*)malloc(total_vals * sizeof(uint16_t));
  CHECK(Fail, result);
  CU(Fail, cuMemcpyDtoH(result, d_values, total_vals * sizeof(uint16_t)));

Fail:
  cuMemFree(d_src);
  cuMemFree(d_values);
  cuMemFree(d_lod_shape);
  cuMemFree(d_lod_strides);
  cuMemFree(d_gather_lut);
  cuMemFree(d_batch_offsets);
  cuMemFree(d_ends);
  cuMemFree(d_child_shape);
  cuMemFree(d_parent_shape);
  cuEventDestroy(ev_start);
  cuEventDestroy(ev_scatter);
  cuEventDestroy(ev_done);
  cuStreamDestroy(stream);
  return result;
}

static int
test_lod_gpu_u16_method(const char* label,
                        int ndim,
                        const uint64_t* shape,
                        uint32_t lod_mask,
                        int niter,
                        enum lod_reduce_method method)
{
  log_info("=== %s ===", label);
  int ok = 0;
  uint16_t* src = NULL;
  uint16_t* cpu_values = NULL;
  uint16_t* gpu_values = NULL;
  struct lod_plan plan = { 0 };

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  src = (uint16_t*)malloc(n * sizeof(uint16_t));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (uint16_t)((i + 1) & 0xFFFF);

  CHECK(Fail,
        lod_plan_init(&plan, ndim, shape, NULL, lod_mask, MAX_LOD, 0) == 0);
  log_info(
    "  lod_mask=0x%x  lod_ndim=%d  batch_ndim=%d  batch_count=%llu  nlod=%d",
    lod_mask,
    plan.lod_ndim,
    plan.batch_ndim,
    (unsigned long long)plan.batch_count,
    plan.nlod);

  CHECK(Fail, lod_compute_u16(&plan, src, &cpu_values, method));

  struct test_lod_metrics metrics;
  test_lod_metrics_init(&metrics);
  for (int iter = 0; iter < niter; ++iter) {
    free(gpu_values);
    gpu_values = lod_compute_gpu_u16(&plan, src, &metrics, method);
    CHECK(Fail, gpu_values);
  }
  test_lod_metrics_report(&metrics);

  {
    uint64_t total = plan.levels.ends[plan.nlod - 1];
    for (uint64_t i = 0; i < total; ++i) {
      if (gpu_values[i] != cpu_values[i]) {
        log_error("  FAIL at i=%llu: gpu=%u cpu=%u",
               (unsigned long long)i,
               (unsigned)gpu_values[i],
               (unsigned)cpu_values[i]);
        goto Fail;
      }
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(src);
  free(cpu_values);
  free(gpu_values);
  lod_plan_free(&plan);
  return ok ? 0 : 1;
}

static int
test_lod_gpu_u16(const char* label,
                 int ndim,
                 const uint64_t* shape,
                 uint32_t lod_mask,
                 int niter)
{
  return test_lod_gpu_u16_method(
    label, ndim, shape, lod_mask, niter, lod_reduce_mean);
}

// --- Dim0 accumulator tests ---

static int
test_accum_fold_u16(const char* label,
                    enum lod_reduce_method method,
                    int n_epochs)
{
  log_info("=== %s ===", label);
  int ok = 0;
  const uint64_t n_elements = 128;
  const int nlod = 2;

  CUstream stream = NULL;
  CUdeviceptr d_accum = 0, d_data = 0, d_out = 0;
  CUdeviceptr d_level_ids = 0, d_counts = 0;
  uint16_t* h_data = NULL;
  uint16_t* h_result = NULL;
  uint16_t* h_expected = NULL;
  uint8_t* h_level_ids = NULL;

  size_t accum_bpe = lod_accum_bpe(lod_dtype_u16, method);

  h_data = (uint16_t*)malloc(n_epochs * n_elements * sizeof(uint16_t));
  h_result = (uint16_t*)malloc(n_elements * sizeof(uint16_t));
  h_expected = (uint16_t*)malloc(n_elements * sizeof(uint16_t));
  h_level_ids = (uint8_t*)malloc(n_elements);
  CHECK(Fail, h_data && h_result && h_expected && h_level_ids);

  for (int e = 0; e < n_epochs; ++e)
    for (uint64_t i = 0; i < n_elements; ++i)
      h_data[e * n_elements + i] = (uint16_t)((e * 100 + i) & 0xFFFF);

  for (uint64_t i = 0; i < n_elements; ++i) {
    if (method == lod_reduce_mean) {
      // GPU fold uses overflow-safe (a+b)>>s per step, s=level=1
      uint16_t accum = h_data[i];
      int s = 1; // level_ids are all 1
      uint16_t mask = (uint16_t)((1u << s) - 1);
      for (int e = 1; e < n_epochs; ++e) {
        uint16_t b = h_data[e * n_elements + i];
        accum = (uint16_t)((accum >> s) + (b >> s)
                           + (((accum & mask) + (b & mask)) >> s));
      }
      h_expected[i] = accum;
    } else if (method == lod_reduce_min) {
      uint16_t best = h_data[i];
      for (int e = 1; e < n_epochs; ++e)
        if (h_data[e * n_elements + i] < best)
          best = h_data[e * n_elements + i];
      h_expected[i] = best;
    } else if (method == lod_reduce_max) {
      uint16_t best = h_data[i];
      for (int e = 1; e < n_epochs; ++e)
        if (h_data[e * n_elements + i] > best)
          best = h_data[e * n_elements + i];
      h_expected[i] = best;
    }
  }

  memset(h_level_ids, 1, n_elements);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuMemAlloc(&d_accum, n_elements * accum_bpe));
  CU(Fail, cuMemAlloc(&d_data, n_elements * sizeof(uint16_t)));
  CU(Fail, cuMemAlloc(&d_out, n_elements * sizeof(uint16_t)));
  CHECK(Fail, upload(&d_level_ids, h_level_ids, n_elements));
  CU(Fail, cuMemAlloc(&d_counts, nlod * sizeof(uint32_t)));

  {
    uint32_t counts[2] = { 0, 0 };
    for (int e = 0; e < n_epochs; ++e) {
      CU(Fail,
         cuMemcpyHtoD(
           d_data, h_data + e * n_elements, n_elements * sizeof(uint16_t)));
      CU(Fail, cuMemcpyHtoD(d_counts, counts, nlod * sizeof(uint32_t)));
      lod_accum_fold_fused(d_accum,
                           d_data,
                           d_level_ids,
                           d_counts,
                           lod_dtype_u16,
                           method,
                           n_elements,
                           stream);
      counts[1]++;
    }
  }

  lod_accum_emit(d_out,
                 d_accum,
                 lod_dtype_u16,
                 method,
                 n_elements,
                 (uint32_t)n_epochs,
                 stream);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(h_result, d_out, n_elements * sizeof(uint16_t)));

  for (uint64_t i = 0; i < n_elements; ++i) {
    if (h_result[i] != h_expected[i]) {
      log_error("  FAIL at i=%llu: gpu=%u expected=%u",
             (unsigned long long)i,
             (unsigned)h_result[i],
             (unsigned)h_expected[i]);
      goto Fail;
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(h_data);
  free(h_result);
  free(h_expected);
  free(h_level_ids);
  cuMemFree(d_accum);
  cuMemFree(d_data);
  cuMemFree(d_out);
  cuMemFree(d_level_ids);
  cuMemFree(d_counts);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

static int
test_accum_fold_f32(const char* label,
                    enum lod_reduce_method method,
                    int n_epochs)
{
  log_info("=== %s ===", label);
  int ok = 0;
  const uint64_t n_elements = 128;
  const int nlod = 2;

  CUstream stream = NULL;
  CUdeviceptr d_accum = 0, d_data = 0, d_result = 0;
  CUdeviceptr d_level_ids = 0, d_counts = 0;
  float* h_data = NULL;
  float* h_result = NULL;
  float* h_expected = NULL;
  uint8_t* h_level_ids = NULL;

  h_data = (float*)malloc(n_epochs * n_elements * sizeof(float));
  h_result = (float*)malloc(n_elements * sizeof(float));
  h_expected = (float*)malloc(n_elements * sizeof(float));
  h_level_ids = (uint8_t*)malloc(n_elements);
  CHECK(Fail, h_data && h_result && h_expected && h_level_ids);

  for (int e = 0; e < n_epochs; ++e)
    for (uint64_t i = 0; i < n_elements; ++i)
      h_data[e * n_elements + i] = (float)(e * 100 + i) + 0.5f;

  for (uint64_t i = 0; i < n_elements; ++i) {
    if (method == lod_reduce_mean) {
      float sum = 0;
      for (int e = 0; e < n_epochs; ++e)
        sum += h_data[e * n_elements + i];
      h_expected[i] = sum / (float)n_epochs;
    } else if (method == lod_reduce_min) {
      float best = h_data[i];
      for (int e = 1; e < n_epochs; ++e)
        if (h_data[e * n_elements + i] < best)
          best = h_data[e * n_elements + i];
      h_expected[i] = best;
    } else if (method == lod_reduce_max) {
      float best = h_data[i];
      for (int e = 1; e < n_epochs; ++e)
        if (h_data[e * n_elements + i] > best)
          best = h_data[e * n_elements + i];
      h_expected[i] = best;
    }
  }

  memset(h_level_ids, 1, n_elements);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuMemAlloc(&d_accum, n_elements * sizeof(float)));
  CU(Fail, cuMemAlloc(&d_data, n_elements * sizeof(float)));
  CHECK(Fail, upload(&d_level_ids, h_level_ids, n_elements));
  CU(Fail, cuMemAlloc(&d_counts, nlod * sizeof(uint32_t)));

  {
    uint32_t counts[2] = { 0, 0 };
    for (int e = 0; e < n_epochs; ++e) {
      CU(Fail,
         cuMemcpyHtoD(
           d_data, h_data + e * n_elements, n_elements * sizeof(float)));
      CU(Fail, cuMemcpyHtoD(d_counts, counts, nlod * sizeof(uint32_t)));
      lod_accum_fold_fused(d_accum,
                           d_data,
                           d_level_ids,
                           d_counts,
                           lod_dtype_f32,
                           method,
                           n_elements,
                           stream);
      counts[1]++;
    }
  }

  CU(Fail, cuMemAlloc(&d_result, n_elements * sizeof(float)));
  lod_accum_emit(d_result,
                 d_accum,
                 lod_dtype_f32,
                 method,
                 n_elements,
                 (uint32_t)n_epochs,
                 stream);
  CU(Fail, cuStreamSynchronize(stream));

  CU(Fail, cuMemcpyDtoH(h_result, d_result, n_elements * sizeof(float)));

  for (uint64_t i = 0; i < n_elements; ++i) {
    if (fabsf(h_result[i] - h_expected[i]) > 1e-3f) {
      log_error("  FAIL at i=%llu: gpu=%f expected=%f",
             (unsigned long long)i,
             h_result[i],
             h_expected[i]);
      goto Fail;
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(h_data);
  free(h_result);
  free(h_expected);
  free(h_level_ids);
  cuMemFree(d_accum);
  cuMemFree(d_data);
  cuMemFree(d_result);
  cuMemFree(d_level_ids);
  cuMemFree(d_counts);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

// Test fused fold kernel: 2 levels packed together, 4 epochs.
// Level 1 has period=2 (emits at epoch 2,4), level 2 has period=4 (emits at 4).
// After 4 epochs, both levels should match CPU reference.
static int
test_accum_fold_fused_u16(const char* label, enum lod_reduce_method method)
{
  log_info("=== %s ===", label);
  int ok = 0;
  const int n_epochs = 4;
  const int nlod = 3; // levels 0,1,2; fused operates on 1,2
  const uint64_t n_lv1 = 64, n_lv2 = 16;
  const uint64_t total = n_lv1 + n_lv2;

  CUstream stream = NULL;
  CUdeviceptr d_accum = 0, d_data = 0, d_level_ids = 0, d_counts = 0;
  uint16_t* h_data = NULL;
  uint16_t* h_result = NULL;

  size_t accum_bpe = lod_accum_bpe(lod_dtype_u16, method);
  uint8_t* h_level_ids = NULL;

  h_data = (uint16_t*)malloc(n_epochs * total * sizeof(uint16_t));
  h_result = (uint16_t*)malloc(total * sizeof(uint16_t));
  h_level_ids = (uint8_t*)malloc(total);
  CHECK(Fail, h_data && h_result && h_level_ids);

  // Generate test data
  for (int e = 0; e < n_epochs; ++e)
    for (uint64_t i = 0; i < total; ++i)
      h_data[e * total + i] = (uint16_t)((e * 100 + i * 7 + 3) & 0xFFFF);

  // Build level-ID buffer: first n_lv1 are level 1, next n_lv2 are level 2
  memset(h_level_ids, 1, n_lv1);
  memset(h_level_ids + n_lv1, 2, n_lv2);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuMemAlloc(&d_accum, total * accum_bpe));
  CU(Fail, cuMemAlloc(&d_data, total * sizeof(uint16_t)));
  CU(Fail, cuMemAlloc(&d_level_ids, total));
  CU(Fail, cuMemcpyHtoD(d_level_ids, h_level_ids, total));
  CU(Fail, cuMemAlloc(&d_counts, nlod * sizeof(uint32_t)));

  // Simulate 4 epochs with fused kernel
  uint32_t counts[3] = { 0, 0, 0 }; // [0] unused, [1] for lv1, [2] for lv2

  for (int e = 0; e < n_epochs; ++e) {
    CU(Fail,
       cuMemcpyHtoD(d_data, h_data + e * total, total * sizeof(uint16_t)));
    CU(Fail, cuMemcpyHtoD(d_counts, counts, nlod * sizeof(uint32_t)));

    lod_accum_fold_fused(d_accum,
                         d_data,
                         d_level_ids,
                         d_counts,
                         lod_dtype_u16,
                         method,
                         total,
                         stream);

    counts[1]++;
    counts[2]++;
  }

  // After 4 epochs: counts[1]=4, counts[2]=4. Both ready.
  // Emit level 1 (first n_lv1 elements of accum, count=4)
  {
    CUdeviceptr d_out = 0;
    CU(Fail, cuMemAlloc(&d_out, n_lv1 * sizeof(uint16_t)));
    lod_accum_emit(
      d_out, d_accum, lod_dtype_u16, method, n_lv1, counts[1], stream);
    CU(Fail, cuStreamSynchronize(stream));
    CU(Fail, cuMemcpyDtoH(h_result, d_out, n_lv1 * sizeof(uint16_t)));
    cuMemFree(d_out);

    // Verify level 1
    for (uint64_t i = 0; i < n_lv1; ++i) {
      uint16_t expected;
      if (method == lod_reduce_mean) {
        uint16_t accum = h_data[i];
        int s = 1; // level 1
        uint16_t mask = (uint16_t)((1u << s) - 1);
        for (int e = 1; e < n_epochs; ++e) {
          uint16_t b = h_data[e * total + i];
          accum = (uint16_t)((accum >> s) + (b >> s)
                             + (((accum & mask) + (b & mask)) >> s));
        }
        expected = accum;
      } else if (method == lod_reduce_min) {
        expected = h_data[i];
        for (int e = 1; e < n_epochs; ++e)
          if (h_data[e * total + i] < expected)
            expected = h_data[e * total + i];
      } else {
        expected = h_data[i];
        for (int e = 1; e < n_epochs; ++e)
          if (h_data[e * total + i] > expected)
            expected = h_data[e * total + i];
      }
      if (h_result[i] != expected) {
        log_error("  FAIL lv1 at i=%llu: gpu=%u expected=%u",
               (unsigned long long)i,
               (unsigned)h_result[i],
               (unsigned)expected);
        goto Fail;
      }
    }
  }

  // Emit level 2 (next n_lv2 elements of accum, offset by n_lv1)
  {
    CUdeviceptr d_out = 0;
    CUdeviceptr accum_lv2 = d_accum + n_lv1 * accum_bpe;
    CU(Fail, cuMemAlloc(&d_out, n_lv2 * sizeof(uint16_t)));
    lod_accum_emit(
      d_out, accum_lv2, lod_dtype_u16, method, n_lv2, counts[2], stream);
    CU(Fail, cuStreamSynchronize(stream));
    CU(Fail, cuMemcpyDtoH(h_result, d_out, n_lv2 * sizeof(uint16_t)));
    cuMemFree(d_out);

    // Verify level 2
    for (uint64_t i = 0; i < n_lv2; ++i) {
      uint64_t si = n_lv1 + i; // source index in packed data
      uint16_t expected;
      if (method == lod_reduce_mean) {
        uint16_t accum = h_data[si];
        int s = 2; // level 2
        uint16_t mask = (uint16_t)((1u << s) - 1);
        for (int e = 1; e < n_epochs; ++e) {
          uint16_t b = h_data[e * total + si];
          accum = (uint16_t)((accum >> s) + (b >> s)
                             + (((accum & mask) + (b & mask)) >> s));
        }
        expected = accum;
      } else if (method == lod_reduce_min) {
        expected = h_data[si];
        for (int e = 1; e < n_epochs; ++e)
          if (h_data[e * total + si] < expected)
            expected = h_data[e * total + si];
      } else {
        expected = h_data[si];
        for (int e = 1; e < n_epochs; ++e)
          if (h_data[e * total + si] > expected)
            expected = h_data[e * total + si];
      }
      if (h_result[i] != expected) {
        log_error("  FAIL lv2 at i=%llu: gpu=%u expected=%u",
               (unsigned long long)i,
               (unsigned)h_result[i],
               (unsigned)expected);
        goto Fail;
      }
    }
  }

  log_info("  PASS");
  ok = 1;
Fail:
  free(h_data);
  free(h_result);
  free(h_level_ids);
  cuMemFree(d_accum);
  cuMemFree(d_data);
  cuMemFree(d_level_ids);
  cuMemFree(d_counts);
  cuStreamDestroy(stream);
  return ok ? 0 : 1;
}

int
main(void)
{
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    log_error("CUDA init failed");
    return 1;
  }

  int nfail = 0;

  // All dims downsampled
  nfail += test_lod_gpu("gpu_lod_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3, 1);
  nfail += test_lod_gpu("gpu_lod_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7, 1);

  // Mixed: only some dims downsampled
  nfail += test_lod_gpu("gpu_lod_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5, 1);
  nfail += test_lod_gpu("gpu_lod_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2, 1);
  nfail += test_lod_gpu("gpu_lod_2d_d0", 2, (uint64_t[]){ 5, 3 }, 0x1, 1);
  nfail += test_lod_gpu("gpu_lod_2d_d1", 2, (uint64_t[]){ 3, 7 }, 0x2, 1);

  // No dims downsampled (trivial: nlod=1)
  nfail += test_lod_gpu("gpu_lod_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0, 1);
  // 1D
  nfail += test_lod_gpu("gpu_lod_1d", 1, (uint64_t[]){ 9 }, 0x1, 1);

  // Larger mixed
  nfail +=
    test_lod_gpu("gpu_lod_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA, 1);

  // Larger cases for throughput estimation
  nfail +=
    test_lod_gpu("gpu_lod_3d_256", 3, (uint64_t[]){ 256, 256, 256 }, 0x7, 10);
  nfail += test_lod_gpu(
    "gpu_lod_3d_mixed_large", 3, (uint64_t[]){ 64, 256, 256 }, 0x6, 10);

  // u16 tests (exact integer match)
  nfail +=
    test_lod_gpu_u16("gpu_lod_u16_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3, 1);
  nfail +=
    test_lod_gpu_u16("gpu_lod_u16_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7, 1);
  nfail +=
    test_lod_gpu_u16("gpu_lod_u16_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5, 1);
  nfail +=
    test_lod_gpu_u16("gpu_lod_u16_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2, 1);
  nfail += test_lod_gpu_u16(
    "gpu_lod_u16_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0, 1);
  nfail += test_lod_gpu_u16("gpu_lod_u16_1d", 1, (uint64_t[]){ 9 }, 0x1, 1);
  nfail += test_lod_gpu_u16(
    "gpu_lod_u16_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA, 1);

  // --- Reduce method tests (f32) ---
  {
    const uint64_t shape[] = { 3, 5 };
    const uint32_t mask = 0x3;
    nfail +=
      test_lod_gpu_method("reduce_min_f32", 2, shape, mask, 1, lod_reduce_min);
    nfail +=
      test_lod_gpu_method("reduce_max_f32", 2, shape, mask, 1, lod_reduce_max);
    nfail += test_lod_gpu_method(
      "reduce_median_f32", 2, shape, mask, 1, lod_reduce_median);
    nfail += test_lod_gpu_method(
      "reduce_max_sup_f32", 2, shape, mask, 1, lod_reduce_max_suppressed);
    nfail += test_lod_gpu_method(
      "reduce_min_sup_f32", 2, shape, mask, 1, lod_reduce_min_suppressed);
  }

  // --- Reduce method tests (u16) ---
  {
    const uint64_t shape[] = { 3, 5 };
    const uint32_t mask = 0x3;
    nfail += test_lod_gpu_u16_method(
      "reduce_min_u16", 2, shape, mask, 1, lod_reduce_min);
    nfail += test_lod_gpu_u16_method(
      "reduce_max_u16", 2, shape, mask, 1, lod_reduce_max);
    nfail += test_lod_gpu_u16_method(
      "reduce_median_u16", 2, shape, mask, 1, lod_reduce_median);
    nfail += test_lod_gpu_u16_method(
      "reduce_max_sup_u16", 2, shape, mask, 1, lod_reduce_max_suppressed);
    nfail += test_lod_gpu_u16_method(
      "reduce_min_sup_u16", 2, shape, mask, 1, lod_reduce_min_suppressed);
  }

  // --- Reduce method tests with mixed dims (3D, partial mask) ---
  {
    const uint64_t shape[] = { 6, 3, 5 };
    const uint32_t mask = 0x5;
    nfail += test_lod_gpu_method(
      "reduce_min_3d_d02", 3, shape, mask, 1, lod_reduce_min);
    nfail += test_lod_gpu_method(
      "reduce_max_3d_d02", 3, shape, mask, 1, lod_reduce_max);
    nfail += test_lod_gpu_u16_method(
      "reduce_min_u16_3d_d02", 3, shape, mask, 1, lod_reduce_min);
    nfail += test_lod_gpu_u16_method(
      "reduce_max_u16_3d_d02", 3, shape, mask, 1, lod_reduce_max);
  }

  // --- Dim0 accumulator tests ---
  nfail += test_accum_fold_u16("accum_mean_u16_4ep", lod_reduce_mean, 4);
  nfail += test_accum_fold_u16("accum_mean_u16_8ep", lod_reduce_mean, 8);
  nfail += test_accum_fold_u16("accum_min_u16_4ep", lod_reduce_min, 4);
  nfail += test_accum_fold_u16("accum_max_u16_4ep", lod_reduce_max, 4);
  nfail += test_accum_fold_u16("accum_mean_u16_1ep", lod_reduce_mean, 1);
  nfail += test_accum_fold_f32("accum_mean_f32_4ep", lod_reduce_mean, 4);
  nfail += test_accum_fold_f32("accum_min_f32_4ep", lod_reduce_min, 4);
  nfail += test_accum_fold_f32("accum_max_f32_4ep", lod_reduce_max, 4);

  // --- Fused dim0 accumulator tests ---
  nfail += test_accum_fold_fused_u16("accum_fused_mean_u16", lod_reduce_mean);
  nfail += test_accum_fold_fused_u16("accum_fused_min_u16", lod_reduce_min);
  nfail += test_accum_fold_fused_u16("accum_fused_max_u16", lod_reduce_max);

  log_info("\n%s (%d failures)", nfail ? "FAIL" : "ALL PASSED", nfail);

  cuCtxDestroy(ctx);
  return nfail ? 1 : 0;
}
