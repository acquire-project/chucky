// GPU LOD test: compare GPU kernels against CPU reference from morton.util.c

#include "morton.util.c"

#include "lod.h"
#include "metric.cuda.h"
#include "prelude.cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void
report_metric(const struct stream_metric* m)
{
  if (m->count == 0)
    return;
  float avg_ms = m->ms / (float)m->count;
  double avg_bytes = m->total_bytes / (double)m->count;
  printf("  %-8s %7.3f ms  %6.2f GB/s  (%d iters)\n",
         m->name, avg_ms, avg_bytes / ((double)avg_ms * 1e6), m->count);
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
                struct test_lod_metrics* metrics)
{
  float* result = NULL;
  CUstream stream = NULL;
  CUevent ev_start = NULL, ev_scatter = NULL, ev_done = NULL;

  CUdeviceptr d_src = 0, d_values = 0;
  CUdeviceptr d_full_shape = 0, d_lod_shape = 0;
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
        upload(&d_full_shape, p->shapes[0], p->ndim * sizeof(uint64_t)));
  if (p->lod_ndim > 0)
    CHECK(Fail,
          upload(&d_lod_shape, p->lod_shapes[0],
                 p->lod_ndim * sizeof(uint64_t)));

  CU(Fail, cuEventRecord(ev_start, stream));

  lod_scatter(d_values, d_src, lod_dtype_f32, p->ndim, n_elements,
              d_full_shape, d_lod_shape, p->lod_ndim,
              p->lod_shapes[0], p->lod_mask, p->lod_counts[0], stream);

  CU(Fail, cuEventRecord(ev_scatter, stream));

  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    cuMemFree(d_child_shape);
    cuMemFree(d_parent_shape);
    d_child_shape = 0;
    d_parent_shape = 0;

    CHECK(Fail,
          upload(&d_child_shape, p->lod_shapes[l],
                 p->lod_ndim * sizeof(uint64_t)));
    CHECK(Fail,
          upload(&d_parent_shape, p->lod_shapes[l + 1],
                 p->lod_ndim * sizeof(uint64_t)));

    cuMemFree(d_ends);
    d_ends = 0;
    CU(Fail, cuMemAlloc(&d_ends, n_parents * sizeof(uint64_t)));

    lod_fill_ends_gpu(d_ends, p->lod_ndim,
                      d_child_shape, d_parent_shape,
                      p->lod_shapes[l], p->lod_shapes[l + 1],
                      n_parents, stream);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    lod_reduce(d_values, d_ends, lod_dtype_f32,
               src_level.beg, dst_level.beg,
               p->lod_counts[l], p->lod_counts[l + 1],
               p->batch_count, stream);
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
  cuMemFree(d_full_shape);
  cuMemFree(d_lod_shape);
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
test_lod_gpu(const char* label,
             int ndim,
             const uint64_t* shape,
             uint8_t lod_mask,
             int niter)
{
  printf("--- %s ---\n", label);
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

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, NULL, lod_mask, MAX_LOD));
  printf("  lod_mask=0x%x  lod_ndim=%d  batch_ndim=%d  batch_count=%llu  nlod=%d\n",
         lod_mask, plan.lod_ndim, plan.batch_ndim,
         (unsigned long long)plan.batch_count, plan.nlod);

  CHECK(Fail, lod_compute(&plan, src, &cpu_values));

  struct test_lod_metrics metrics;
  test_lod_metrics_init(&metrics);
  for (int iter = 0; iter < niter; ++iter) {
    free(gpu_values);
    gpu_values = lod_compute_gpu(&plan, src, &metrics);
    CHECK(Fail, gpu_values);
  }
  test_lod_metrics_report(&metrics);

  {
    uint64_t total = plan.levels.ends[plan.nlod - 1];
    for (uint64_t i = 0; i < total; ++i) {
      if (fabsf(gpu_values[i] - cpu_values[i]) > 1e-5f) {
        printf("  FAIL at i=%llu: gpu=%f cpu=%f\n",
               (unsigned long long)i, gpu_values[i], cpu_values[i]);
        goto Fail;
      }
    }
  }

  printf("  PASS\n");
  ok = 1;
Fail:
  free(src);
  free(cpu_values);
  free(gpu_values);
  lod_plan_free(&plan);
  return ok;
}

// --- u16 CPU reference ---

static void
lod_scatter_cpu_u16(const struct lod_plan* p,
                    const uint16_t* src,
                    uint16_t* dst)
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
    uint64_t pos =
      morton_rank(p->lod_ndim, p->lod_shapes[0], lod_coords, 0);
    dst[b * p->lod_counts[0] + pos] = src[i];
  }
}

static void
lod_reduce_cpu_u16(const struct lod_plan* p, uint16_t* values)
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
        uint64_t len = end - start;
        uint32_t sum = 0;
        for (uint64_t j = start; j < end; ++j)
          sum += values[src_base + j];
        values[dst_base + i] = (uint16_t)(sum / (uint32_t)len);
      }
    }
  }
}

static int
lod_compute_u16(const struct lod_plan* p,
                const uint16_t* src,
                uint16_t** out_values)
{
  int ok = 0;
  *out_values = NULL;

  uint64_t total_vals = p->levels.ends[p->nlod - 1];
  uint16_t* values = (uint16_t*)malloc(total_vals * sizeof(uint16_t));
  CHECK(Error, values);
  *out_values = values;

  lod_scatter_cpu_u16(p, src, values);
  lod_reduce_cpu_u16(p, values);

  ok = 1;
Error:
  if (!ok) {
    free(*out_values);
    *out_values = NULL;
  }
  return ok;
}

// Run u16 LOD computation on GPU.
static uint16_t*
lod_compute_gpu_u16(const struct lod_plan* p,
                    const uint16_t* src,
                    struct test_lod_metrics* metrics)
{
  uint16_t* result = NULL;
  CUstream stream = NULL;
  CUevent ev_start = NULL, ev_scatter = NULL, ev_done = NULL;

  CUdeviceptr d_src = 0, d_values = 0;
  CUdeviceptr d_full_shape = 0, d_lod_shape = 0;
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
        upload(&d_full_shape, p->shapes[0], p->ndim * sizeof(uint64_t)));
  if (p->lod_ndim > 0)
    CHECK(Fail,
          upload(&d_lod_shape, p->lod_shapes[0],
                 p->lod_ndim * sizeof(uint64_t)));

  CU(Fail, cuEventRecord(ev_start, stream));

  lod_scatter(d_values, d_src, lod_dtype_u16, p->ndim, n_elements,
              d_full_shape, d_lod_shape, p->lod_ndim,
              p->lod_shapes[0], p->lod_mask, p->lod_counts[0], stream);

  CU(Fail, cuEventRecord(ev_scatter, stream));

  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    cuMemFree(d_child_shape);
    cuMemFree(d_parent_shape);
    d_child_shape = 0;
    d_parent_shape = 0;

    CHECK(Fail,
          upload(&d_child_shape, p->lod_shapes[l],
                 p->lod_ndim * sizeof(uint64_t)));
    CHECK(Fail,
          upload(&d_parent_shape, p->lod_shapes[l + 1],
                 p->lod_ndim * sizeof(uint64_t)));

    cuMemFree(d_ends);
    d_ends = 0;
    CU(Fail, cuMemAlloc(&d_ends, n_parents * sizeof(uint64_t)));

    lod_fill_ends_gpu(d_ends, p->lod_ndim,
                      d_child_shape, d_parent_shape,
                      p->lod_shapes[l], p->lod_shapes[l + 1],
                      n_parents, stream);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    lod_reduce(d_values, d_ends, lod_dtype_u16,
               src_level.beg, dst_level.beg,
               p->lod_counts[l], p->lod_counts[l + 1],
               p->batch_count, stream);
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
  cuMemFree(d_full_shape);
  cuMemFree(d_lod_shape);
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
test_lod_gpu_u16(const char* label,
                 int ndim,
                 const uint64_t* shape,
                 uint8_t lod_mask,
                 int niter)
{
  printf("--- %s ---\n", label);
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

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, NULL, lod_mask, MAX_LOD));
  printf("  lod_mask=0x%x  lod_ndim=%d  batch_ndim=%d  batch_count=%llu  nlod=%d\n",
         lod_mask, plan.lod_ndim, plan.batch_ndim,
         (unsigned long long)plan.batch_count, plan.nlod);

  CHECK(Fail, lod_compute_u16(&plan, src, &cpu_values));

  struct test_lod_metrics metrics;
  test_lod_metrics_init(&metrics);
  for (int iter = 0; iter < niter; ++iter) {
    free(gpu_values);
    gpu_values = lod_compute_gpu_u16(&plan, src, &metrics);
    CHECK(Fail, gpu_values);
  }
  test_lod_metrics_report(&metrics);

  {
    uint64_t total = plan.levels.ends[plan.nlod - 1];
    for (uint64_t i = 0; i < total; ++i) {
      if (gpu_values[i] != cpu_values[i]) {
        printf("  FAIL at i=%llu: gpu=%u cpu=%u\n",
               (unsigned long long)i,
               (unsigned)gpu_values[i], (unsigned)cpu_values[i]);
        goto Fail;
      }
    }
  }

  printf("  PASS\n");
  ok = 1;
Fail:
  free(src);
  free(cpu_values);
  free(gpu_values);
  lod_plan_free(&plan);
  return ok;
}

int
main(void)
{
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS ||
      cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    printf("CUDA init failed\n");
    return 1;
  }

  int nfail = 0;

  // All dims downsampled
  nfail +=
    !test_lod_gpu("gpu_lod_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3, 1);
  nfail +=
    !test_lod_gpu("gpu_lod_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7, 1);

  // Mixed: only some dims downsampled
  nfail +=
    !test_lod_gpu("gpu_lod_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5, 1);
  nfail +=
    !test_lod_gpu("gpu_lod_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2, 1);
  nfail +=
    !test_lod_gpu("gpu_lod_2d_d0", 2, (uint64_t[]){ 5, 3 }, 0x1, 1);
  nfail +=
    !test_lod_gpu("gpu_lod_2d_d1", 2, (uint64_t[]){ 3, 7 }, 0x2, 1);

  // No dims downsampled (trivial: nlod=1)
  nfail +=
    !test_lod_gpu("gpu_lod_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0, 1);
  // 1D
  nfail +=
    !test_lod_gpu("gpu_lod_1d", 1, (uint64_t[]){ 9 }, 0x1, 1);

  // Larger mixed
  nfail +=
    !test_lod_gpu("gpu_lod_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA, 1);

  // Larger cases for throughput estimation
  nfail +=
    !test_lod_gpu("gpu_lod_3d_256", 3, (uint64_t[]){ 256, 256, 256 }, 0x7, 10);
  nfail +=
    !test_lod_gpu("gpu_lod_3d_mixed_large", 3,
                  (uint64_t[]){ 64, 256, 256 }, 0x6, 10);

  // u16 tests (exact integer match)
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3, 1);
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7, 1);
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5, 1);
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2, 1);
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0, 1);
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_1d", 1, (uint64_t[]){ 9 }, 0x1, 1);
  nfail +=
    !test_lod_gpu_u16("gpu_lod_u16_4d_d13", 4,
                      (uint64_t[]){ 3, 8, 2, 6 }, 0xA, 1);

  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);

  cuCtxDestroy(ctx);
  return nfail ? 1 : 0;
}
