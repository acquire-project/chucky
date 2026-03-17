#include "index.ops.util.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.ingest.h"

#include "test_runner.h"

#include <stdlib.h>
#include <string.h>

// ingest_init / ingest_destroy from stream_ingest.h

static int
upload_layout_gpu(struct tile_stream_layout_gpu* gpu,
                  uint8_t lifted_rank,
                  const uint64_t* lifted_shape,
                  const int64_t* lifted_strides)
{
  size_t sb = lifted_rank * sizeof(uint64_t);
  size_t stb = lifted_rank * sizeof(int64_t);
  CU(Fail, cuMemAlloc((CUdeviceptr*)&gpu->d_lifted_shape, sb));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&gpu->d_lifted_strides, stb));
  CU(Fail, cuMemcpyHtoD((CUdeviceptr)gpu->d_lifted_shape, lifted_shape, sb));
  CU(Fail,
     cuMemcpyHtoD((CUdeviceptr)gpu->d_lifted_strides, lifted_strides, stb));
  return 0;
Fail:
  return 1;
}

static void
destroy_layout_gpu(struct tile_stream_layout_gpu* gpu)
{
  cu_mem_free((CUdeviceptr)gpu->d_lifted_shape);
  cu_mem_free((CUdeviceptr)gpu->d_lifted_strides);
}

// --- Tests ---

// Single full-epoch ingest: write all epoch elements into staging, dispatch
// once, verify chunks in pool match CPU ravel reference.
static int
test_ingest_single_epoch(void)
{
  log_info("=== test_ingest_single_epoch ===");

  const int rank = 3;
  const uint64_t dim_sizes[] = { 4, 4, 6 };
  const uint64_t chunk_sizes[] = { 2, 2, 3 };
  const size_t bpe = 2;

  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];
  uint64_t chunk_elements, chunk_stride, chunks_per_epoch, epoch_elements;

  build_lifted_layout(rank,
                      dim_sizes,
                      chunk_sizes,
                      NULL,
                      &lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      &chunk_elements,
                      &chunk_stride,
                      &chunks_per_epoch,
                      &epoch_elements);

  const size_t src_bytes = epoch_elements * bpe;
  const size_t pool_bytes = chunks_per_epoch * chunk_stride * bpe;

  log_info("  epoch_elements=%lu chunks_per_epoch=%lu pool_bytes=%lu",
           (unsigned long)epoch_elements,
           (unsigned long)chunks_per_epoch,
           (unsigned long)pool_bytes);

  struct staging_state stage = { 0 };
  struct tile_stream_layout layout = { 0 };
  struct tile_stream_layout_gpu layout_gpu = { 0 };
  CUstream h2d = 0, compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent pool_ready = 0;
  void* h_pool = NULL;
  uint16_t* h_src = NULL;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, ingest_init(&stage, src_bytes, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CU(Fail, cuMemsetD8(d_pool, 0, pool_bytes));
  CU(Fail, cuEventCreate(&pool_ready, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(pool_ready, compute));

  layout.lifted_rank = lifted_rank;
  memcpy(layout.lifted_shape, lifted_shape, lifted_rank * sizeof(uint64_t));
  memcpy(layout.lifted_strides, lifted_strides, lifted_rank * sizeof(int64_t));
  layout.chunk_elements = chunk_elements;
  layout.chunk_stride = chunk_stride;
  layout.chunks_per_epoch = chunks_per_epoch;
  layout.epoch_elements = epoch_elements;
  CHECK(Fail,
        upload_layout_gpu(
          &layout_gpu, lifted_rank, lifted_shape, lifted_strides) == 0);

  h_src = (uint16_t*)malloc(src_bytes);
  CHECK(Fail, h_src);
  for (uint64_t i = 0; i < epoch_elements; ++i)
    h_src[i] = (uint16_t)(i & 0xFFFF);

  memcpy(stage.slot[0].h_in, h_src, src_bytes);
  stage.bytes_written = src_bytes;

  {
    uint64_t cursor = 0;
    CHECK(Fail,
          ingest_dispatch_scatter(&stage,
                                  &layout,
                                  &layout_gpu,
                                  (void*)d_pool,
                                  pool_ready,
                                  &cursor,
                                  bpe,
                                  h2d,
                                  compute) == 0);
    CHECK(Fail, cursor == epoch_elements);
  }

  CU(Fail, cuStreamSynchronize(compute));
  CU(Fail, cuStreamSynchronize(h2d));

  h_pool = calloc(1, pool_bytes);
  CHECK(Fail, h_pool);
  CU(Fail, cuMemcpyDtoH(h_pool, d_pool, pool_bytes));

  {
    int errors = 0;
    for (uint64_t i = 0; i < epoch_elements; ++i) {
      uint64_t off = ravel(lifted_rank, lifted_shape, lifted_strides, i);
      uint16_t src_val = h_src[i];
      uint16_t dst_val = ((uint16_t*)h_pool)[off];
      if (dst_val != src_val) {
        if (errors < 5)
          log_error("  elem %lu: expected pool[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)off,
                    src_val,
                    dst_val);
        errors++;
      }
    }
    if (errors > 0) {
      log_error("  %d mismatches", errors);
      goto Fail;
    }
  }

  ok = 1;

Fail:
  free(h_src);
  free(h_pool);
  ingest_destroy(&stage);
  destroy_layout_gpu(&layout_gpu);
  cu_mem_free(d_pool);
  cu_event_destroy(pool_ready);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Incremental ingest: feed one epoch in two halves, verify pool.
static int
test_ingest_incremental(void)
{
  log_info("=== test_ingest_incremental ===");

  const int rank = 3;
  const uint64_t dim_sizes[] = { 4, 4, 6 };
  const uint64_t chunk_sizes[] = { 2, 2, 3 };
  const size_t bpe = 2;

  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];
  uint64_t chunk_elements, chunk_stride, chunks_per_epoch, epoch_elements;

  build_lifted_layout(rank,
                      dim_sizes,
                      chunk_sizes,
                      NULL,
                      &lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      &chunk_elements,
                      &chunk_stride,
                      &chunks_per_epoch,
                      &epoch_elements);

  const size_t src_bytes = epoch_elements * bpe;
  const size_t pool_bytes = chunks_per_epoch * chunk_stride * bpe;
  const size_t half = src_bytes / 2;

  struct staging_state stage = { 0 };
  struct tile_stream_layout layout = { 0 };
  struct tile_stream_layout_gpu layout_gpu = { 0 };
  CUstream h2d = 0, compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent pool_ready = 0;
  void* h_pool = NULL;
  uint16_t* h_src = NULL;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, ingest_init(&stage, half, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CU(Fail, cuMemsetD8(d_pool, 0, pool_bytes));
  CU(Fail, cuEventCreate(&pool_ready, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(pool_ready, compute));

  layout.lifted_rank = lifted_rank;
  memcpy(layout.lifted_shape, lifted_shape, lifted_rank * sizeof(uint64_t));
  memcpy(layout.lifted_strides, lifted_strides, lifted_rank * sizeof(int64_t));
  layout.chunk_elements = chunk_elements;
  layout.chunk_stride = chunk_stride;
  layout.chunks_per_epoch = chunks_per_epoch;
  layout.epoch_elements = epoch_elements;
  CHECK(Fail,
        upload_layout_gpu(
          &layout_gpu, lifted_rank, lifted_shape, lifted_strides) == 0);

  h_src = (uint16_t*)malloc(src_bytes);
  CHECK(Fail, h_src);
  for (uint64_t i = 0; i < epoch_elements; ++i)
    h_src[i] = (uint16_t)(i & 0xFFFF);

  {
    uint64_t cursor = 0;

    memcpy(stage.slot[stage.current].h_in, h_src, half);
    stage.bytes_written = half;
    CHECK(Fail,
          ingest_dispatch_scatter(&stage,
                                  &layout,
                                  &layout_gpu,
                                  (void*)d_pool,
                                  pool_ready,
                                  &cursor,
                                  bpe,
                                  h2d,
                                  compute) == 0);
    CHECK(Fail, cursor == epoch_elements / 2);

    CU(Fail, cuEventSynchronize(stage.slot[stage.current].t_h2d_end));
    memcpy(stage.slot[stage.current].h_in, (uint8_t*)h_src + half, half);
    stage.bytes_written = half;
    CHECK(Fail,
          ingest_dispatch_scatter(&stage,
                                  &layout,
                                  &layout_gpu,
                                  (void*)d_pool,
                                  pool_ready,
                                  &cursor,
                                  bpe,
                                  h2d,
                                  compute) == 0);
    CHECK(Fail, cursor == epoch_elements);
  }

  CU(Fail, cuStreamSynchronize(compute));
  CU(Fail, cuStreamSynchronize(h2d));

  h_pool = calloc(1, pool_bytes);
  CHECK(Fail, h_pool);
  CU(Fail, cuMemcpyDtoH(h_pool, d_pool, pool_bytes));

  {
    int errors = 0;
    for (uint64_t i = 0; i < epoch_elements; ++i) {
      uint64_t off = ravel(lifted_rank, lifted_shape, lifted_strides, i);
      uint16_t src_val = h_src[i];
      uint16_t dst_val = ((uint16_t*)h_pool)[off];
      if (dst_val != src_val) {
        if (errors < 5)
          log_error("  elem %lu: expected pool[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)off,
                    src_val,
                    dst_val);
        errors++;
      }
    }
    if (errors > 0) {
      log_error("  %d mismatches", errors);
      goto Fail;
    }
  }

  ok = 1;

Fail:
  free(h_src);
  free(h_pool);
  ingest_destroy(&stage);
  destroy_layout_gpu(&layout_gpu);
  cu_mem_free(d_pool);
  cu_event_destroy(pool_ready);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Multiscale ingest: verify data arrives in linear buffer.
static int
test_ingest_multiscale(void)
{
  log_info("=== test_ingest_multiscale ===");

  const size_t bpe = 2;
  const uint64_t epoch_elements = 48;
  const size_t src_bytes = epoch_elements * bpe;

  struct staging_state stage = { 0 };
  CUstream h2d = 0, compute = 0;
  CUdeviceptr d_linear = 0;
  uint16_t* h_src = NULL;
  void* h_out = NULL;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, ingest_init(&stage, src_bytes, compute) == 0);

  CU(Fail, cuMemAlloc(&d_linear, src_bytes));
  CU(Fail, cuMemsetD8(d_linear, 0, src_bytes));

  h_src = (uint16_t*)malloc(src_bytes);
  CHECK(Fail, h_src);
  for (uint64_t i = 0; i < epoch_elements; ++i)
    h_src[i] = (uint16_t)(i + 1);

  memcpy(stage.slot[0].h_in, h_src, src_bytes);
  stage.bytes_written = src_bytes;

  {
    uint64_t cursor = 0;
    CHECK(Fail,
          ingest_dispatch_multiscale(
            &stage, d_linear, epoch_elements, &cursor, bpe, h2d, compute) == 0);
    CHECK(Fail, cursor == epoch_elements);
  }

  CU(Fail, cuStreamSynchronize(compute));
  CU(Fail, cuStreamSynchronize(h2d));

  h_out = malloc(src_bytes);
  CHECK(Fail, h_out);
  CU(Fail, cuMemcpyDtoH(h_out, d_linear, src_bytes));

  CHECK(Fail, memcmp(h_out, h_src, src_bytes) == 0);

  ok = 1;

Fail:
  free(h_src);
  free(h_out);
  ingest_destroy(&stage);
  cu_mem_free(d_linear);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "ingest_single_epoch", test_ingest_single_epoch },
              { "ingest_incremental", test_ingest_incremental },
              { "ingest_multiscale", test_ingest_multiscale }, )
