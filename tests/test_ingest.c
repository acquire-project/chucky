#include "gpu/prelude.cuda.h"
#include "gpu/stream.ingest.h"
#include "index.ops.util.h"
#include "util/prelude.h"

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

static struct gpu_pool_view
as_view(CUdeviceptr d)
{
  return (struct gpu_pool_view){ .p = (void*)(uintptr_t)d };
}

// --- Tests ---

// Incremental ingest: feed one epoch in two halves, verify pool.
static int
test_ingest_incremental(void)
{
  log_info("=== test_ingest_incremental ===");

  const int rank = 3;
  const uint64_t dim_sizes[] = { 4, 4, 6 };
  const uint64_t chunk_sizes[] = { 2, 2, 3 };
  const size_t bytes_per_element = 2;

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

  const size_t src_bytes = epoch_elements * bytes_per_element;
  const size_t pool_bytes = chunks_per_epoch * chunk_stride * bytes_per_element;
  const size_t half = src_bytes / 2;

  struct staging_state stage = { 0 };
  struct gpu_ordering ord = { 0 };
  struct tile_stream_layout layout = { 0 };
  struct tile_stream_layout_gpu layout_gpu = { 0 };
  CUstream h2d = 0, compute = 0;
  CUdeviceptr d_pool = 0;
  void* h_pool = NULL;
  uint16_t* h_src = NULL;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, gpu_ordering_init(&ord, compute) == 0);
  gpu_ordering_register_stream(&ord, GPU_STREAM_H2D, h2d);
  gpu_ordering_register_stream(&ord, GPU_STREAM_COMPUTE, compute);
  CHECK(Fail, ingest_init(&stage, half, &ord, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CU(Fail, cuMemsetD8(d_pool, 0, pool_bytes));

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
    const struct scatter_destination dst = {
      .first_epoch = as_view(d_pool),
      .epoch_bytes = pool_bytes,
      .epochs = 1,
    };

    memcpy(gpu_pool_at(&stage.h_pool, stage.current, 0).p, h_src, half);
    stage.bytes_written = half;
    CHECK(Fail,
          ingest_dispatch_scatter(&stage,
                                  &layout,
                                  &layout_gpu,
                                  dst,
                                  0,
                                  bytes_per_element,
                                  h2d,
                                  compute) == 0);

    struct gpu_pool_view h_in;
    CHECK(Fail,
          gpu_pool_host_acquire_produce(&stage.h_pool, stage.current, &h_in) ==
            0);
    memcpy(h_in.p, (uint8_t*)h_src + half, half);
    stage.bytes_written = half;
    CHECK(Fail,
          ingest_dispatch_scatter(&stage,
                                  &layout,
                                  &layout_gpu,
                                  dst,
                                  epoch_elements / 2,
                                  bytes_per_element,
                                  h2d,
                                  compute) == 0);
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
  gpu_ordering_destroy(&ord);
  destroy_layout_gpu(&layout_gpu);
  cu_mem_free(d_pool);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// One dispatch covering several epochs: each epoch's data must land in its own
// region of the pool (#173). first_element places the buffer's first element in
// the stream, so a value inside an epoch splits the dispatch unevenly.
static int
run_epochs_from(uint32_t n_epochs, uint64_t first_element)
{
  const int rank = 3;
  const uint64_t dim_sizes[] = { 4, 4, 6 };
  const uint64_t chunk_sizes[] = { 2, 2, 3 };
  const size_t bytes_per_element = 2;

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

  const size_t epoch_bytes =
    chunks_per_epoch * chunk_stride * bytes_per_element;
  const uint64_t src_elements = n_epochs * epoch_elements;
  const size_t src_bytes = src_elements * bytes_per_element;
  // The dispatch runs past its first epoch by however far first_element sits
  // into it, so allow one more region than the epochs it fully covers.
  const size_t pool_bytes = (n_epochs + 1) * epoch_bytes;

  struct staging_state stage = { 0 };
  struct gpu_ordering ord = { 0 };
  struct tile_stream_layout layout = { 0 };
  struct tile_stream_layout_gpu layout_gpu = { 0 };
  CUstream h2d = 0, compute = 0;
  CUdeviceptr d_pool = 0;
  void* h_pool = NULL;
  uint16_t* h_src = NULL;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, gpu_ordering_init(&ord, compute) == 0);
  gpu_ordering_register_stream(&ord, GPU_STREAM_H2D, h2d);
  gpu_ordering_register_stream(&ord, GPU_STREAM_COMPUTE, compute);
  CHECK(Fail, ingest_init(&stage, src_bytes, &ord, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CU(Fail, cuMemsetD8(d_pool, 0, pool_bytes));

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
  for (uint64_t i = 0; i < src_elements; ++i)
    h_src[i] = (uint16_t)((i + 1) & 0xFFFF);

  memcpy(gpu_pool_at(&stage.h_pool, 0, 0).p, h_src, src_bytes);
  stage.bytes_written = src_bytes;

  CHECK(Fail,
        ingest_dispatch_scatter(&stage,
                                &layout,
                                &layout_gpu,
                                (struct scatter_destination){
                                  .first_epoch = as_view(d_pool),
                                  .epoch_bytes = epoch_bytes,
                                  .epochs = n_epochs + 1,
                                },
                                first_element,
                                bytes_per_element,
                                h2d,
                                compute) == 0);

  CU(Fail, cuStreamSynchronize(compute));
  CU(Fail, cuStreamSynchronize(h2d));

  h_pool = calloc(1, pool_bytes);
  CHECK(Fail, h_pool);
  CU(Fail, cuMemcpyDtoH(h_pool, d_pool, pool_bytes));

  {
    const uint64_t first_epoch = first_element / epoch_elements;
    int errors = 0;
    for (uint64_t i = 0; i < src_elements; ++i) {
      const uint64_t element = first_element + i;
      const uint64_t region = element / epoch_elements - first_epoch;
      const uint64_t base = region * epoch_bytes / bytes_per_element;
      const uint64_t off = ravel(
        lifted_rank, lifted_shape, lifted_strides, element % epoch_elements);
      uint16_t dst_val = ((uint16_t*)h_pool)[base + off];
      if (dst_val != h_src[i]) {
        if (errors < 5)
          log_error("  element %lu: expected %u, got %u",
                    (unsigned long)element,
                    h_src[i],
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
  gpu_ordering_destroy(&ord);
  destroy_layout_gpu(&layout_gpu);
  cu_mem_free(d_pool);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// One epoch, starting at the beginning: the simplest shape the scatter takes.
static int
test_ingest_single_epoch(void)
{
  log_info("=== test_ingest_single_epoch ===");
  return run_epochs_from(1, 0);
}

static int
test_ingest_many_epochs_one_dispatch(void)
{
  log_info("=== test_ingest_many_epochs_one_dispatch ===");
  return run_epochs_from(3, 0);
}

// Starting one element in leaves every epoch after the first at an odd element
// offset, so the scatter's combined load starts partway into a word.
static int
test_ingest_many_epochs_from_mid_epoch(void)
{
  log_info("=== test_ingest_many_epochs_from_mid_epoch ===");
  return run_epochs_from(3, 1);
}

// Multiscale ingest: verify data arrives in linear buffer.
static int
test_ingest_multiscale(void)
{
  log_info("=== test_ingest_multiscale ===");

  const size_t bytes_per_element = 2;
  const uint64_t epoch_elements = 48;
  const size_t src_bytes = epoch_elements * bytes_per_element;

  struct staging_state stage = { 0 };
  struct gpu_ordering ord = { 0 };
  CUstream h2d = 0, compute = 0;
  CUdeviceptr d_linear = 0;
  uint16_t* h_src = NULL;
  void* h_out = NULL;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, gpu_ordering_init(&ord, compute) == 0);
  gpu_ordering_register_stream(&ord, GPU_STREAM_H2D, h2d);
  gpu_ordering_register_stream(&ord, GPU_STREAM_COMPUTE, compute);
  CHECK(Fail, ingest_init(&stage, src_bytes, &ord, compute) == 0);

  CU(Fail, cuMemAlloc(&d_linear, src_bytes));
  CU(Fail, cuMemsetD8(d_linear, 0, src_bytes));

  h_src = (uint16_t*)malloc(src_bytes);
  CHECK(Fail, h_src);
  for (uint64_t i = 0; i < epoch_elements; ++i)
    h_src[i] = (uint16_t)(i + 1);

  memcpy(gpu_pool_at(&stage.h_pool, 0, 0).p, h_src, src_bytes);
  stage.bytes_written = src_bytes;

  CHECK(
    Fail,
    ingest_dispatch_multiscale(
      &stage, d_linear, epoch_elements, 0, bytes_per_element, h2d, compute) ==
      0);

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
  gpu_ordering_destroy(&ord);
  cu_mem_free(d_linear);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "ingest_single_epoch", test_ingest_single_epoch },
              { "ingest_incremental", test_ingest_incremental },
              { "ingest_many_epochs_one_dispatch",
                test_ingest_many_epochs_one_dispatch },
              { "ingest_many_epochs_from_mid_epoch",
                test_ingest_many_epochs_from_mid_epoch },
              { "ingest_multiscale", test_ingest_multiscale }, )
