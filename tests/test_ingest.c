#include "gpu/prelude.cuda.h"
#include "gpu/stream.ingest.h"
#include "index.ops.util.h"
#include "stream/config.h"
#include "util/prelude.h"

#include "test_metric_check.h"
#include "test_runner.h"

#include <stdlib.h>
#include <string.h>

// ingest_init / ingest_destroy from stream_ingest.h

static struct gpu_pool_view
as_view(CUdeviceptr d)
{
  return (struct gpu_pool_view){ .p = (void*)(uintptr_t)d };
}

// A timestamp for the dispatches' timing events to be ordered against. The
// synchronize matters: events seeded when the staging state was created must
// run first, or the marker lands before them and nothing below can fail.
static int
record_marker(CUevent* marker, CUstream compute, CUstream h2d)
{
  CU(Error, cuEventCreate(marker, CU_EVENT_DEFAULT));
  CU(Error, cuStreamSynchronize(compute));
  CU(Error, cuEventRecord(*marker, h2d));
  return 0;

Error:
  return 1;
}

static int
event_is_after(CUevent marker, CUevent e, const char* what, int which)
{
  float ms = 0;
  CU(Error, cuEventElapsedTime(&ms, marker, e));
  if (ms < 0.0f) {
    log_error("  %s %d reported a stale event", what, which);
    return 0;
  }
  return 1;

Error:
  return 0;
}

// Setup seeds these events and a reuse overwrites them in place, so a dispatch
// that failed to record one still reports a measurement, taken from the stale
// event. Counting cannot see that; ordering against a marker that predates the
// dispatches can.
static int
timing_events_are_fresh(struct staging_state* stage, CUevent marker)
{
  int ok = 1;
  for (int i = 0; i < (int)countof(stage->slot); ++i)
    if (stage->slot[i].h2d_pending)
      ok &= event_is_after(marker, stage->slot[i].t_h2d_start, "H2D slot", i);
  for (int i = 0; i < SCATTER_TIMING_SLOTS; ++i)
    if (stage->timing[i].pending)
      ok &= event_is_after(marker, stage->timing[i].t_end, "Scatter entry", i);
  return ok;
}

// Reports every broken stage rather than the first: a rerun costs a GPU.
// `marker` must predate every dispatch, and both streams must be synchronized.
static int
check_ingest_timing(struct staging_state* stage,
                    CUevent marker,
                    int h2d_count,
                    int scatter_count,
                    uint64_t expected_lost)
{
  struct stream_metric h2d = mk_stream_metric("H2D", METRIC_OWNER_H2D);
  struct stream_metric scatter =
    mk_stream_metric("Scatter", METRIC_OWNER_COMPUTE);

  // Read before collecting, which clears the pending flags these key on.
  int ok = timing_events_are_fresh(stage, marker);

  ingest_collect_h2d_timing(stage, &h2d);
  ingest_collect_scatter_timing(stage, &scatter);

  ok &= metric_arrived_timed(&h2d, h2d_count);
  ok &= metric_arrived_timed(&scatter, scatter_count);
  if (stage->scatter_samples_lost != expected_lost) {
    log_error("  scatter ring wrapped %llu times, expected %llu",
              (unsigned long long)stage->scatter_samples_lost,
              (unsigned long long)expected_lost);
    ok = 0;
  }
  return ok;
}

// --- Tests ---

// Incremental ingest: feed one epoch in two halves, verify pool.
static int
test_ingest_incremental(void)
{
  log_info("=== test_ingest_incremental ===");

  const uint8_t rank = 3;
  const uint64_t dim_sizes[] = { 4, 4, 6 };
  const uint64_t chunk_sizes[] = { 2, 2, 3 };
  const size_t bytes_per_element = 2;

  struct tile_stream_layout layout;
  if (test_level_layout(&layout,
                        rank,
                        1,
                        dim_sizes,
                        chunk_sizes,
                        NULL,
                        bytes_per_element,
                        TEST_CHUNK_ALIGNMENT))
    return 1;

  const uint64_t epoch_elements = layout.epoch_elements;
  const size_t src_bytes = epoch_elements * bytes_per_element;
  const size_t pool_bytes =
    layout.chunks_per_epoch * layout.chunk_stride * bytes_per_element;
  const size_t half = src_bytes / 2;

  struct staging_state stage = { 0 };
  struct gpu_ordering ord = { 0 };
  CUstream h2d = 0, compute = 0;
  CUevent marker = 0;
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
  CHECK(Fail, record_marker(&marker, compute, h2d) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  // On the scatter's stream, so the clear cannot land after it.
  CU(Fail, cuMemsetD8Async(d_pool, 0, pool_bytes, compute));

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
          ingest_dispatch_scatter(
            &stage, &layout, dst, 0, bytes_per_element, h2d, compute) == 0);

    struct gpu_pool_view h_in;
    CHECK(Fail,
          gpu_pool_host_acquire_produce(&stage.h_pool, stage.current, &h_in) ==
            0);
    memcpy(h_in.p, (uint8_t*)h_src + half, half);
    stage.bytes_written = half;
    CHECK(Fail,
          ingest_dispatch_scatter(&stage,
                                  &layout,
                                  dst,
                                  epoch_elements / 2,
                                  bytes_per_element,
                                  h2d,
                                  compute) == 0);
  }

  CU(Fail, cuStreamSynchronize(compute));
  CU(Fail, cuStreamSynchronize(h2d));

  CHECK(Fail, check_ingest_timing(&stage, marker, 2, 2, 0));

  h_pool = calloc(1, pool_bytes);
  CHECK(Fail, h_pool);
  CU(Fail, cuMemcpyDtoH(h_pool, d_pool, pool_bytes));

  {
    int errors = 0;
    for (uint64_t i = 0; i < epoch_elements; ++i) {
      const uint64_t off =
        expected_scatter_offset(layout.lifted_rank,
                                layout.lifted_shape,
                                layout.lifted_strides,
                                epoch_elements,
                                pool_bytes / bytes_per_element,
                                i);
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
  cu_stream_sync(compute);
  cu_stream_sync(h2d);
  free(h_src);
  free(h_pool);
  ingest_destroy(&stage);
  gpu_ordering_destroy(&ord);
  cu_event_destroy(marker);
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
  const uint8_t rank = 3;
  const uint64_t dim_sizes[] = { 4, 4, 6 };
  const uint64_t chunk_sizes[] = { 2, 2, 3 };
  const size_t bytes_per_element = 2;

  struct tile_stream_layout layout;
  if (test_level_layout(&layout,
                        rank,
                        1,
                        dim_sizes,
                        chunk_sizes,
                        NULL,
                        bytes_per_element,
                        TEST_CHUNK_ALIGNMENT))
    return 1;

  const uint64_t epoch_elements = layout.epoch_elements;
  const size_t epoch_bytes =
    (layout.chunks_per_epoch * layout.chunk_stride + TEST_REGION_PAD_ELEMENTS) *
    bytes_per_element;
  const uint64_t src_elements = n_epochs * epoch_elements;
  const size_t src_bytes = src_elements * bytes_per_element;
  // The dispatch runs past its first epoch by however far first_element sits
  // into it, so allow one more region than the epochs it fully covers.
  const size_t pool_bytes = (n_epochs + 1) * epoch_bytes;

  struct staging_state stage = { 0 };
  struct gpu_ordering ord = { 0 };
  CUstream h2d = 0, compute = 0;
  CUevent marker = 0;
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
  CHECK(Fail, record_marker(&marker, compute, h2d) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  // On the scatter's stream, so the clear cannot land after it.
  CU(Fail, cuMemsetD8Async(d_pool, 0, pool_bytes, compute));

  h_src = (uint16_t*)malloc(src_bytes);
  CHECK(Fail, h_src);
  for (uint64_t i = 0; i < src_elements; ++i)
    h_src[i] = (uint16_t)((i + 1) & 0xFFFF);

  memcpy(gpu_pool_at(&stage.h_pool, 0, 0).p, h_src, src_bytes);
  stage.bytes_written = src_bytes;

  CHECK(Fail,
        ingest_dispatch_scatter(&stage,
                                &layout,
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

  CHECK(Fail, check_ingest_timing(&stage, marker, 1, 1, 0));

  h_pool = calloc(1, pool_bytes);
  CHECK(Fail, h_pool);
  CU(Fail, cuMemcpyDtoH(h_pool, d_pool, pool_bytes));

  {
    int errors = 0;
    for (uint64_t i = 0; i < src_elements; ++i) {
      const uint64_t off =
        expected_scatter_offset(layout.lifted_rank,
                                layout.lifted_shape,
                                layout.lifted_strides,
                                epoch_elements,
                                epoch_bytes / bytes_per_element,
                                first_element % epoch_elements + i);
      uint16_t dst_val = ((uint16_t*)h_pool)[off];
      if (dst_val != h_src[i]) {
        if (errors < 5)
          log_error("  element %lu: expected %u, got %u",
                    (unsigned long)(first_element + i),
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
  cu_stream_sync(compute);
  cu_stream_sync(h2d);
  free(h_src);
  free(h_pool);
  ingest_destroy(&stage);
  gpu_ordering_destroy(&ord);
  cu_event_destroy(marker);
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

// Starting one element in splits the dispatch unevenly across epochs, so the
// first and last regions each take part of one.
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
  CUevent marker = 0;
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
  CHECK(Fail, record_marker(&marker, compute, h2d) == 0);

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

  // #191 lost this path's measurements and the report lost its Copy row.
  CHECK(Fail, check_ingest_timing(&stage, marker, 1, 1, 0));

  h_out = malloc(src_bytes);
  CHECK(Fail, h_out);
  CU(Fail, cuMemcpyDtoH(h_out, d_linear, src_bytes));

  CHECK(Fail, memcmp(h_out, h_src, src_bytes) == 0);

  ok = 1;

Fail:
  cu_stream_sync(compute);
  cu_stream_sync(h2d);
  free(h_src);
  free(h_out);
  ingest_destroy(&stage);
  gpu_ordering_destroy(&ord);
  cu_event_destroy(marker);
  cu_mem_free(d_linear);
  cu_stream_destroy(h2d);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Overrunning the ring overwrites the oldest measurements, and only the
// lost-sample counter records it. Other tests assert it is zero; this drives
// it.
static int
test_ingest_timing_ring_wraps(void)
{
  log_info("=== test_ingest_timing_ring_wraps ===");

  const size_t bytes_per_element = 2;
  const uint64_t epoch_elements = 48;
  const size_t src_bytes = epoch_elements * bytes_per_element;
  const int extra = 2;
  const int dispatches = SCATTER_TIMING_SLOTS + extra;

  struct staging_state stage = { 0 };
  struct gpu_ordering ord = { 0 };
  CUstream h2d = 0, compute = 0;
  CUevent before_all = 0, before_reuse = 0;
  CUdeviceptr d_linear = 0;
  int ok = 0;

  CU(Fail, cuStreamCreate(&h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, gpu_ordering_init(&ord, compute) == 0);
  gpu_ordering_register_stream(&ord, GPU_STREAM_H2D, h2d);
  gpu_ordering_register_stream(&ord, GPU_STREAM_COMPUTE, compute);
  CHECK(Fail, ingest_init(&stage, src_bytes, &ord, compute) == 0);
  CHECK(Fail, record_marker(&before_all, compute, h2d) == 0);

  CU(Fail, cuMemAlloc(&d_linear, src_bytes));

  for (int i = 0; i < dispatches; ++i) {
    struct gpu_pool_view h_in;
    CHECK(Fail,
          gpu_pool_host_acquire_produce(&stage.h_pool, stage.current, &h_in) ==
            0);
    // The last two dispatches leave each staging slot's start event behind, so
    // a marker here proves both were re-recorded. No other test reuses a slot.
    if (i == dispatches - 2)
      CHECK(Fail, record_marker(&before_reuse, compute, h2d) == 0);
    memset(h_in.p, i + 1, src_bytes);
    stage.bytes_written = src_bytes;
    CHECK(
      Fail,
      ingest_dispatch_multiscale(
        &stage, d_linear, epoch_elements, 0, bytes_per_element, h2d, compute) ==
        0);
  }

  CU(Fail, cuStreamSynchronize(compute));
  CU(Fail, cuStreamSynchronize(h2d));

  // Before the collect, which clears the flags this reads.
  for (int i = 0; i < (int)countof(stage.slot); ++i)
    CHECK(Fail,
          event_is_after(
            before_reuse, stage.slot[i].t_h2d_start, "reused H2D slot", i));

  // Each side keeps one entry per slot, so the surviving counts are the
  // capacities rather than the dispatch count.
  CHECK(Fail,
        check_ingest_timing(&stage,
                            before_all,
                            (int)countof(stage.slot),
                            SCATTER_TIMING_SLOTS,
                            (uint64_t)extra));

  ok = 1;

Fail:
  cu_stream_sync(compute);
  cu_stream_sync(h2d);
  ingest_destroy(&stage);
  gpu_ordering_destroy(&ord);
  cu_event_destroy(before_all);
  cu_event_destroy(before_reuse);
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
              { "ingest_multiscale", test_ingest_multiscale },
              { "ingest_timing_ring_wraps", test_ingest_timing_ring_wraps }, )
