// Destroying a stream with a kicked, undelivered batch must drain the
// delivery worker and the sink before freeing anything the queued work
// references — including when sink IO is stalled at that moment.

#include "gpu/prelude.cuda.h"
#include "gpu/stream.internal.h" // white-box: hold the delivery worker
#include "platform/platform.h"
#include "store.h"
#include "stream.gpu.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"
#include "zarr/zarr_array.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#define DRAIN_OBSERVE_MS 200
#define POST_RELEASE_TIMEOUT_MS 10000
#define POLL_STEP_MS 10

struct destroy_args
{
  struct tile_stream_gpu* s;
  _Atomic int done;
};

static void
destroy_thread_fn(void* arg)
{
  struct destroy_args* da = (struct destroy_args*)arg;
  tile_stream_gpu_destroy(da->s);
  atomic_store(&da->done, 1);
}

static int
wait_for_done(_Atomic int* done, int timeout_ms)
{
  int waited_ms = 0;
  while (atomic_load(done) == 0 && waited_ms < timeout_ms) {
    platform_sleep_ns((int64_t)POLL_STEP_MS * 1000000LL);
    waited_ms += POLL_STEP_MS;
  }
  return atomic_load(done) != 0 ? 0 : -1;
}

// Append 1.5 batches (slot kicked, delivery in flight, partial batch
// accumulated) and destroy immediately, without flushing.
static int
test_destroy_with_delivery_in_flight(const char* tmpdir, int rep)
{
  log_info("=== test_destroy_with_delivery_in_flight rep %d ===", rep);

  struct dimension dims[3] = {
    { .size = 12,
      .chunk_size = 4,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 16,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 64,
      .chunk_size = 16,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };
  const size_t epoch_elements = 4 * 64 * 64;
  const size_t append_elements = 3 * epoch_elements; // 1.5 batches at K=2

  struct store* store = NULL;
  struct shard_pool* pool = NULL;
  struct zarr_array* arr = NULL;
  struct tile_stream_gpu* s = NULL;
  uint16_t* src = NULL;
  int rc = 1;

  store = store_fs_create(tmpdir, 1);
  CHECK(Cleanup, store);
  CHECK(Cleanup, store->mkdirs(store, ".") == 0);
  CHECK(Cleanup, store->mkdirs(store, "0") == 0);

  pool = store->create_pool(store, 8);
  CHECK(Cleanup, pool);

  struct zarr_array_config acfg = {
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
  };
  arr = zarr_array_create_with_pool(store, pool, 0, "0", &acfg);
  CHECK(Cleanup, arr);

  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = epoch_elements * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .epochs_per_batch = 2,
  };
  s = tile_stream_gpu_create(&cfg, zarr_array_as_shard_sink(arr));
  CHECK(Cleanup, s);

  src = (uint16_t*)malloc(append_elements * sizeof(uint16_t));
  CHECK(Cleanup, src);
  for (size_t i = 0; i < append_elements; ++i)
    src[i] = (uint16_t)(i * 31 + (size_t)rep);

  {
    struct slice input = { .beg = src, .end = src + append_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  tile_stream_gpu_destroy(s);
  s = NULL;

  CHECK(Cleanup, zarr_array_has_error(arr) == 0);

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  tile_stream_gpu_destroy(s);
  zarr_array_destroy(arr);
  shard_pool_destroy(pool);
  store_destroy(store);
  return rc;
}

// Two pipelined batches kicked (both enqueued on the delivery worker), the
// sink forced into an error state, and the worker held after each job so the
// second batch's delivery stays queued. destroy's auto-flush aborts on the
// first failed drain without joining the second job; stop_join then has to run
// that still-queued job out — the teardown run-out path. The hold makes the
// queued-at-stop_join state deterministic rather than timing-dependent.
// Asserts no hang and (under ASAN) no use-after-free.
static int
test_destroy_runs_out_queued_delivery(const char* tmpdir)
{
  log_info("=== test_destroy_runs_out_queued_delivery ===");

  struct dimension dims[3] = {
    { .size = 12,
      .chunk_size = 4,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 16,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 64,
      .chunk_size = 16,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };
  const size_t epoch_elements = 4 * 64 * 64;
  const size_t append_elements = 4 * epoch_elements; // two full batches at K=2

  struct store* store = NULL;
  struct shard_pool* pool = NULL;
  struct zarr_array* arr = NULL;
  struct tile_stream_gpu* s = NULL;
  uint16_t* src = NULL;
  test_thread* thr = NULL;
  int rc = 1;

  store = store_fs_create(tmpdir, 1);
  CHECK(Cleanup, store);
  CHECK(Cleanup, store->mkdirs(store, ".") == 0);
  CHECK(Cleanup, store->mkdirs(store, "0") == 0);

  pool = store->create_pool(store, 8);
  CHECK(Cleanup, pool);

  struct zarr_array_config acfg = {
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
  };
  arr = zarr_array_create_with_pool(store, pool, 0, "0", &acfg);
  CHECK(Cleanup, arr);

  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = epoch_elements * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .epochs_per_batch = 2,
  };
  s = tile_stream_gpu_create(&cfg, zarr_array_as_shard_sink(arr));
  CHECK(Cleanup, s);

  src = (uint16_t*)malloc(append_elements * sizeof(uint16_t));
  CHECK(Cleanup, src);
  for (size_t i = 0; i < append_elements; ++i)
    src[i] = (uint16_t)(i * 31);

  // Hold the worker after each completed job so the second batch's delivery is
  // still queued when stop_join runs, exercising the run-out line. No-op if the
  // worker is absent (drains run inline; no run-out line to hit).
  gpu_delivery_set_hold(&s->engine.delivery, 1);

  // Force the sink errored before the kicks so every delivery the worker
  // attempts short-circuits on has_error and fails fast.
  shard_pool_fs_set_error(pool);

  {
    struct slice input = { .beg = src, .end = src + append_elements };
    // Append succeeds (the kick path does not check has_error); both batches
    // are kicked and enqueued on the delivery worker, each failing on the
    // sink error.
    writer_append(tile_stream_gpu_writer(s), input);
  }

  // Destroy on another thread so a hung run-out shows up as a timeout rather
  // than wedging the test.
  struct destroy_args da = { .s = s };
  atomic_store(&da.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, destroy_thread_fn, &da) == 0);
  s = NULL;

  if (wait_for_done(&da.done, POST_RELEASE_TIMEOUT_MS)) {
    log_error("destroy hung with a delivery job still queued");
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  // The sink is errored by construction; the point of the test is the clean
  // teardown, which ASAN validates.
  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  tile_stream_gpu_destroy(s);
  test_thread_join(thr);
  zarr_array_destroy(arr);
  shard_pool_destroy(pool);
  store_destroy(store);
  return rc;
}

// One full batch appended (slot kicked; its delivery ran on the worker and
// queued sink IO), sink IO gated: destroy must block until the IO drains,
// then finish clean.
static int
test_destroy_blocks_on_gated_sink(const char* tmpdir)
{
  log_info("=== test_destroy_blocks_on_gated_sink ===");

  struct dimension dims[3] = {
    { .size = 16,
      .chunk_size = 4,
      .chunks_per_shard = 4,
      .name = "z",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 12,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };
  const size_t epoch_elements = 4 * 8 * 12;
  const size_t append_elements = 2 * epoch_elements; // one full batch at K=2

  struct store* store = NULL;
  struct shard_pool* pool = NULL;
  struct zarr_array* arr = NULL;
  struct tile_stream_gpu* s = NULL;
  uint32_t* src = NULL;
  test_thread* thr = NULL;
  int rc = 1;
  _Atomic int gate;
  atomic_store(&gate, 0);

  store = store_fs_create(tmpdir, 0);
  CHECK(Cleanup, store);
  CHECK(Cleanup, store->mkdirs(store, ".") == 0);
  CHECK(Cleanup, store->mkdirs(store, "0") == 0);

  pool = store->create_pool(store, 8);
  CHECK(Cleanup, pool);

  struct zarr_array_config acfg = {
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };
  arr = zarr_array_create_with_pool(store, pool, 0, "0", &acfg);
  CHECK(Cleanup, arr);

  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = epoch_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 2,
  };
  s = tile_stream_gpu_create(&cfg, zarr_array_as_shard_sink(arr));
  CHECK(Cleanup, s);

  src = (uint32_t*)calloc(append_elements, sizeof(uint32_t));
  CHECK(Cleanup, src);

  {
    struct slice input = { .beg = src, .end = src + append_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  CHECK(Cleanup, shard_pool_fs_inject_blocking_job(pool, &gate) == 0);

  struct destroy_args da = { .s = s };
  atomic_store(&da.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, destroy_thread_fn, &da) == 0);
  s = NULL;

  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  int destroy_returned_early = (atomic_load(&da.done) != 0);

  atomic_store(&gate, 1);

  if (wait_for_done(&da.done, POST_RELEASE_TIMEOUT_MS)) {
    log_error("destroy did not finish within %d ms after gate release",
              POST_RELEASE_TIMEOUT_MS);
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  if (destroy_returned_early) {
    log_error("destroy returned before sink IO drained");
    goto Cleanup;
  }

  CHECK(Cleanup, zarr_array_has_error(arr) == 0);

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  tile_stream_gpu_destroy(s);
  test_thread_join(thr);
  zarr_array_destroy(arr);
  shard_pool_destroy(pool);
  store_destroy(store);
  return rc;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  char tmpdir[4096];
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  log_info("temp dir: %s", tmpdir);

  CUcontext ctx = 0;
  CUdevice dev;
  CU(Cleanup, cuInit(0));
  CU(Cleanup, cuDeviceGet(&dev, 0));
  CU(Cleanup, cu_ctx_create(&ctx, 0, dev));

  for (int rep = 0; rep < 5; ++rep) {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/midstream_%d", tmpdir, rep);
    test_mkdir(sub);
    ecode |= test_destroy_with_delivery_in_flight(sub, rep);
  }

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/runout", tmpdir);
    test_mkdir(sub);
    ecode |= test_destroy_runs_out_queued_delivery(sub);
  }

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/gated", tmpdir);
    test_mkdir(sub);
    ecode |= test_destroy_blocks_on_gated_sink(sub);
  }

Cleanup:
  if (ctx)
    cuCtxDestroy(ctx);
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
