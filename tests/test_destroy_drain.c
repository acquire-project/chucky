// Regression test for issue #110.
//
// On GPU stream teardown, h_aggregated (pinned host memory referenced by
// async pwrite_ref jobs) used to be freed before the sink's IO queue had
// drained. tile_stream_gpu_destroy now calls shard_sink_drain to record
// and wait on a terminal fence per level before freeing the aggregate
// slots.
//
// We verify the drain is actually present by injecting a job that blocks
// on an atomic gate. With the drain in place, destroy runs on a worker
// thread and blocks until the test releases the gate. Without the drain,
// destroy returns immediately and the assertion that the worker is still
// alive fails.

#include "gpu/prelude.cuda.h"
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

// Timing budgets. We know the sink IO is a single CPU-bound spin and a
// handful of pwrite jobs against ~512 KiB of compressed data, so we can
// afford tight bounds: the test does not depend on disk speed because
// the gate job blocks the io_queue worker until we release it.
#define DRAIN_OBSERVE_MS 200         // wait this long while gate is closed
#define POST_RELEASE_TIMEOUT_MS 5000 // generous bound after gate releases
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

static int
test_destroy_waits_for_sink_io(const char* tmpdir)
{
  log_info("=== test_destroy_waits_for_sink_io ===");

  // Mirror tests/test_zarr_fs_sink.c::test_pipeline geometry: 12x8x12
  // elements, 2x4x3 chunks, (3,2,2) chunks/shard => 4 shards. This config
  // is known-good for the compress pipeline, so writes actually queue
  // pwrite_ref jobs behind the blocking gate.
  struct dimension dims[3] = {
    { .size = 12,
      .chunk_size = 2,
      .chunks_per_shard = 3,
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
  const size_t total_elements = 12 * 8 * 12;

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

  // 8 shards => 8 slots so writes don't serialize through one slot.
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
    .buffer_capacity_bytes = total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };
  s = tile_stream_gpu_create(&cfg, zarr_array_as_shard_sink(arr));
  CHECK(Cleanup, s);

  src = (uint32_t*)calloc(total_elements, sizeof(uint32_t));
  CHECK(Cleanup, src);
  for (size_t i = 0; i < total_elements; ++i)
    src[i] = (uint32_t)i;

  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  // Inject the blocking job before destroy. It sits at seq=1 in the
  // io_queue. The destroy auto-flush will queue pwrite jobs behind it,
  // and shard_sink_drain will record a terminal fence behind everything.
  // wait_fence then blocks until the gate is released.
  CHECK(Cleanup, shard_pool_fs_inject_blocking_job(pool, &gate) == 0);

  struct destroy_args da = { .s = s };
  atomic_store(&da.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, destroy_thread_fn, &da) == 0);

  // Hand ownership of the stream to the worker thread.
  s = NULL;

  // Phase 1: while the gate is closed, destroy MUST be blocked in
  // shard_sink_drain. If `done` is set within DRAIN_OBSERVE_MS, the
  // drain isn't waiting — fix is missing.
  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  int destroy_returned_early = (atomic_load(&da.done) != 0);

  // Release the gate. The blocking job exits, queued pwrites drain, the
  // terminal fence retires, destroy unblocks. Always release so the
  // failure path can still tear down the pool cleanly.
  atomic_store(&gate, 1);

  // Phase 2: bounded wait for destroy to finish. With the fix this is
  // ~milliseconds; we allow POST_RELEASE_TIMEOUT_MS as a slop budget for
  // CI noise. If we exceed it, something is genuinely wrong — fail loud
  // and leak the worker rather than hanging the test.
  if (wait_for_done(&da.done, POST_RELEASE_TIMEOUT_MS)) {
    log_error("destroy did not finish within %d ms after gate release",
              POST_RELEASE_TIMEOUT_MS);
    // Intentionally do NOT join — would hang. Leak the test_thread.
    thr = NULL;
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  if (destroy_returned_early) {
    log_error("destroy returned before sink IO drained — fix is not in place");
    goto Cleanup;
  }

  if (zarr_array_has_error(arr)) {
    log_error("zarr_array reported IO error after drain");
    goto Cleanup;
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  // If the worker is still pending (e.g. early init failure before
  // handoff), we never spawned it; if we leaked it on timeout, thr is
  // already NULL.
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
  CU(Cleanup, cuCtxCreate(&ctx, 0, dev));

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/destroy_drain", tmpdir);
    test_mkdir(sub);
    ecode |= test_destroy_waits_for_sink_io(sub);
  }

Cleanup:
  if (ctx)
    cuCtxDestroy(ctx);
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
