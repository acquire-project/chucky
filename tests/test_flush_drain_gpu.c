// Regression test: writer_flush() on the GPU stream must drain sink IO
// before returning, so flush is a commit point: when it returns, all
// queued pwrite_ref jobs are durable.

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

#include <cuda.h>

#define DRAIN_OBSERVE_MS 200
#define POST_RELEASE_TIMEOUT_MS 5000
#define POLL_STEP_MS 10

struct flush_args
{
  struct writer* w;
  _Atomic int done;
  int error;
};

static void
flush_thread_fn(void* arg)
{
  struct flush_args* fa = (struct flush_args*)arg;
  struct writer_result r = writer_flush(fa->w);
  fa->error = r.error;
  atomic_store(&fa->done, 1);
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
test_flush_waits_for_sink_io(const char* tmpdir)
{
  log_info("=== test_flush_waits_for_sink_io ===");

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

  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  // Gate sits at seq=1 in the io_queue ahead of any pwrite jobs the
  // flush will queue behind it.
  CHECK(Cleanup, shard_pool_fs_inject_blocking_job(pool, &gate) == 0);

  struct flush_args fa = { .w = tile_stream_gpu_writer(s) };
  atomic_store(&fa.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, flush_thread_fn, &fa) == 0);

  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  int flush_returned_early = (atomic_load(&fa.done) != 0);

  atomic_store(&gate, 1);

  if (wait_for_done(&fa.done, POST_RELEASE_TIMEOUT_MS)) {
    log_error("flush did not finish within %d ms after gate release",
              POST_RELEASE_TIMEOUT_MS);
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  if (flush_returned_early) {
    log_error("flush returned before sink IO drained — fix is not in place");
    goto Cleanup;
  }

  if (fa.error) {
    log_error("flush returned error %d", fa.error);
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
    snprintf(sub, sizeof(sub), "%s/flush_drain", tmpdir);
    test_mkdir(sub);
    ecode |= test_flush_waits_for_sink_io(sub);
  }

Cleanup:
  if (ctx)
    cuCtxDestroy(ctx);
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
