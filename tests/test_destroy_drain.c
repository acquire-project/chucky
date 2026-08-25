// Regression test: tile_stream_gpu_destroy must drain sink IO before
// freeing pinned aggregate buffers it references.

#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "store.h"
#include "stream.gpu.h"
#include "test_io_faults.h"
#include "test_platform.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#define DRAIN_OBSERVE_MS 200
#define POST_RELEASE_TIMEOUT_MS 5000

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
test_destroy_waits_for_sink_io(const char* tmpdir)
{
  log_info("=== test_destroy_waits_for_sink_io ===");

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

  struct io_faults faults;
  struct test_zarr_sink z = { 0 };
  struct tile_stream_gpu* s = NULL;
  uint32_t* src = NULL;
  test_thread* thr = NULL;
  int rc = 1;
  _Atomic int gate;
  atomic_store(&gate, 0);

  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          io_faults_store_create(&faults, tmpdir, 1),
          "0",
          dims,
          3,
          dtype_u32,
          (struct codec_config){ .id = CODEC_NONE }) == 0);

  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };
  s = tile_stream_gpu_create(&cfg, zarr_array_as_shard_sink(z.array));
  CHECK(Cleanup, s);

  src = (uint32_t*)calloc(total_elements, sizeof(uint32_t));
  CHECK(Cleanup, src);

  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  // Gate sits at seq=1 in the io_queue ahead of any pwrite jobs the
  // destroy auto-flush will queue behind it.
  CHECK(Cleanup, io_faults_inject_blocking_job(&faults, &gate) == 0);

  struct destroy_args da = { .s = s };
  atomic_store(&da.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, destroy_thread_fn, &da) == 0);
  s = NULL;

  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  int destroy_returned_early = (atomic_load(&da.done) != 0);

  atomic_store(&gate, 1);

  if (test_wait_flag(&da.done, POST_RELEASE_TIMEOUT_MS)) {
    log_error("destroy did not finish within %d ms after gate release",
              POST_RELEASE_TIMEOUT_MS);
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  if (destroy_returned_early) {
    log_error("destroy returned before sink IO drained — fix is not in place");
    goto Cleanup;
  }

  if (zarr_array_has_error(z.array)) {
    log_error("zarr_array reported IO error after drain");
    goto Cleanup;
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  atomic_store(&gate, 1);
  free(src);
  tile_stream_gpu_destroy(s);
  test_thread_join(thr);
  test_zarr_sink_close(&z);
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
