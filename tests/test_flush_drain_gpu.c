// Regression test: a flush on the GPU stream drains sink IO before returning,
// so when it returns every queued write is durable and any that failed is
// reported (#218).

#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "store.h"
#include "stream.gpu.h"
#include "test_io_faults.h"
#include "test_platform.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"
#include "writer.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#define DRAIN_OBSERVE_MS 200
#define POST_RELEASE_TIMEOUT_MS 5000

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
          io_faults_store_create(&faults, tmpdir, 1, NULL),
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
  s = tile_stream_gpu_create(&cfg, test_zarr_sink_as_shard_sink(&z));
  CHECK(Cleanup, s);

  src = (uint32_t*)calloc(total_elements, sizeof(uint32_t));
  CHECK(Cleanup, src);

  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  // Gate sits at seq=1 in the scheduler ahead of any pwrite jobs the
  // flush will queue behind it.
  CHECK(Cleanup, io_faults_inject_blocking_job(&faults, &gate) == 0);

  struct flush_args fa = { .w = tile_stream_gpu_writer(s) };
  atomic_store(&fa.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, flush_thread_fn, &fa) == 0);

  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  int flush_returned_early = (atomic_load(&fa.done) != 0);

  atomic_store(&gate, 1);

  if (test_wait_flag(&fa.done, POST_RELEASE_TIMEOUT_MS)) {
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

  if (test_zarr_sink_has_error(&z)) {
    log_error("zarr_array reported IO error after drain");
    goto Cleanup;
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  atomic_store(&gate, 1);
  free(src);
  test_thread_join(thr);
  tile_stream_gpu_destroy(s);
  test_zarr_sink_close(&z);
  return rc;
}

static int
test_flush_reports_queued_truncate_failure(const char* tmpdir)
{
  log_info("=== test_flush_reports_queued_truncate_failure ===");

  // One epoch into a 2-per-shard append dim leaves a partial shard, so the
  // flush has a truncate to queue.
  struct dimension dims[3] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 1 },
    { .size = 8,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 2 },
  };
  const size_t epoch_elements = 8 * 8;

  struct io_faults faults;
  struct test_zarr_sink z = { 0 };
  struct tile_stream_gpu* s = NULL;
  uint16_t* src = NULL;
  int rc = 1;

  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          io_faults_store_create(&faults, tmpdir, 1, NULL),
          "0",
          dims,
          3,
          dtype_u16,
          (struct codec_config){ .id = CODEC_NONE }) == 0);

  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = epoch_elements * sizeof(uint16_t),
    .epochs_per_batch = 1,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .metadata_update_interval_s = 3600.0f,
  };
  s = tile_stream_gpu_create(&cfg, test_zarr_sink_as_shard_sink(&z));
  CHECK(Cleanup, s);

  src = (uint16_t*)calloc(epoch_elements, sizeof(uint16_t));
  CHECK(Cleanup, src);

  {
    struct slice input = { .beg = src, .end = src + epoch_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  // The truncate fails on the worker, so only a flush that waits can see it.
  io_faults_fail_next_truncate(&faults);

  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    if (!r.error) {
      log_error("flush missed the failure of the truncate it queued");
      goto Cleanup;
    }
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  tile_stream_gpu_destroy(s);
  test_zarr_sink_close(&z);
  return rc;
}

struct foreign_args
{
  struct writer* w;
  struct slice input;
  CUcontext own;    // the context this thread holds coming in
  int error;        // non-zero if the append or flush reported one
  int context_kept; // 1 if `own` was still current on return
  int consumed_all; // 1 if the append took the whole slice
};

// The writer runs on whatever thread the caller hands it, and a thread
// holding a context of its own must come back holding it.
static void
foreign_context_thread_fn(void* arg)
{
  struct foreign_args* fa = (struct foreign_args*)arg;
  if (cuCtxSetCurrent(fa->own) != CUDA_SUCCESS) {
    fa->error = 1;
    return;
  }

  struct writer_result r = writer_append(fa->w, fa->input);
  // Reporting success while consuming nothing is the shape a wrong-context
  // run would take, so the whole slice has to be gone.
  fa->consumed_all = r.error == 0 && r.rest.beg == fa->input.end;
  if (r.error == 0)
    r = writer_flush(fa->w);
  fa->error = r.error;

  CUcontext after = NULL;
  fa->context_kept =
    cuCtxGetCurrent(&after) == CUDA_SUCCESS && after == fa->own;
}

static int
test_writes_from_a_foreign_context(const char* tmpdir, CUdevice dev)
{
  log_info("=== test_writes_from_a_foreign_context ===");

  struct dimension dims[3] = {
    { .size = 12, .chunk_size = 2, .chunks_per_shard = 3, .name = "z" },
    { .size = 8, .chunk_size = 4, .chunks_per_shard = 2, .name = "y" },
    { .size = 12, .chunk_size = 3, .chunks_per_shard = 2, .name = "x" },
  };
  for (int d = 0; d < 3; ++d)
    dims[d].storage_position = (uint8_t)d;
  const size_t total_elements = 12 * 8 * 12;

  struct test_zarr_sink z = { 0 };
  struct tile_stream_gpu* s = NULL;
  uint32_t* src = NULL;
  test_thread* thr = NULL;
  CUcontext other = 0;
  int rc = 1;

  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          store_fs_create(tmpdir, 1),
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
  // Created here, so the stream's context is the one this thread holds.
  s = tile_stream_gpu_create(&cfg, test_zarr_sink_as_shard_sink(&z));
  CHECK(Cleanup, s);

  src = (uint32_t*)calloc(total_elements, sizeof(uint32_t));
  CHECK(Cleanup, src);

  CUcontext mine = NULL;
  CU(Cleanup, cuCtxGetCurrent(&mine));
  CU(Cleanup, cu_ctx_create(&other, 0, dev));
  // Creating a context makes it current here too, and only the writer's
  // thread should hold it.
  CU(Cleanup, cuCtxSetCurrent(mine));

  struct foreign_args fa = {
    .w = tile_stream_gpu_writer(s),
    .input = { .beg = src, .end = src + total_elements },
    .own = other,
  };
  CHECK(Cleanup, test_thread_start(&thr, foreign_context_thread_fn, &fa) == 0);
  test_thread_join(thr);
  thr = NULL;

  if (fa.error) {
    log_error("append/flush from a thread holding another context failed");
    goto Cleanup;
  }
  if (!fa.context_kept) {
    log_error("the writer left the calling thread on a different context");
    goto Cleanup;
  }
  if (!fa.consumed_all) {
    log_error("the append reported success without taking the input");
    goto Cleanup;
  }
  if (test_zarr_sink_has_error(&z)) {
    log_error("zarr_array reported an IO error");
    goto Cleanup;
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  test_thread_join(thr);
  tile_stream_gpu_destroy(s);
  test_zarr_sink_close(&z);
  if (other)
    cuCtxDestroy(other);
  return rc;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 1;
  char tmpdir[4096];
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  log_info("temp dir: %s", tmpdir);

  CUcontext ctx = 0;
  CUdevice dev;
  CU(Cleanup, cuInit(0));
  CU(Cleanup, cuDeviceGet(&dev, 0));
  CU(Cleanup, cu_ctx_create(&ctx, 0, dev));
  ecode = 0;

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/flush_drain", tmpdir);
    test_mkdir(sub);
    ecode |= test_flush_waits_for_sink_io(sub);
  }

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/flush_truncate_error", tmpdir);
    test_mkdir(sub);
    ecode |= test_flush_reports_queued_truncate_failure(sub);
  }

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/foreign_context", tmpdir);
    test_mkdir(sub);
    ecode |= test_writes_from_a_foreign_context(sub, dev);
  }

Cleanup:
  if (ctx)
    cuCtxDestroy(ctx);
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
