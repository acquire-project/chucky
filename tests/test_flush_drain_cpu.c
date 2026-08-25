// Regression test: writer_close() on the CPU stream must drain sink IO
// before returning. finalize_shards queues footer write_direct jobs that
// reference stream-owned footer_buf memory; without the drain, stream
// destroy frees that memory while the IO worker may still read it.

#include "platform/platform.h"
#include "stream.cpu.h"
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
  if (!r.error)
    r = writer_close(fa->w);
  fa->error = r.error;
  atomic_store(&fa->done, 1);
}

static int
test_flush_waits_for_sink_io(const char* tmpdir)
{
  log_info("=== test_flush_waits_for_sink_io (cpu) ===");

  // Unbounded append dim with chunks_per_shard=2 and one appended epoch:
  // the shard is partial at flush time, so writer_flush's finalize_shards
  // queues the footer write_direct behind the injected gate.
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

  struct io_faults faults = { 0 };
  struct test_zarr_sink z = { 0 };
  struct tile_stream_cpu* s = NULL;
  uint16_t* src = NULL;
  test_thread* thr = NULL;
  int rc = 1;
  _Atomic int gate;
  atomic_store(&gate, 0);

  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          io_faults_store_create(&faults, tmpdir, /*unbuffered=*/1),
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
  };
  s = tile_stream_cpu_create(&cfg, test_zarr_sink_as_shard_sink(&z));
  CHECK(Cleanup, s);

  src = (uint16_t*)calloc(epoch_elements, sizeof(uint16_t));
  CHECK(Cleanup, src);

  {
    struct slice input = { .beg = src, .end = src + epoch_elements };
    struct writer_result r = writer_append(tile_stream_cpu_writer(s), input);
    CHECK(Cleanup, r.error == 0);
  }

  // Gate sits in the io_queue ahead of the footer jobs the flush will queue
  // behind it.
  CHECK(Cleanup, io_faults_inject_blocking_job(&faults, &gate) == 0);

  struct flush_args fa = { .w = tile_stream_cpu_writer(s) };
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
    log_error("close returned before sink IO drained — fix is not in place");
    goto Cleanup;
  }

  if (fa.error) {
    log_error("flush returned error %d", fa.error);
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
  test_thread_join(thr);
  tile_stream_cpu_destroy(s);
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

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/flush_drain_cpu", tmpdir);
    test_mkdir(sub);
    ecode |= test_flush_waits_for_sink_io(sub);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
