// A flush is required to wait for the work it queued and to report its
// failures.

#include "stream.cpu.h"
#include "test_io_faults.h"
#include "test_platform.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"
#include "writer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Regression (#218): finalizing a partial shard posts a truncate and a close,
// and a flush that returned without waiting reported success for a shard whose
// size on disk is wrong.
static int
test_flush_reports_queued_truncate_failure(const char* tmpdir)
{
  log_info("=== test_flush_reports_queued_truncate_failure (cpu) ===");

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

  struct io_faults faults = { 0 };
  struct test_zarr_sink z = { 0 };
  struct tile_stream_cpu* s = NULL;
  uint16_t* src = NULL;
  int rc = 1;

  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          io_faults_store_create(&faults, tmpdir, /*unbuffered=*/1, NULL),
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

  // The truncate fails on the worker, so only a flush that waits can see it.
  io_faults_fail_next_truncate(&faults);

  {
    struct writer_result r = writer_flush(tile_stream_cpu_writer(s));
    if (!r.error) {
      log_error("flush missed the failure of the truncate it queued");
      goto Cleanup;
    }
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  tile_stream_cpu_destroy(s);
  test_zarr_sink_close(&z);
  return rc;
}

// Regression (#240): at a call site that flushes and then reads the store
// back, the reported error is the only signal, because the shard the site
// checks is already whole on disk.
static int
test_sink_flush_reports_io_failure(const char* tmpdir)
{
  log_info("=== test_sink_flush_reports_io_failure (cpu) ===");

  // One epoch fills the whole shard, so the file a call site checks is
  // complete before the fault is armed.
  struct dimension dims[3] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
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
  int rc = 1;

  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          io_faults_store_create(&faults, tmpdir, /*unbuffered=*/0, NULL),
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
  {
    struct writer_result r = writer_flush(tile_stream_cpu_writer(s));
    CHECK(Cleanup, r.error == 0);
  }

  // A clean flush is checked first, so the failure below can only come from
  // the fault.
  CHECK(Cleanup, test_zarr_sink_flush(&z) == 0);
  CHECK(Cleanup, io_faults_inject_failing_job(&faults) == 0);
  CHECK(Cleanup, test_zarr_sink_flush(&z) != 0);

  {
    char path[4300];
    snprintf(path, sizeof(path), "%s/0/c/0/0/0", tmpdir);
    CHECK(Cleanup, test_file_exists(path));
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  tile_stream_cpu_destroy(s);
  test_zarr_sink_close(&z);
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
  ecode = 0;
  log_info("temp dir: %s", tmpdir);

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/flush_error_drain_cpu", tmpdir);
    test_mkdir(sub);
    ecode |= test_flush_reports_queued_truncate_failure(sub);
  }

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/sink_flush_io_failure", tmpdir);
    test_mkdir(sub);
    ecode |= test_sink_flush_reports_io_failure(sub);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
