// Regression test (#218): a flush has to report the failures of the work it
// queued. Finalizing a partial shard posts a truncate and a close, and a flush
// that returned without waiting reported success for a shard whose size on disk
// is wrong.

#include "platform/platform.h"
#include "store.h"
#include "stream.cpu.h"
#include "test_io_faults.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_pool.h"
#include "zarr/zarr_array.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

  struct store* store = NULL;
  struct io_faults faults;
  struct shard_pool* pool = NULL;
  struct zarr_array* arr = NULL;
  struct tile_stream_cpu* s = NULL;
  uint16_t* src = NULL;
  int rc = 1;

  store = io_faults_store_create(&faults, tmpdir, /*unbuffered=*/1);
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
    .codec = { .id = CODEC_NONE },
  };
  arr = zarr_array_create_with_pool(store, pool, 0, "0", &acfg);
  CHECK(Cleanup, arr);

  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = epoch_elements * sizeof(uint16_t),
    .epochs_per_batch = 1,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };
  s = tile_stream_cpu_create(&cfg, zarr_array_as_shard_sink(arr));
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

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/flush_error_drain_cpu", tmpdir);
    test_mkdir(sub);
    ecode |= test_flush_reports_queued_truncate_failure(sub);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
