// Regression test (#147): a flush that errors with footer IO still queued must
// drain the sink before returning, or destroy frees buffers the IO worker still
// reads (use-after-free; ASAN catches it).

#include "platform/platform.h"
#include "store.h"
#include "stream.cpu.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"
#include "zarr/zarr_array.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int
test_destroy_drains_after_flush_error(const char* tmpdir)
{
  log_info("=== test_destroy_drains_after_flush_error (cpu) ===");

  // One epoch into a 2-per-shard append dim leaves a partial shard, so flush
  // queues a footer write then truncates — and the injected hook fails truncate.
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
  struct shard_pool* pool = NULL;
  struct zarr_array* arr = NULL;
  struct tile_stream_cpu* s = NULL;
  uint16_t* src = NULL;
  int rc = 1;

  store = store_fs_create(tmpdir, /*unbuffered=*/1);
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

  // Fail the next truncate so flush errors with a footer write still queued.
  CHECK(Cleanup, shard_pool_fs_inject_failing_truncate(pool) == 0);

  {
    struct writer_result r = writer_flush(tile_stream_cpu_writer(s));
    if (!r.error) {
      log_error("flush unexpectedly succeeded — fault hook not in effect");
      goto Cleanup;
    }
  }

  // Destroy frees the footer buffers; the errored flush must have drained the
  // queued IO, or this reads freed heap under ASAN.
  tile_stream_cpu_destroy(s);
  s = NULL;

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
    ecode |= test_destroy_drains_after_flush_error(sub);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
