// Regression test: multiarray GPU flush across two FS-pool-backed zarr arrays
// must not deadlock. The shared d2h_deliver stage stores a per-(level, fc)
// io_event fence in `agg[fc].io_done`. When array A's prior round delivers,
// `deliver_to_shards_batch` stamps that slot with sink A's fence; if array B
// then kicks D2H on the same fc without quiescing the engine first,
// `wait_io_fences` would pass A's fence to B's pool (different counter) and
// hang forever. unbind_context now drains the d2h pipeline for the departing
// array, so the test must complete promptly.
//
// Without the fix this hangs forever; with the fix it returns within
// FLUSH_TIMEOUT_MS.
//
// We use real FS pools (one per array) so the sink's record_fence/wait_fence
// are non-NULL — that is what arms wait_io_fences. The in-process
// test_shard_sink leaves them NULL and short-circuits the bug.

#include "gpu/prelude.cuda.h"
#include "multiarray.gpu.h"
#include "platform/platform.h"
#include "store.h"
#include "stream.gpu.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr.h"
#include "zarr/zarr_array.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#define FLUSH_TIMEOUT_MS 10000
#define POLL_STEP_MS 10

struct flush_args
{
  struct multiarray_writer* w;
  _Atomic int done;
  int error;
};

static void
flush_thread_fn(void* arg)
{
  struct flush_args* fa = (struct flush_args*)arg;
  struct multiarray_writer_result r = fa->w->flush(fa->w);
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

// Build a 3D u32 zarr array under {tmpdir}/{name} with its own FS pool.
// Returns 0 on success, populates *out_store, *out_pool, *out_arr.
static int
make_array(const char* tmpdir,
           const char* name,
           const struct dimension* dims,
           int rank,
           struct store** out_store,
           struct shard_pool** out_pool,
           struct zarr_array** out_arr)
{
  char root[4096];
  snprintf(root, sizeof(root), "%s/%s", tmpdir, name);
  test_mkdir(root);

  struct store* store = store_fs_create(root, 0);
  if (!store)
    return 1;
  if (store->mkdirs(store, ".")) {
    store_destroy(store);
    return 1;
  }
  if (store->mkdirs(store, "0")) {
    store_destroy(store);
    return 1;
  }
  struct shard_pool* pool = store->create_pool(store, 8);
  if (!pool) {
    store_destroy(store);
    return 1;
  }
  struct zarr_array_config acfg = {
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = (uint8_t)rank,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };
  struct zarr_array* arr =
    zarr_array_create_with_pool(store, pool, 0, "0", &acfg);
  if (!arr) {
    shard_pool_destroy(pool);
    store_destroy(store);
    return 1;
  }
  *out_store = store;
  *out_pool = pool;
  *out_arr = arr;
  return 0;
}

static int
test_multiarray_flush_fs(const char* tmpdir)
{
  log_info("=== test_multiarray_flush_fs ===");

  // 3D 4x4x6, chunk 2x2x3, cps 2x1x1 → epoch_elements=48, 2 epochs total.
  struct dimension dimsA[3] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .name = "z", .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .name = "y", .storage_position = 1 },
    { .size = 6, .chunk_size = 3, .chunks_per_shard = 1,
      .name = "x", .storage_position = 2 },
  };
  struct dimension dimsB[3] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .name = "z", .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .name = "y", .storage_position = 1 },
    { .size = 6, .chunk_size = 3, .chunks_per_shard = 1,
      .name = "x", .storage_position = 2 },
  };
  const size_t total_elements = 4 * 4 * 6;

  struct store* storeA = NULL;
  struct store* storeB = NULL;
  struct shard_pool* poolA = NULL;
  struct shard_pool* poolB = NULL;
  struct zarr_array* arrA = NULL;
  struct zarr_array* arrB = NULL;
  struct multiarray_tile_stream_gpu* ms = NULL;
  uint32_t* src = NULL;
  test_thread* thr = NULL;
  int rc = 1;

  CHECK(Cleanup,
        make_array(tmpdir, "A", dimsA, 3, &storeA, &poolA, &arrA) == 0);
  CHECK(Cleanup,
        make_array(tmpdir, "B", dimsB, 3, &storeB, &poolB, &arrB) == 0);

  struct tile_stream_configuration configA = {
    .buffer_capacity_bytes = total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dimsA,
    .codec = { .id = CODEC_NONE },
  };
  struct tile_stream_configuration configB = configA;
  configB.dimensions = dimsB;

  struct tile_stream_configuration configs[] = { configA, configB };
  struct shard_sink* sinks[] = { zarr_array_as_shard_sink(arrA),
                                 zarr_array_as_shard_sink(arrB) };
  ms = multiarray_tile_stream_gpu_create(2, configs, sinks, 0);
  CHECK(Cleanup, ms);

  struct multiarray_writer* w = multiarray_tile_stream_gpu_writer(ms);

  src = (uint32_t*)calloc(total_elements, sizeof(uint32_t));
  CHECK(Cleanup, src);
  for (size_t i = 0; i < total_elements; ++i)
    src[i] = (uint32_t)i;

  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct multiarray_writer_result r = w->update(w, 0, input);
    CHECK(Cleanup, r.error == multiarray_writer_ok);
  }
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct multiarray_writer_result r = w->update(w, 1, input);
    CHECK(Cleanup, r.error == multiarray_writer_ok);
  }

  struct flush_args fa = { .w = w, .error = 0 };
  atomic_store(&fa.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, flush_thread_fn, &fa) == 0);

  if (wait_for_done(&fa.done, FLUSH_TIMEOUT_MS)) {
    log_error("multiarray flush did not complete within %d ms — hang regression",
              FLUSH_TIMEOUT_MS);
    // Don't join — would hang. Leak the worker.
    thr = NULL;
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  CHECK(Cleanup, fa.error == multiarray_writer_ok);

  rc = 0;
  log_info("  PASS");

Cleanup:
  free(src);
  multiarray_tile_stream_gpu_destroy(ms);
  test_thread_join(thr);
  zarr_array_destroy(arrA);
  zarr_array_destroy(arrB);
  shard_pool_destroy(poolA);
  shard_pool_destroy(poolB);
  store_destroy(storeA);
  store_destroy(storeB);
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

  ecode |= test_multiarray_flush_fs(tmpdir);

Cleanup:
  if (ctx)
    cuCtxDestroy(ctx);
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
