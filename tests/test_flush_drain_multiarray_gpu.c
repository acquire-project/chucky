// Regression test: shard_sink_drain_many's fan-out walks every sink. An
// indexing or stride bug in the record-all-then-wait-all loop would
// otherwise wait on only one sink and slip past CI. Exercises the helper
// directly with N=2 zarr_array sinks backed by independent FS pools.

#include "platform/platform.h"
#include "store.h"
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

#define DRAIN_OBSERVE_MS 200
#define POST_RELEASE_TIMEOUT_MS 5000
#define POLL_STEP_MS 10

#define N_SINKS 2

struct drain_args
{
  struct shard_sink** sinks;
  int* nlods;
  int n;
  _Atomic int done;
  int errors;
};

static void
drain_thread_fn(void* arg)
{
  struct drain_args* da = (struct drain_args*)arg;
  da->errors = shard_sink_drain_many(da->sinks, da->nlods, da->n);
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
test_drain_many_fans_out(const char* tmpdir)
{
  log_info("=== test_drain_many_fans_out ===");

  // Trivial 2D shape per zarr_array. The arrays don't need to receive any
  // data — we only exercise the sink's record_fence/wait_fence path,
  // which is what shard_sink_drain_many fans out across.
  struct dimension dims_proto[2] = {
    { .size = 2,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 0 },
    { .size = 2,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 1 },
  };

  struct store* stores[N_SINKS] = { 0 };
  struct shard_pool* pools[N_SINKS] = { 0 };
  struct zarr_array* arrs[N_SINKS] = { 0 };
  // gates[i]=blocking sentinel for sink i. priming_gate retires pool 1's
  // seq=1 immediately so its retired_seq starts at 1; gates[1] then sits
  // at seq=2. Pool 0 has only one sentinel at seq=1. The asymmetry lets
  // the test catch a buggy _many that mixes events across sinks (e.g.
  // always passes evs[0] to wait_fence): sink 1's wait on sink 0's seq=1
  // would return early because pool 1 already retired its own seq=1.
  _Atomic int gates[N_SINKS];
  _Atomic int priming_gate;
  atomic_store(&priming_gate, 1); // pre-released
  for (int i = 0; i < N_SINKS; ++i)
    atomic_store(&gates[i], 0);

  test_thread* thr = NULL;
  int rc = 1;

  for (int i = 0; i < N_SINKS; ++i) {
    char sub[4096];
    snprintf(sub, sizeof(sub), "%s/sink%d", tmpdir, i);
    test_mkdir(sub);

    stores[i] = store_fs_create(sub, 0);
    CHECK(Cleanup, stores[i]);
    CHECK(Cleanup, stores[i]->mkdirs(stores[i], ".") == 0);
    CHECK(Cleanup, stores[i]->mkdirs(stores[i], "0") == 0);

    pools[i] = stores[i]->create_pool(stores[i], 8);
    CHECK(Cleanup, pools[i]);

    struct dimension dims[2];
    for (int d = 0; d < 2; ++d)
      dims[d] = dims_proto[d];

    struct zarr_array_config acfg = {
      .data_type = dtype_u32,
      .fill_value = 0,
      .rank = 2,
      .dimensions = dims,
      .codec = { .id = CODEC_NONE },
    };
    arrs[i] = zarr_array_create_with_pool(stores[i], pools[i], 0, "0", &acfg);
    CHECK(Cleanup, arrs[i]);
  }

  struct shard_sink* sinks[N_SINKS];
  int nlods[N_SINKS];
  for (int i = 0; i < N_SINKS; ++i) {
    sinks[i] = zarr_array_as_shard_sink(arrs[i]);
    nlods[i] = 1;
  }

  // Pool 1: priming sentinel (already released) at seq=1, then real
  // sentinel at seq=2. The priming job retires immediately, advancing
  // pool 1's retired_seq to 1 before drain_many records its fence.
  CHECK(Cleanup,
        shard_pool_fs_inject_blocking_job(pools[1], &priming_gate) == 0);

  // Real sentinels: pool 0 seq=1, pool 1 seq=2.  Drain_many will record
  // fences at these seqs, fanning out across both sinks.
  for (int i = 0; i < N_SINKS; ++i)
    CHECK(Cleanup,
          shard_pool_fs_inject_blocking_job(pools[i], &gates[i]) == 0);

  // Wait for pool 1's priming sentinel to retire so its retired_seq
  // advances to 1 deterministically before drain_many records.  Without
  // this, drain_many's record could observe retired_seq=0 and the bug
  // detection becomes timing-dependent.
  pools[1]->wait_fence(pools[1], (struct io_event){ .seq = 1 });

  struct drain_args da = {
    .sinks = sinks,
    .nlods = nlods,
    .n = N_SINKS,
  };
  atomic_store(&da.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, drain_thread_fn, &da) == 0);

  // Phase 1: nothing released — drain_many must be blocked.
  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  if (atomic_load(&da.done) != 0) {
    log_error("drain_many returned before any sentinel was released");
    goto Cleanup;
  }

  // Phase 2: release ONLY the first sentinel.  A buggy _many that only
  // waits on sinks[0] (e.g., wrong stride or missing loop tail) would
  // unblock here; the correct implementation must still be waiting on
  // sinks[1].
  atomic_store(&gates[0], 1);
  platform_sleep_ns((int64_t)DRAIN_OBSERVE_MS * 1000000LL);
  if (atomic_load(&da.done) != 0) {
    log_error("drain_many returned after only the first sentinel was released "
              "— fan-out is not waiting on every sink");
    goto Cleanup;
  }

  // Phase 3: release the second sentinel; drain_many should now complete.
  atomic_store(&gates[1], 1);

  if (wait_for_done(&da.done, POST_RELEASE_TIMEOUT_MS)) {
    log_error("drain_many did not finish within %d ms after second gate "
              "release",
              POST_RELEASE_TIMEOUT_MS);
    // Don't join — would hang. Leak the worker.
    thr = NULL;
    goto Cleanup;
  }

  test_thread_join(thr);
  thr = NULL;

  if (da.errors) {
    log_error("drain_many reported %d sink errors", da.errors);
    goto Cleanup;
  }

  for (int i = 0; i < N_SINKS; ++i) {
    if (zarr_array_has_error(arrs[i])) {
      log_error("array %d sink reported IO error after drain", i);
      goto Cleanup;
    }
  }

  rc = 0;
  log_info("  PASS");

Cleanup:
  test_thread_join(thr);
  for (int i = 0; i < N_SINKS; ++i) {
    zarr_array_destroy(arrs[i]);
    shard_pool_destroy(pools[i]);
    store_destroy(stores[i]);
  }
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
    snprintf(sub, sizeof(sub), "%s/multi_drain", tmpdir);
    test_mkdir(sub);
    ecode |= test_drain_many_fans_out(sub);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}
