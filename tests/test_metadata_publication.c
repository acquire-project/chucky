#include "dimension.h"
#include "ngff.h"
#include "ngff/ngff_multiscale.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/json_validate.h"
#include "zarr/shard_delivery.h"
#include "zarr/zarr_array.h"

#include <stdio.h>
#include <string.h>

// Deterministically hold publication dependencies without worker timing.
// The real scheduler tests cover executing these jobs after IO completion;
// these tests cover immutable metadata, ordering and the sink flush hooks.
struct pending_metadata
{
  struct strbuf key;
  struct strbuf json;
  uint64_t seq;
};

struct deferred_pool
{
  struct shard_pool base;
  struct store* store;
  struct pending_metadata jobs[16];
  size_t head;
  size_t count;
  uint64_t submitted;
  uint64_t ready;
  unsigned flushes;
  int error;
  int reject;
  int fail_completion;
};

static char tmpdir[512];

static void
progress(struct deferred_pool* p)
{
  while (p->head < p->count) {
    struct pending_metadata* job = &p->jobs[p->head];
    if (job->seq - 1 > p->ready)
      break;
    if (p->fail_completion)
      p->error = 1;
    if (!p->error && p->store->put(p->store,
                                   strbuf_cstr(&job->key),
                                   strbuf_cstr(&job->json),
                                   strbuf_len(&job->json)))
      p->error = 1;
    if (job->seq > p->ready)
      p->ready = job->seq;
    strbuf_free(&job->key);
    strbuf_free(&job->json);
    ++p->head;
  }
}

static int
deferred_queue_metadata(struct shard_pool* self,
                        const char* key,
                        const void* data,
                        size_t len)
{
  struct deferred_pool* p = container_of(self, struct deferred_pool, base);
  if (p->reject || p->error || p->count == countof(p->jobs)) {
    p->error = 1;
    return 1;
  }
  struct pending_metadata* job = &p->jobs[p->count];
  if (strbuf_set(&job->key, key) || strbuf_append(&job->json, data, len)) {
    strbuf_free(&job->key);
    strbuf_free(&job->json);
    p->error = 1;
    return 1;
  }
  // The pool orders metadata after its current prefix; no caller supplies
  // or owns the dependency.
  job->seq = ++p->submitted;
  ++p->count;
  return 0;
}

static struct io_event
deferred_record(struct shard_pool* self)
{
  struct deferred_pool* p = container_of(self, struct deferred_pool, base);
  return (struct io_event){ p->submitted };
}

static void
deferred_wait(struct shard_pool* self, struct io_event ev)
{
  struct deferred_pool* p = container_of(self, struct deferred_pool, base);
  if (ev.seq > p->ready)
    p->ready = ev.seq;
  progress(p);
}

static int
deferred_flush(struct shard_pool* self)
{
  struct deferred_pool* p = container_of(self, struct deferred_pool, base);
  ++p->flushes;
  deferred_wait(self, deferred_record(self));
  return p->error;
}

static int
deferred_error(const struct shard_pool* self)
{
  const struct deferred_pool* p =
    container_of(self, struct deferred_pool, base);
  return p->error;
}

static void
deferred_init(struct deferred_pool* p, struct store* store)
{
  *p = (struct deferred_pool){
    .base = { .queue_metadata = deferred_queue_metadata,
              .record_fence = deferred_record,
              .wait_fence = deferred_wait,
              .flush = deferred_flush,
              .has_error = deferred_error },
    .store = store,
  };
}

static struct io_event
data_fence(struct deferred_pool* p)
{
  return (struct io_event){ ++p->submitted };
}

static int
metadata_contains(const char* key, const char* needle)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, key);
  FILE* f = fopen(path, "rb");
  if (!f)
    return 0;
  char json[8192];
  size_t len = fread(json, 1, sizeof(json) - 1, f);
  int ok = !ferror(f) && feof(f);
  fclose(f);
  json[len] = '\0';
  return ok && json_value_is_valid(json, len) && strstr(json, needle) != NULL;
}

static struct zarr_array*
create_array(struct store* store, struct deferred_pool* p, const char* key)
{
  struct dimension dims[] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 4, .name = "t" },
    { .size = 64, .chunk_size = 16, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
  };
  if (store->mkdirs(store, key))
    return NULL;
  return zarr_array_create_with_pool(store, &p->base, 0, key, &cfg);
}

static int
test_array_snapshots(void)
{
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct deferred_pool pool;
  deferred_init(&pool, store);
  struct zarr_array* a = create_array(store, &pool, "array");
  CHECK(Fail_store, a);
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  CHECK(Fail_array, sink->queue_append);
  unsigned initial_flushes = pool.flushes;
  const size_t first_job = pool.count;
  CHECK(Fail_array, pool.head == first_job);

  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"first\"") == 0);
  uint64_t size = 4;
  struct io_event first = data_fence(&pool);
  CHECK(Fail_array, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array, pool.flushes == initial_flushes);
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"shape\":[0,64]"));

  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"second\"") == 0);
  size = 8;
  struct io_event second = data_fence(&pool);
  CHECK(Fail_array, sink->queue_append(sink, 0, 1, &size) == 0);
  size = 999; // Neither the caller's extent nor mutable attrs are borrowed.
  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"latest\"") == 0);
  CHECK(Fail_array, pool.count == first_job + 2 && pool.head == first_job);
  CHECK(Fail_array, pool.flushes == initial_flushes);
  CHECK(Fail_array, pool.jobs[first_job].seq == first.seq + 1);
  CHECK(Fail_array, pool.jobs[first_job + 1].seq == second.seq + 1);
  CHECK(Fail_array,
        strstr(strbuf_cstr(&pool.jobs[first_job].json), "\"shape\":[4,64]"));
  CHECK(Fail_array,
        strstr(strbuf_cstr(&pool.jobs[first_job].json), "\"tag\":\"first\""));
  CHECK(
    Fail_array,
    strstr(strbuf_cstr(&pool.jobs[first_job + 1].json), "\"shape\":[8,64]"));
  CHECK(
    Fail_array,
    strstr(strbuf_cstr(&pool.jobs[first_job + 1].json), "\"tag\":\"second\""));

  // Completing the first dependency publishes without another append.
  pool.ready = first.seq;
  progress(&pool);
  CHECK(Fail_array, pool.head == first_job + 1);
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"shape\":[4,64]"));
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"tag\":\"first\""));

  // A manual attribute rewrite must drain the older second snapshot first.
  CHECK(Fail_array, zarr_array_flush_metadata(a) == 0);
  CHECK(Fail_array, pool.head == pool.count);
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"shape\":[8,64]"));
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"tag\":\"latest\""));

  size = 16;
  data_fence(&pool);
  CHECK(Fail_array, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array, sink->flush(sink) == 0); // No dirty attributes.
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"shape\":[16,64]"));

  size = 20;
  data_fence(&pool);
  CHECK(Fail_array, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"destroyed\"") == 0);
  zarr_array_destroy(a);
  a = NULL;
  CHECK(Fail_array, pool.head == pool.count);
  CHECK(Fail_array, metadata_contains("array/zarr.json", "\"shape\":[20,64]"));
  CHECK(Fail_array,
        metadata_contains("array/zarr.json", "\"tag\":\"destroyed\""));

  store_destroy(store);
  return 0;
Fail_array:
  zarr_array_destroy(a);
Fail_store:
  deferred_flush(&pool.base);
  store_destroy(store);
Fail:
  return 1;
}

static int
test_array_synchronous_updates(void)
{
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct deferred_pool pool;
  deferred_init(&pool, store);
  struct zarr_array* a = create_array(store, &pool, "sync-updates");
  CHECK(Fail_store, a);
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  CHECK(Fail_array, pool.count > 0 && pool.head == pool.count);
  CHECK(Fail_array,
        metadata_contains("sync-updates/zarr.json", "\"shape\":[0,64]"));
  const size_t initial_jobs = pool.count;
  unsigned initial_flushes = pool.flushes;

  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"earlier\"") == 0);
  uint64_t size = 4;
  data_fence(&pool);
  CHECK(Fail_array, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array,
        metadata_contains("sync-updates/zarr.json", "\"shape\":[0,64]"));

  // A changed synchronous update joins the same ordered queue and waits for
  // its own snapshot, so the older snapshot cannot overwrite it afterward.
  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"latest\"") == 0);
  size = 8;
  CHECK(Fail_array, sink->update_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array, pool.count == initial_jobs + 2);
  CHECK(Fail_array, pool.head == pool.count && pool.flushes > initial_flushes);
  CHECK(Fail_array,
        metadata_contains("sync-updates/zarr.json", "\"shape\":[8,64]"));
  CHECK(Fail_array,
        metadata_contains("sync-updates/zarr.json", "\"tag\":\"latest\""));

  // Equal in-memory shape does not mean the accepted snapshot is visible.
  size = 12;
  data_fence(&pool);
  CHECK(Fail_array, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array,
        metadata_contains("sync-updates/zarr.json", "\"shape\":[8,64]"));
  initial_flushes = pool.flushes;
  CHECK(Fail_array, sink->update_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array, pool.head == pool.count && pool.flushes > initial_flushes);
  CHECK(Fail_array,
        metadata_contains("sync-updates/zarr.json", "\"shape\":[12,64]"));

  zarr_array_destroy(a);
  store_destroy(store);
  return 0;
Fail_array:
  zarr_array_destroy(a);
Fail_store:
  deferred_flush(&pool.base);
  store_destroy(store);
Fail:
  return 1;
}

static int
test_array_failure(int reject, int synchronous)
{
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct deferred_pool pool;
  deferred_init(&pool, store);
  struct zarr_array* a = create_array(store, &pool, "failure");
  CHECK(Fail_store, a);
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  pool.reject = reject;
  pool.fail_completion = !reject;
  uint64_t size = 4;
  data_fence(&pool);
  int rc = synchronous ? sink->update_append(sink, 0, 1, &size)
                       : sink->queue_append(sink, 0, 1, &size);
  CHECK(Fail_array, (reject || synchronous) ? rc != 0 : rc == 0);
  if (reject)
    CHECK(Fail_array, zarr_array_dimensions(a)[0].size == 0);
  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"unsafe\"") == 0);
  CHECK(Fail_array, sink->flush(sink) != 0);
  CHECK(Fail_array, zarr_array_has_error(a) != 0);
  // Clearing the injected fault does not clear the sticky IO error. A later
  // synchronous rewrite or destroy must not expose the failed extent.
  pool.reject = 0;
  pool.fail_completion = 0;
  CHECK(Fail_array, sink->update_append(sink, 0, 1, &size) != 0);
  zarr_array_destroy(a);
  a = NULL;
  CHECK(Fail_array, metadata_contains("failure/zarr.json", "\"shape\":[0,64]"));
  CHECK(Fail_array, !metadata_contains("failure/zarr.json", "unsafe"));
  store_destroy(store);
  return 0;
Fail_array:
  zarr_array_destroy(a);
Fail_store:
  deferred_flush(&pool.base);
  store_destroy(store);
Fail:
  return 1;
}

static int
test_synchronous_pool(void)
{
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct deferred_pool pool;
  deferred_init(&pool, store);
  pool.base.queue_metadata = NULL;
  struct zarr_array* a = create_array(store, &pool, "synchronous");
  CHECK(Fail_store, a);
  struct ngff_multiscale* ms = NULL;
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  CHECK(Fail_array, !sink->queue_append);
  const struct io_event outstanding = data_fence(&pool);
  uint64_t size = 4;
  CHECK(Fail_array, sink->update_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array,
        metadata_contains("synchronous/zarr.json", "\"shape\":[4,64]"));
  // S3/custom synchronous metadata must not flush active uploads, even when
  // there is outstanding shard IO or the requested shape is unchanged.
  CHECK(Fail_array, sink->update_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_array, zarr_array_set_attribute(a, "tag", "\"direct\"") == 0);
  CHECK(Fail_array, zarr_array_flush_metadata(a) == 0);
  CHECK(Fail_array,
        metadata_contains("synchronous/zarr.json", "\"tag\":\"direct\""));
  CHECK(Fail_array, pool.flushes == 0 && pool.count == 0);
  CHECK(Fail_array, pool.ready < outstanding.seq);

  struct dimension dims[] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 4, .name = "t" },
    { .size = 64,
      .chunk_size = 8,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };
  const struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .nlod = 2,
  };
  ms = ngff_multiscale_create_with_pool(store, &pool.base, "direct-ms", &cfg);
  CHECK(Fail_array, ms);
  CHECK(Fail_ms, ngff_multiscale_set_attribute(ms, "tag", "\"direct\"") == 0);
  CHECK(Fail_ms, ngff_multiscale_flush_metadata(ms) == 0);
  CHECK(Fail_ms,
        metadata_contains("direct-ms/zarr.json", "\"tag\":\"direct\""));
  CHECK(Fail_ms, pool.flushes == 0 && pool.count == 0);
  CHECK(Fail_ms, pool.ready < outstanding.seq);
  ngff_multiscale_destroy(ms);
  ms = NULL;
  zarr_array_destroy(a);
  CHECK(Fail_store, pool.flushes == 0);
  store_destroy(store);
  return 0;
Fail_ms:
  ngff_multiscale_destroy(ms);
Fail_array:
  zarr_array_destroy(a);
Fail_store:
  store_destroy(store);
Fail:
  return 1;
}

static int
test_async_hook_without_flush(void)
{
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct deferred_pool pool;
  deferred_init(&pool, store);
  struct zarr_array* a = create_array(store, &pool, "no-flush");
  CHECK(Fail_store, a);
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  CHECK(Fail_array, sink->queue_append);
  sink->flush = NULL;

  const struct dimension dim = { .size = 0,
                                 .chunk_size = 1,
                                 .chunks_per_shard = 4 };
  struct dim_info info;
  CHECK(Fail_array, dim_info_init(&info, &dim, 1) == 0);
  struct shard_state state = {
    .finalized_append_chunks = 4,
    .finalized_fence = data_fence(&pool),
    .fence_pending = 1,
  };
  CHECK(Fail_array,
        shard_state_publish_append(&state, sink, &info, 0, NULL, NULL) == 0);
  // Close cannot drain this sink's queued metadata. Publication must use
  // the synchronous callback and consume its data fence before returning.
  CHECK(Fail_array, pool.head == pool.count && state.fence_pending == 0);
  CHECK(Fail_array, pool.ready >= state.finalized_fence.seq);
  CHECK(Fail_array,
        metadata_contains("no-flush/zarr.json", "\"shape\":[4,64]"));

  zarr_array_destroy(a);
  store_destroy(store);
  return 0;
Fail_array:
  zarr_array_destroy(a);
Fail_store:
  deferred_flush(&pool.base);
  store_destroy(store);
Fail:
  return 1;
}

static int
test_ngff_snapshots(void)
{
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct deferred_pool pool;
  deferred_init(&pool, store);
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .name = "t",
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 8,
      .name = "x",
      .storage_position = 1,
      .downsample = 1 },
  };
  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .nlod = 2,
  };
  struct ngff_multiscale* ms =
    ngff_multiscale_create_with_pool(store, &pool.base, "ms", &cfg);
  CHECK(Fail_store, ms);
  struct shard_sink* sink = ngff_multiscale_as_shard_sink(ms);
  CHECK(Fail_ms, sink->queue_append);
  unsigned initial_flushes = pool.flushes;
  const size_t first_job = pool.count;
  CHECK(Fail_ms, pool.head == first_job);
  CHECK(Fail_ms, metadata_contains("ms/zarr.json", "\"attributes\":{\"ome\""));

  CHECK(Fail_ms, ngff_multiscale_set_attribute(ms, "tag", "\"first\"") == 0);
  uint64_t size = 4;
  struct io_event first = data_fence(&pool);
  CHECK(Fail_ms, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_ms, ngff_multiscale_set_attribute(ms, "tag", "\"second\"") == 0);
  struct io_event second = data_fence(&pool);
  CHECK(Fail_ms, sink->queue_append(sink, 1, 1, &size) == 0);
  CHECK(Fail_ms,
        pool.flushes == initial_flushes && pool.count == first_job + 4);
  CHECK(Fail_ms, pool.jobs[first_job].seq == first.seq + 1);
  CHECK(Fail_ms, pool.jobs[first_job + 2].seq == second.seq + 1);
  CHECK(Fail_ms,
        strcmp(strbuf_cstr(&pool.jobs[first_job].key), "ms/0/zarr.json") == 0);
  CHECK(Fail_ms,
        strcmp(strbuf_cstr(&pool.jobs[first_job + 1].key), "ms/zarr.json") ==
          0);
  CHECK(Fail_ms,
        strcmp(strbuf_cstr(&pool.jobs[first_job + 2].key), "ms/1/zarr.json") ==
          0);
  CHECK(Fail_ms,
        strcmp(strbuf_cstr(&pool.jobs[first_job + 3].key), "ms/zarr.json") ==
          0);
  CHECK(
    Fail_ms,
    strstr(strbuf_cstr(&pool.jobs[first_job + 1].json), "\"tag\":\"first\""));
  CHECK(
    Fail_ms,
    strstr(strbuf_cstr(&pool.jobs[first_job + 3].json), "\"tag\":\"second\""));
  CHECK(Fail_ms, metadata_contains("ms/0/zarr.json", "\"shape\":[0,64]"));

  pool.ready = first.seq;
  progress(&pool);
  CHECK(Fail_ms, pool.head == first_job + 2);
  CHECK(Fail_ms, metadata_contains("ms/0/zarr.json", "\"shape\":[4,64]"));
  CHECK(Fail_ms, metadata_contains("ms/1/zarr.json", "\"shape\":[0,32]"));
  CHECK(Fail_ms, metadata_contains("ms/zarr.json", "\"tag\":\"first\""));
  CHECK(Fail_ms, ngff_multiscale_set_attribute(ms, "tag", "\"latest\"") == 0);
  CHECK(Fail_ms, ngff_multiscale_flush_metadata(ms) == 0);
  CHECK(Fail_ms, pool.head == pool.count);
  CHECK(Fail_ms, metadata_contains("ms/1/zarr.json", "\"shape\":[4,32]"));
  CHECK(Fail_ms, metadata_contains("ms/zarr.json", "\"tag\":\"latest\""));

  // Synchronous publication uses the same child-before-group order and does
  // not return until both snapshots are visible.
  const size_t before_sync = pool.count;
  initial_flushes = pool.flushes;
  size = 6;
  data_fence(&pool);
  CHECK(Fail_ms, sink->update_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_ms, pool.count == before_sync + 2 && pool.head == pool.count);
  CHECK(Fail_ms, pool.flushes > initial_flushes);
  CHECK(Fail_ms, metadata_contains("ms/0/zarr.json", "\"shape\":[6,64]"));
  CHECK(Fail_ms, metadata_contains("ms/zarr.json", "\"tag\":\"latest\""));

  size = 8;
  data_fence(&pool);
  CHECK(Fail_ms, sink->queue_append(sink, 0, 1, &size) == 0);
  CHECK(Fail_ms,
        ngff_multiscale_set_attribute(ms, "tag", "\"destroyed\"") == 0);
  CHECK(Fail_ms,
        zarr_array_set_attribute(ngff_multiscale_level(ms, 0), "child", "1") ==
          0);
  ngff_multiscale_destroy(ms);
  ms = NULL;
  CHECK(Fail_ms, pool.head == pool.count);
  CHECK(Fail_ms, metadata_contains("ms/0/zarr.json", "\"shape\":[8,64]"));
  CHECK(Fail_ms, metadata_contains("ms/0/zarr.json", "\"child\":1"));
  CHECK(Fail_ms, metadata_contains("ms/zarr.json", "\"tag\":\"destroyed\""));
  store_destroy(store);
  return 0;
Fail_ms:
  ngff_multiscale_destroy(ms);
Fail_store:
  deferred_flush(&pool.base);
  store_destroy(store);
Fail:
  return 1;
}

int
main(void)
{
  if (test_tmpdir_create(tmpdir, sizeof(tmpdir)))
    return 1;
  int err = 0;
  err |= test_array_snapshots();
  err |= test_array_synchronous_updates();
  err |= test_array_failure(0, 0);
  err |= test_array_failure(1, 0);
  err |= test_array_failure(0, 1);
  err |= test_array_failure(1, 1);
  err |= test_synchronous_pool();
  err |= test_async_hook_without_flush();
  err |= test_ngff_snapshots();
  test_tmpdir_remove(tmpdir);
  return err;
}
