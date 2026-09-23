#include "zarr/zarr_array.h"
#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/attr_set.h"
#include "zarr/metadata_io.h"
#include "zarr/zarr_metadata.h"

#include <stdatomic.h>
#include <stdlib.h>

struct zarr_array
{
  struct shard_sink base;
  struct store* store;     // borrowed
  struct shard_pool* pool; // borrowed or owned (see owns_pool)
  int owns_pool;
  struct strbuf prefix; // owned

  uint8_t rank;
  uint64_t shard_counts[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_inner_count;
  uint64_t slot_base; // first pool slot used by this array
  uint64_t shard_limit;
  uint64_t prepared_capacity;
  _Atomic int preparing;

  // Mutable copy for metadata updates
  struct dimension dimensions[MAX_ZARR_RANK];
  enum dtype data_type;
  double fill_value;
  struct codec_config codec;

  // User-supplied custom attributes (buffered until next metadata rewrite).
  struct attr_set attrs;
};

// --- Metadata writing ---

static int
submit_array_metadata(struct zarr_array* a)
{
  struct strbuf json = { 0 };
  int rc = 1;

  if (zarr_array_json(&json,
                      a->rank,
                      a->dimensions,
                      a->data_type,
                      a->fill_value,
                      a->chunks_per_shard,
                      a->codec,
                      &a->attrs))
    goto done;

  rc = zarr_metadata_submit(a->store, a->pool, strbuf_cstr(&a->prefix), &json);
  if (rc == 0)
    a->attrs.dirty = 0;

done:
  strbuf_free(&json);
  return rc;
}

// --- shard_sink vtable ---

static int
array_shard_key(struct zarr_array* a, uint64_t shard_index, struct strbuf* key)
{
  char suffix[256];
  if (zarr_shard_key(
        suffix, sizeof(suffix), a->rank, a->shard_counts, shard_index))
    return 1;
  return strbuf_len(&a->prefix)
           ? strbuf_appendf(key, "%s/%s", strbuf_cstr(&a->prefix), suffix)
           : strbuf_append_cstr(key, suffix);
}

static int
prepare_shard(struct zarr_array* a, uint64_t shard_index)
{
  struct strbuf key = { 0 };
  int rc = array_shard_key(a, shard_index, &key);
  if (!rc)
    rc = a->pool->prepare(a->pool,
                          a->slot_base + shard_index % a->shard_inner_count,
                          strbuf_cstr(&key),
                          a->prepared_capacity);
  strbuf_free(&key);
  return rc;
}

static void
zarr_array_stop_preparing(struct shard_sink* self)
{
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  atomic_store(&a->preparing, 0);
}

static int
zarr_array_cancel_prepared(struct shard_sink* self)
{
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  zarr_array_stop_preparing(self);
  return a->pool->cancel_prepared(a->pool, a->slot_base, a->shard_inner_count);
}

static int
zarr_array_prepare_shards(struct shard_sink* self,
                          uint8_t level,
                          uint64_t capacity)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  if (atomic_load(&a->preparing))
    return 1;
  a->prepared_capacity = capacity;
  for (uint64_t i = 0; i < a->shard_inner_count; ++i)
    if (prepare_shard(a, i))
      goto Fail;
  for (uint64_t i = 0; i < a->shard_inner_count; ++i)
    if (a->pool->wait_prepared(a->pool, a->slot_base + i))
      goto Fail;
  atomic_store(&a->preparing, 1);
  return 0;
Fail:
  if (zarr_array_cancel_prepared(self))
    log_error("zarr_array: preparation rollback failed");
  return 1;
}

static struct shard_writer*
zarr_array_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  uint64_t slot = a->slot_base + shard_index % a->shard_inner_count;
  struct strbuf key = { 0 };
  int rc = array_shard_key(a, shard_index, &key);
  struct shard_writer* w =
    rc ? NULL : a->pool->open(a->pool, slot, strbuf_cstr(&key));
  strbuf_free(&key);
  if (w && atomic_load(&a->preparing) &&
      shard_index <= UINT64_MAX - a->shard_inner_count) {
    const uint64_t next = shard_index + a->shard_inner_count;
    if ((!a->shard_limit || next < a->shard_limit) && prepare_shard(a, next))
      return NULL;
  }
  return w;
}

int
zarr_array_submit_append(struct zarr_array* a,
                         uint8_t n_append,
                         const uint64_t* append_sizes)
{
  if (n_append == 0 || n_append > a->rank || !append_sizes)
    return 1;

  int changed = 0;
  for (uint8_t d = 0; d < n_append; ++d) {
    if (a->dimensions[d].size != append_sizes[d]) {
      changed = 1;
      break;
    }
  }
  if (!changed)
    return 0;

  uint64_t old_sizes[MAX_ZARR_RANK];
  for (uint8_t d = 0; d < n_append; ++d) {
    old_sizes[d] = a->dimensions[d].size;
    a->dimensions[d].size = append_sizes[d];
  }

  if (submit_array_metadata(a)) {
    for (uint8_t d = 0; d < n_append; ++d)
      a->dimensions[d].size = old_sizes[d];
    log_error("zarr_array: failed to rewrite zarr.json for %s",
              strbuf_cstr(&a->prefix));
    return 1;
  }
  return 0;
}

static int
zarr_array_update_append(struct shard_sink* self,
                         uint8_t level,
                         uint8_t n_append,
                         const uint64_t* append_sizes)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  int rc = zarr_array_submit_append(a, n_append, append_sizes);
  // Even an unchanged shape may refer to a snapshot still in the queue.
  rc |= zarr_metadata_wait(a->pool);
  return rc;
}

static int
zarr_array_queue_append(struct shard_sink* self,
                        uint8_t level,
                        uint8_t n_append,
                        const uint64_t* append_sizes)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  // Serialize while dimensions, attributes and prefix are stable. The pool
  // copies the resulting key and bytes; workers never borrow array state.
  return zarr_array_submit_append(a, n_append, append_sizes);
}

static struct io_event
zarr_array_record_fence_fn(struct shard_sink* self)
{
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  return a->pool->record_fence(a->pool);
}

static void
zarr_array_wait_fence_fn(struct shard_sink* self, struct io_event ev)
{
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  a->pool->wait_fence(a->pool, ev);
}

static int
zarr_array_has_error_fn(const struct shard_sink* self)
{
  const struct zarr_array* a = container_of(self, struct zarr_array, base);
  return a->pool->has_error(a->pool);
}

static uint64_t
zarr_array_pending_bytes_fn(const struct shard_sink* self)
{
  const struct zarr_array* a = container_of(self, struct zarr_array, base);
  return shard_pool_pending_bytes(a->pool);
}

static size_t
zarr_array_required_shard_alignment_fn(const struct shard_sink* self)
{
  const struct zarr_array* a = container_of(self, struct zarr_array, base);
  return shard_pool_required_shard_alignment(a->pool);
}

static int
zarr_array_flush_fn(struct shard_sink* self)
{
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  return zarr_array_flush_metadata(a);
}

// --- Core init (geometry already computed) ---

static struct zarr_array*
zarr_array_init(struct store* store,
                struct shard_pool* pool,
                const char* prefix,
                const struct zarr_array_config* cfg,
                const uint64_t* shard_counts,
                const uint64_t* chunks_per_shard,
                uint64_t shard_inner_count,
                uint64_t slot_base)
{
  struct zarr_array* a = (struct zarr_array*)calloc(1, sizeof(*a));
  CHECK(Fail, a);

  a->store = store;
  a->pool = pool;
  a->owns_pool = 0;
  a->rank = cfg->rank;
  a->shard_inner_count = shard_inner_count;
  a->slot_base = slot_base;
  a->data_type = cfg->data_type;
  a->fill_value = cfg->fill_value;
  a->codec = cfg->codec;
  attr_set_init(&a->attrs);

  if (prefix && prefix[0])
    CHECK(Fail_alloc, strbuf_set(&a->prefix, prefix) == 0);

  CHECK(Fail_alloc, dims_copy(a->dimensions, cfg->dimensions, cfg->rank) == 0);
  for (int d = 0; d < cfg->rank; ++d) {
    a->shard_counts[d] = shard_counts[d];
    a->chunks_per_shard[d] = chunks_per_shard[d];
  }

  a->shard_limit = cfg->dimensions[0].size ? 1 : 0;
  for (int d = 0; d < cfg->rank; ++d) {
    if (!shard_counts[d] || a->shard_limit > UINT64_MAX / shard_counts[d]) {
      a->shard_limit = 0;
      break;
    }
    a->shard_limit *= shard_counts[d];
  }

  a->base.open = zarr_array_open;
  if (pool->prepare && pool->wait_prepared && pool->cancel_prepared) {
    a->base.prepare_shards = zarr_array_prepare_shards;
    a->base.stop_preparing = zarr_array_stop_preparing;
    a->base.cancel_prepared = zarr_array_cancel_prepared;
  }
  a->base.update_append = zarr_array_update_append;
  a->base.queue_append = pool->queue_metadata ? zarr_array_queue_append : NULL;
  a->base.record_fence = zarr_array_record_fence_fn;
  a->base.wait_fence = zarr_array_wait_fence_fn;
  a->base.flush = zarr_array_flush_fn;
  a->base.has_error = zarr_array_has_error_fn;
  a->base.pending_bytes = zarr_array_pending_bytes_fn;
  a->base.required_shard_alignment = zarr_array_required_shard_alignment_fn;

  // Creation returns only after the initial metadata is visible.
  int rc = submit_array_metadata(a);
  rc |= zarr_metadata_wait(pool);
  CHECK(Fail_alloc, rc == 0);

  return a;

Fail_alloc:
  dims_free_names(a->dimensions, cfg->rank);
  strbuf_free(&a->prefix);
  free(a);
Fail:
  return NULL;
}

// --- Private API ---

struct zarr_array*
zarr_array_create_with_pool(struct store* store,
                            struct shard_pool* pool,
                            uint64_t slot_base,
                            const char* prefix,
                            const struct zarr_array_config* cfg)
{
  CHECK(Fail, store);
  CHECK(Fail, pool);
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  uint64_t sc[MAX_ZARR_RANK], cps[MAX_ZARR_RANK];
  uint64_t sic =
    dims_compute_shard_geometry(cfg->dimensions, cfg->rank, sc, cps);
  CHECK(Fail, sic > 0);

  return zarr_array_init(store, pool, prefix, cfg, sc, cps, sic, slot_base);

Fail:
  return NULL;
}

// --- Public API ---

struct zarr_array*
zarr_array_create(struct store* store,
                  const char* prefix,
                  const struct zarr_array_config* cfg)
{
  CHECK(Fail, store);
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  uint64_t sc[MAX_ZARR_RANK], cps[MAX_ZARR_RANK];
  uint64_t sic =
    dims_compute_shard_geometry(cfg->dimensions, cfg->rank, sc, cps);
  CHECK(Fail, sic > 0);

  struct shard_pool* pool = store->create_pool(store, sic);
  CHECK(Fail, pool);

  struct zarr_array* a =
    zarr_array_init(store, pool, prefix, cfg, sc, cps, sic, 0);
  if (!a) {
    shard_pool_destroy(pool);
    return NULL;
  }
  a->owns_pool = 1;
  return a;

Fail:
  return NULL;
}

void
zarr_array_destroy(struct zarr_array* a)
{
  if (!a)
    return;
  if (shard_sink_cancel_prepared(&a->base))
    log_error("zarr_array: unused shard cleanup failed during destroy");
  // Fallback for callers that didn't flush via the writer/sink — same shape
  // as the auto-flush log in stream destroys.
  if (zarr_array_flush_metadata(a))
    log_error("zarr_array: metadata flush failed during destroy");
  attr_set_destroy(&a->attrs);
  dims_free_names(a->dimensions, a->rank);
  strbuf_free(&a->prefix);
  // Drain via the sink interface before the pool is destroyed so teardown
  // doesn't depend on shard_pool internals.
  if (shard_sink_drain(&a->base))
    log_error("zarr_array: sink reported IO errors during teardown");
  struct shard_pool* pool = a->owns_pool ? a->pool : NULL;
  free(a);
  shard_pool_destroy(pool);
}

struct shard_sink*
zarr_array_as_shard_sink(struct zarr_array* a)
{
  return a ? &a->base : NULL;
}

int
zarr_array_flush(struct zarr_array* a)
{
  return a ? a->pool->flush(a->pool) : 0;
}

int
zarr_array_has_error(const struct zarr_array* a)
{
  return a ? a->pool->has_error(a->pool) : 0;
}

uint64_t
zarr_array_pending_bytes(const struct zarr_array* a)
{
  return a ? a->pool->pending_bytes(a->pool) : 0;
}

const struct dimension*
zarr_array_dimensions(const struct zarr_array* a)
{
  return a ? a->dimensions : NULL;
}

int
zarr_array_set_attribute(struct zarr_array* a,
                         const char* attr_key,
                         const char* json_value)
{
  CHECK(Fail, a);
  return attr_set_upsert(&a->attrs, attr_key, json_value);
Fail:
  return 1;
}

int
zarr_array_flush_metadata(struct zarr_array* a)
{
  CHECK(Fail, a);
  int rc = a->attrs.dirty ? submit_array_metadata(a) : 0;
  rc |= zarr_metadata_wait(a->pool);
  return rc;
Fail:
  return 1;
}
