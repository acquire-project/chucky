#include "zarr/zarr_array.h"
#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/attr_set.h"
#include "zarr/zarr_metadata.h"

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
write_array_metadata(struct zarr_array* a)
{
  struct strbuf key = { 0 };
  struct strbuf json = { 0 };
  int rc = 1;

  if (strbuf_len(&a->prefix) > 0) {
    if (strbuf_appendf(&key, "%s/zarr.json", strbuf_cstr(&a->prefix)))
      goto done;
  } else {
    if (strbuf_append_cstr(&key, "zarr.json"))
      goto done;
  }

  if (zarr_array_json(&json,
                      a->rank,
                      a->dimensions,
                      a->data_type,
                      a->fill_value,
                      a->chunks_per_shard,
                      a->codec,
                      &a->attrs))
    goto done;

  rc = a->store->put(
    a->store, strbuf_cstr(&key), strbuf_cstr(&json), strbuf_len(&json));
  if (rc == 0)
    a->attrs.dirty = 0;

done:
  strbuf_free(&key);
  strbuf_free(&json);
  return rc;
}

// --- shard_sink vtable ---

static struct shard_writer*
zarr_array_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);

  uint64_t slot = a->slot_base + shard_index % a->shard_inner_count;

  char suffix[256];
  if (zarr_shard_key(
        suffix, sizeof(suffix), a->rank, a->shard_counts, shard_index) != 0) {
    log_error("zarr_array: shard key too long for shard %llu",
              (unsigned long long)shard_index);
    return NULL;
  }

  struct strbuf key = { 0 };
  int ok;
  if (strbuf_len(&a->prefix) > 0)
    ok = strbuf_appendf(&key, "%s/%s", strbuf_cstr(&a->prefix), suffix) == 0;
  else
    ok = strbuf_append_cstr(&key, suffix) == 0;

  struct shard_writer* w =
    ok ? a->pool->open(a->pool, slot, strbuf_cstr(&key)) : NULL;
  strbuf_free(&key);
  return w;
}

static int
zarr_array_update_append(struct shard_sink* self,
                         uint8_t level,
                         uint8_t n_append,
                         const uint64_t* append_sizes)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  if (n_append == 0 || n_append > a->rank)
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

  for (uint8_t d = 0; d < n_append; ++d)
    a->dimensions[d].size = append_sizes[d];

  if (write_array_metadata(a)) {
    log_error("zarr_array: failed to rewrite zarr.json for %s",
              strbuf_cstr(&a->prefix));
    return 1;
  }
  return 0;
}

static struct io_event
zarr_array_record_fence_fn(struct shard_sink* self, uint8_t level)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  return a->pool->record_fence(a->pool);
}

static void
zarr_array_wait_fence_fn(struct shard_sink* self,
                         uint8_t level,
                         struct io_event ev)
{
  (void)level;
  struct zarr_array* a = container_of(self, struct zarr_array, base);
  a->pool->wait_fence(a->pool, ev);
}

static int
zarr_array_has_error_fn(const struct shard_sink* self)
{
  const struct zarr_array* a = container_of(self, struct zarr_array, base);
  return a->pool->has_error(a->pool);
}

static size_t
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

  a->base.open = zarr_array_open;
  a->base.update_append = zarr_array_update_append;
  a->base.record_fence = zarr_array_record_fence_fn;
  a->base.wait_fence = zarr_array_wait_fence_fn;
  a->base.has_error = zarr_array_has_error_fn;
  a->base.pending_bytes = zarr_array_pending_bytes_fn;
  a->base.required_shard_alignment = zarr_array_required_shard_alignment_fn;

  // Write array zarr.json
  CHECK(Fail_alloc, write_array_metadata(a) == 0);

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
    pool->destroy(pool);
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
  // Flush dirty attrs before teardown.
  if (a->attrs.dirty)
    write_array_metadata(a);
  attr_set_destroy(&a->attrs);
  dims_free_names(a->dimensions, a->rank);
  strbuf_free(&a->prefix);
  struct shard_pool* pool = a->owns_pool ? a->pool : NULL;
  free(a);
  if (pool)
    pool->destroy(pool);
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

size_t
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
  if (!a->attrs.dirty)
    return 0;
  return write_array_metadata(a);
Fail:
  return 1;
}
