#include "zarr/zarr_array.h"
#include "defs.limits.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct zarr_array
{
  struct shard_sink base;
  struct store* store;     // borrowed
  struct shard_pool* pool; // borrowed or owned (see owns_pool)
  int owns_pool;
  char prefix[4096];

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
};

// --- Metadata writing ---

static int
write_array_metadata(struct zarr_array* a)
{
  char key[4096];
  if (a->prefix[0])
    snprintf(key, sizeof(key), "%s/zarr.json", a->prefix);
  else
    snprintf(key, sizeof(key), "zarr.json");

  char buf[4096];
  int len = zarr_array_json(buf,
                            sizeof(buf),
                            a->rank,
                            a->dimensions,
                            a->data_type,
                            a->fill_value,
                            a->chunks_per_shard,
                            a->codec);
  if (len < 0)
    return 1;
  return a->store->put(a->store, key, buf, (size_t)len);
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

  char key[4096];
  if (a->prefix[0])
    snprintf(key, sizeof(key), "%s/%s", a->prefix, suffix);
  else
    snprintf(key, sizeof(key), "%s", suffix);

  return a->pool->open(a->pool, slot, key);
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
    log_error("zarr_array: failed to rewrite zarr.json for %s", a->prefix);
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

  if (prefix)
    snprintf(a->prefix, sizeof(a->prefix), "%s", prefix);

  for (int d = 0; d < cfg->rank; ++d) {
    a->dimensions[d] = cfg->dimensions[d];
    a->shard_counts[d] = shard_counts[d];
    a->chunks_per_shard[d] = chunks_per_shard[d];
  }

  a->base.open = zarr_array_open;
  a->base.update_append = zarr_array_update_append;
  a->base.record_fence = zarr_array_record_fence_fn;
  a->base.wait_fence = zarr_array_wait_fence_fn;
  a->base.has_error = zarr_array_has_error_fn;
  a->base.pending_bytes = zarr_array_pending_bytes_fn;

  // Write array zarr.json
  CHECK(Fail_alloc, write_array_metadata(a) == 0);

  return a;

Fail_alloc:
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
