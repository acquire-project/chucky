#include "zarr_fs_sink.h"
#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "ngff/ngff_multiscale.h"
#include "util/prelude.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_group.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

// --- Helper: write root + intermediate groups ---

struct fs_intermediate_ctx
{
  struct store* store;
};

static int
write_fs_intermediate(const char* partial, void* ctx)
{
  const struct fs_intermediate_ctx* c = (const struct fs_intermediate_ctx*)ctx;
  if (c->store->mkdirs(c->store, partial))
    return -1;
  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", partial);
  return zarr_write_group(c->store, key, NULL);
}

static int
write_root_and_intermediates(struct store* store, const char* array_name)
{
  CHECK(Fail, zarr_write_group(store, "zarr.json", NULL) == 0);
  if (array_name) {
    struct fs_intermediate_ctx ictx = { .store = store };
    CHECK(Fail,
          zarr_for_each_intermediate(
            array_name, write_fs_intermediate, &ictx) == 0);
    CHECK(Fail, store->mkdirs(store, array_name) == 0);
  }
  return 0;
Fail:
  return 1;
}

// --- Single-array sink ---

struct zarr_fs_sink
{
  struct store* store;
  struct shard_pool* pool;
  struct zarr_array* array;
};

struct zarr_fs_sink*
zarr_fs_sink_create(const struct zarr_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->store_path);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);
  if (cfg->array_name)
    CHECK(Fail,
          strlen(cfg->store_path) + 1 + strlen(cfg->array_name) +
              sizeof("/zarr.json") <
            4096);

  struct zarr_fs_sink* zs = (struct zarr_fs_sink*)calloc(1, sizeof(*zs));
  CHECK(Fail, zs);

  zs->store = store_fs_create(cfg->store_path, cfg->unbuffered);
  CHECK(Fail_alloc, zs->store);

  // Compute geometry
  uint64_t shard_counts[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_inner_count = dims_compute_shard_geometry(
    cfg->dimensions, cfg->rank, shard_counts, chunks_per_shard);

  zs->pool = zs->store->create_pool(zs->store, shard_inner_count);
  CHECK(Fail_store, zs->pool);

  // Write root + intermediate group metadata
  if (cfg->array_name)
    CHECK(Fail_pool,
          write_root_and_intermediates(zs->store, cfg->array_name) == 0);

  // Build array prefix (array_name or "")
  const char* prefix = cfg->array_name ? cfg->array_name : "";

  struct zarr_array_config acfg = {
    .data_type = cfg->data_type,
    .fill_value = cfg->fill_value,
    .rank = cfg->rank,
    .dimensions = cfg->dimensions,
    .codec = cfg->codec,
    .shard_counts = shard_counts,
    .chunks_per_shard = chunks_per_shard,
    .shard_inner_count = shard_inner_count,
  };

  zs->array = zarr_array_create(zs->store, zs->pool, prefix, &acfg);
  CHECK(Fail_pool, zs->array);

  return zs;

Fail_pool:
  zs->pool->destroy(zs->pool);
Fail_store:
  zs->store->destroy(zs->store);
Fail_alloc:
  free(zs);
Fail:
  return NULL;
}

size_t
zarr_fs_sink_pending_bytes(struct zarr_fs_sink* s)
{
  return s ? s->pool->pending_bytes(s->pool) : 0;
}

void
zarr_fs_sink_flush(struct zarr_fs_sink* s)
{
  if (s)
    s->pool->flush(s->pool);
}

void
zarr_fs_sink_destroy(struct zarr_fs_sink* s)
{
  if (!s)
    return;
  zarr_array_destroy(s->array);
  s->pool->destroy(s->pool);
  s->store->destroy(s->store);
  free(s);
}

struct shard_sink*
zarr_fs_sink_as_shard_sink(struct zarr_fs_sink* s)
{
  return s ? zarr_array_as_shard_sink(s->array) : NULL;
}

// --- Multiscale sink ---

struct zarr_fs_multiscale_sink
{
  struct store* store;
  struct shard_pool* pool;
  struct ngff_multiscale* ms;
};

struct zarr_fs_multiscale_sink*
zarr_fs_multiscale_sink_create(const struct zarr_multiscale_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->store_path);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);
  if (cfg->array_name)
    CHECK(Fail,
          strlen(cfg->store_path) + 1 + strlen(cfg->array_name) +
              sizeof("/zarr.json") <
            4096);

  struct zarr_fs_multiscale_sink* s =
    (struct zarr_fs_multiscale_sink*)calloc(1, sizeof(*s));
  CHECK(Fail, s);

  s->store = store_fs_create(cfg->store_path, cfg->unbuffered);
  CHECK(Fail_alloc, s->store);

  // Compute L0 geometry for pool sizing
  uint64_t shard_counts[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_inner_count = dims_compute_shard_geometry(
    cfg->dimensions, cfg->rank, shard_counts, chunks_per_shard);

  s->pool = s->store->create_pool(s->store, shard_inner_count);
  CHECK(Fail_store, s->pool);

  // Build group prefix
  const char* prefix = cfg->array_name ? cfg->array_name : "";

  // Write root + intermediate groups
  CHECK(Fail_pool,
        write_root_and_intermediates(s->store, cfg->array_name) == 0);

  struct ngff_multiscale_config mscfg = {
    .data_type = cfg->data_type,
    .fill_value = cfg->fill_value,
    .rank = cfg->rank,
    .dimensions = cfg->dimensions,
    .nlod = cfg->nlod,
    .codec = cfg->codec,
    .axes = cfg->axes,
  };

  s->ms = ngff_multiscale_create(s->store, s->pool, prefix, &mscfg);
  CHECK(Fail_pool, s->ms);

  return s;

Fail_pool:
  s->pool->destroy(s->pool);
Fail_store:
  s->store->destroy(s->store);
Fail_alloc:
  free(s);
Fail:
  return NULL;
}

size_t
zarr_fs_multiscale_sink_pending_bytes(struct zarr_fs_multiscale_sink* s)
{
  return s ? s->pool->pending_bytes(s->pool) : 0;
}

void
zarr_fs_multiscale_sink_flush(struct zarr_fs_multiscale_sink* s)
{
  if (s)
    s->pool->flush(s->pool);
}

void
zarr_fs_multiscale_sink_destroy(struct zarr_fs_multiscale_sink* s)
{
  if (!s)
    return;
  ngff_multiscale_destroy(s->ms);
  s->pool->destroy(s->pool);
  s->store->destroy(s->store);
  free(s);
}

struct shard_sink*
zarr_fs_multiscale_sink_as_shard_sink(struct zarr_fs_multiscale_sink* s)
{
  return s ? ngff_multiscale_as_shard_sink(s->ms) : NULL;
}
