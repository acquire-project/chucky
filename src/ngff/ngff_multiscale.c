#include "ngff/ngff_multiscale.h"
#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "ngff/ngff_metadata.h"
#include "util/prelude.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ngff_multiscale
{
  struct shard_sink base;
  struct store* store;
  struct shard_pool* pool;
  struct zarr_array** levels;
  int nlod;
  uint8_t rank;
  char prefix[4096];
  struct ngff_axis axes[MAX_ZARR_RANK];
};

// --- Group metadata ---

static int
write_ngff_group_metadata(const struct ngff_multiscale* ms)
{
  const struct dimension* level_ptrs[LOD_MAX_LEVELS];
  for (int lv = 0; lv < ms->nlod; ++lv)
    level_ptrs[lv] = zarr_array_dimensions(ms->levels[lv]);

  char json[ZARR_GROUP_JSON_MAX_LENGTH];
  int len = ngff_multiscale_group_json(
    json, sizeof(json), ms->rank, ms->nlod, level_ptrs, ms->axes);
  if (len < 0)
    return 1;

  // Write the full group zarr.json (the ngff function generates the complete
  // JSON including zarr_format, node_type, and attributes).
  char key[4096];
  if (ms->prefix[0])
    snprintf(key, sizeof(key), "%s/zarr.json", ms->prefix);
  else
    snprintf(key, sizeof(key), "zarr.json");
  return ms->store->put(ms->store, key, json, (size_t)len);
}

// --- shard_sink vtable ---

static struct shard_writer*
ngff_multiscale_open(struct shard_sink* self,
                     uint8_t level,
                     uint64_t shard_index)
{
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  CHECK(Fail, level < ms->nlod);
  struct shard_sink* child = zarr_array_as_shard_sink(ms->levels[level]);
  return child->open(child, level, shard_index);
Fail:
  return NULL;
}

static int
ngff_multiscale_update_append(struct shard_sink* self,
                              uint8_t level,
                              uint8_t n_append,
                              const uint64_t* append_sizes)
{
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  if (level >= ms->nlod)
    return 1;

  struct shard_sink* child = zarr_array_as_shard_sink(ms->levels[level]);
  const struct dimension* dims = zarr_array_dimensions(ms->levels[level]);
  uint64_t old_dim0 = dims[0].size;

  if (child->update_append(child, level, n_append, append_sizes))
    return 1;

  if (old_dim0 == append_sizes[0])
    return 0;

  if (write_ngff_group_metadata(ms)) {
    log_error("ngff_multiscale: failed to rewrite group zarr.json for %s",
              ms->prefix);
    return 1;
  }
  return 0;
}

static struct io_event
ngff_multiscale_record_fence_fn(struct shard_sink* self, uint8_t level)
{
  (void)level;
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  return ms->pool->record_fence(ms->pool);
}

static void
ngff_multiscale_wait_fence_fn(struct shard_sink* self,
                              uint8_t level,
                              struct io_event ev)
{
  (void)level;
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  ms->pool->wait_fence(ms->pool, ev);
}

// --- Public API ---

struct ngff_multiscale*
ngff_multiscale_create(struct store* store,
                       struct shard_pool* pool,
                       const char* prefix,
                       const struct ngff_multiscale_config* cfg)
{
  CHECK(Fail, store);
  CHECK(Fail, pool);
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  // Compute LOD plan
  struct lod_plan plan = { 0 };
  int max_lev = cfg->nlod > 0 ? cfg->nlod : LOD_MAX_LEVELS;
  CHECK(Fail,
        lod_plan_init_from_dims(&plan, cfg->dimensions, cfg->rank, max_lev) ==
          0);

  struct ngff_multiscale* ms = (struct ngff_multiscale*)calloc(1, sizeof(*ms));
  CHECK(Fail_plan, ms);

  ms->store = store;
  ms->pool = pool;
  ms->nlod = plan.levels.nlod;
  ms->rank = cfg->rank;
  if (prefix)
    snprintf(ms->prefix, sizeof(ms->prefix), "%s", prefix);
  if (cfg->axes)
    memcpy(ms->axes, cfg->axes, cfg->rank * sizeof(struct ngff_axis));

  ms->base.open = ngff_multiscale_open;
  ms->base.update_append = ngff_multiscale_update_append;
  ms->base.record_fence = ngff_multiscale_record_fence_fn;
  ms->base.wait_fence = ngff_multiscale_wait_fence_fn;

  // Ensure prefix directory exists (parent handles root/intermediate groups)
  if (prefix && prefix[0])
    CHECK(Fail_ms, store->mkdirs(store, prefix) == 0);

  // Create per-level zarr_arrays
  ms->levels =
    (struct zarr_array**)calloc((size_t)plan.levels.nlod, sizeof(void*));
  CHECK(Fail_ms, ms->levels);

  for (int lv = 0; lv < plan.levels.nlod; ++lv) {
    struct dimension lv_dims[MAX_ZARR_RANK];
    uint64_t shard_counts[MAX_ZARR_RANK];
    uint64_t cps[MAX_ZARR_RANK];
    uint64_t shard_inner_count = 1;
    uint8_t na = dims_n_append(cfg->dimensions, cfg->rank);

    for (int d = 0; d < cfg->rank; ++d) {
      lv_dims[d] = cfg->dimensions[d];
      if (d == 0 && cfg->dimensions[0].size == 0)
        lv_dims[d].size = 0;
      else
        lv_dims[d].size = plan.levels.level[lv].dim[d].size;
      lv_dims[d].chunk_size = plan.levels.level[lv].dim[d].chunk_size;
      lv_dims[d].chunks_per_shard =
        plan.levels.level[lv].dim[d].chunks_per_shard;

      cps[d] = plan.levels.level[lv].dim[d].chunks_per_shard;
      shard_counts[d] = plan.levels.level[lv].dim[d].shard_count;
      if (d >= na)
        shard_inner_count *= shard_counts[d];
    }

    char name[8];
    snprintf(name, sizeof(name), "%d", lv);

    char level_prefix[4096];
    if (prefix && prefix[0])
      snprintf(level_prefix, sizeof(level_prefix), "%s/%s", prefix, name);
    else
      snprintf(level_prefix, sizeof(level_prefix), "%s", name);

    // Ensure level directory exists
    CHECK(Fail_levels, store->mkdirs(store, level_prefix) == 0);

    struct zarr_array_config acfg = {
      .data_type = cfg->data_type,
      .fill_value = cfg->fill_value,
      .rank = cfg->rank,
      .dimensions = lv_dims,
      .codec = cfg->codec,
      .shard_counts = shard_counts,
      .chunks_per_shard = cps,
      .shard_inner_count = shard_inner_count,
    };

    ms->levels[lv] = zarr_array_create(store, pool, level_prefix, &acfg);
    CHECK(Fail_levels, ms->levels[lv]);
  }

  // Write NGFF group metadata (overwrites root/prefix group zarr.json)
  CHECK(Fail_levels, write_ngff_group_metadata(ms) == 0);

  lod_plan_free(&plan);
  return ms;

Fail_levels:
  for (int i = 0; i < plan.levels.nlod; ++i) {
    if (ms->levels[i])
      zarr_array_destroy(ms->levels[i]);
  }
  free(ms->levels);
Fail_ms:
  free(ms);
Fail_plan:
  lod_plan_free(&plan);
Fail:
  return NULL;
}

void
ngff_multiscale_destroy(struct ngff_multiscale* ms)
{
  if (!ms)
    return;
  for (int i = 0; i < ms->nlod; ++i)
    zarr_array_destroy(ms->levels[i]);
  free(ms->levels);
  free(ms);
}

struct shard_sink*
ngff_multiscale_as_shard_sink(struct ngff_multiscale* ms)
{
  return ms ? &ms->base : NULL;
}
