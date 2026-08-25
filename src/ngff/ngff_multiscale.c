#include "ngff/ngff_multiscale.h"
#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "ngff.h"
#include "ngff/ngff_metadata.h"
#include "util/prelude.h"
#include "zarr/attr_set.h"
#include "zarr/zarr_array.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
ngff_axes_copy(struct ngff_axis* dst, const struct ngff_axis* src, uint8_t rank)
{
  if (!dst || !src)
    return 1;
  for (uint8_t d = 0; d < rank; ++d) {
    dst[d] = src[d];
    // Null the aliased source pointer before any allocation so the cleanup
    // below never sees an un-duplicated pointer from src.
    dst[d].unit = NULL;
    if (src[d].unit) {
      char* dup = strdup(src[d].unit);
      if (!dup) {
        ngff_axes_free_units(dst, d);
        for (uint8_t z = d; z < rank; ++z)
          dst[z] = (struct ngff_axis){ 0 };
        return 1;
      }
      dst[d].unit = dup;
    }
  }
  return 0;
}

void
ngff_axes_free_units(struct ngff_axis* axes, uint8_t rank)
{
  if (!axes)
    return;
  for (uint8_t d = 0; d < rank; ++d) {
    free((char*)axes[d].unit);
    axes[d].unit = NULL;
  }
}

struct ngff_multiscale
{
  struct shard_sink base;
  struct store* store;
  struct shard_pool* pool; // borrowed or owned (see owns_pool)
  int owns_pool;
  struct zarr_array** levels;
  int nlod;
  uint8_t rank;
  struct strbuf prefix; // owned
  struct ngff_axis axes[MAX_ZARR_RANK];
  struct attr_set attrs;
};

// --- Group metadata ---

static int
write_ngff_group_metadata(struct ngff_multiscale* ms)
{
  const struct dimension* level_ptrs[LOD_MAX_LEVELS];
  for (int lv = 0; lv < ms->nlod; ++lv)
    level_ptrs[lv] = zarr_array_dimensions(ms->levels[lv]);

  struct strbuf key = { 0 };
  struct strbuf json = { 0 };
  int rc = 1;

  if (ngff_multiscale_group_json(
        &json, ms->rank, ms->nlod, level_ptrs, ms->axes, &ms->attrs))
    goto done;

  if (strbuf_len(&ms->prefix) > 0) {
    if (strbuf_appendf(&key, "%s/zarr.json", strbuf_cstr(&ms->prefix)))
      goto done;
  } else {
    if (strbuf_append_cstr(&key, "zarr.json"))
      goto done;
  }

  rc = ms->store->put(
    ms->store, strbuf_cstr(&key), strbuf_cstr(&json), strbuf_len(&json));
  if (rc == 0)
    ms->attrs.dirty = 0;

done:
  strbuf_free(&key);
  strbuf_free(&json);
  return rc;
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
              strbuf_cstr(&ms->prefix));
    return 1;
  }
  return 0;
}

static struct io_event
ngff_multiscale_record_fence_fn(struct shard_sink* self)
{
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  return ms->pool->record_fence(ms->pool);
}

static void
ngff_multiscale_wait_fence_fn(struct shard_sink* self, struct io_event ev)
{
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  ms->pool->wait_fence(ms->pool, ev);
}

static int
ngff_multiscale_has_error_fn(const struct shard_sink* self)
{
  const struct ngff_multiscale* ms =
    container_of(self, struct ngff_multiscale, base);
  return ms->pool->has_error(ms->pool);
}

static uint64_t
ngff_multiscale_pending_bytes_fn(const struct shard_sink* self)
{
  const struct ngff_multiscale* ms =
    container_of(self, struct ngff_multiscale, base);
  return shard_pool_pending_bytes(ms->pool);
}

static size_t
ngff_multiscale_required_shard_alignment_fn(const struct shard_sink* self)
{
  const struct ngff_multiscale* ms =
    container_of(self, struct ngff_multiscale, base);
  return shard_pool_required_shard_alignment(ms->pool);
}

static int
ngff_multiscale_flush_fn(struct shard_sink* self)
{
  struct ngff_multiscale* ms = container_of(self, struct ngff_multiscale, base);
  int rc = 0;
  for (int lv = 0; lv < ms->nlod; ++lv) {
    struct shard_sink* child = zarr_array_as_shard_sink(ms->levels[lv]);
    if (child && child->flush)
      rc |= child->flush(child);
  }
  if (ms->attrs.dirty)
    rc |= write_ngff_group_metadata(ms);
  return rc;
}

// --- Shared create logic ---

static int
plan_from_config(struct lod_plan* plan,
                 const struct ngff_multiscale_config* cfg)
{
  int max_lev = cfg->nlod > 0 ? cfg->nlod : LOD_MAX_LEVELS;
  return lod_plan_init_from_dims(plan, cfg->dimensions, cfg->rank, max_lev, 0);
}

static void
level_dimensions(struct dimension* out,
                 const struct ngff_multiscale_config* cfg,
                 const struct lod_plan* plan,
                 int level)
{
  const struct level_dims* ld = &plan->levels.level[level];
  for (int d = 0; d < cfg->rank; ++d) {
    out[d] = cfg->dimensions[d];
    out[d].size =
      (d == 0 && cfg->dimensions[0].size == 0) ? 0 : ld->dim[d].size;
    out[d].chunk_size = ld->dim[d].chunk_size;
    out[d].chunks_per_shard = ld->dim[d].chunks_per_shard;
  }
}

// Both the count that spaces slot ranges and the create loop below reach a
// level's slots through here, so the two cannot disagree.
static uint64_t
dims_slot_count(const struct dimension* dims, uint8_t rank)
{
  uint64_t shard_counts[MAX_ZARR_RANK], chunks_per_shard[MAX_ZARR_RANK];
  return dims_compute_shard_geometry(
    dims, rank, shard_counts, chunks_per_shard);
}

// Shared init: caller provides a pre-computed LOD plan.
// The plan is consumed (freed on success, freed on failure).
static struct ngff_multiscale*
ngff_multiscale_init(struct store* store,
                     struct shard_pool* pool,
                     uint64_t slot_base,
                     const char* prefix,
                     const struct ngff_multiscale_config* cfg,
                     struct lod_plan* plan)
{
  struct ngff_multiscale* ms = (struct ngff_multiscale*)calloc(1, sizeof(*ms));
  CHECK(Fail_plan, ms);

  ms->store = store;
  ms->pool = pool;
  ms->owns_pool = 0;
  ms->nlod = plan->levels.nlod;
  ms->rank = cfg->rank;
  attr_set_init(&ms->attrs);
  if (prefix && prefix[0])
    CHECK(Fail_ms, strbuf_set(&ms->prefix, prefix) == 0);
  if (cfg->axes)
    CHECK(Fail_ms, ngff_axes_copy(ms->axes, cfg->axes, cfg->rank) == 0);

  ms->base.open = ngff_multiscale_open;
  ms->base.update_append = ngff_multiscale_update_append;
  ms->base.record_fence = ngff_multiscale_record_fence_fn;
  ms->base.wait_fence = ngff_multiscale_wait_fence_fn;
  ms->base.flush = ngff_multiscale_flush_fn;
  ms->base.has_error = ngff_multiscale_has_error_fn;
  ms->base.pending_bytes = ngff_multiscale_pending_bytes_fn;
  ms->base.required_shard_alignment =
    ngff_multiscale_required_shard_alignment_fn;

  // Ensure prefix directory exists (parent handles root/intermediate groups)
  if (prefix && prefix[0])
    CHECK(Fail_ms, store->mkdirs(store, prefix) == 0);

  // Create per-level zarr_arrays with non-overlapping pool slot ranges.
  ms->levels =
    (struct zarr_array**)calloc((size_t)plan->levels.nlod, sizeof(void*));
  CHECK(Fail_ms, ms->levels);

  for (int lv = 0; lv < plan->levels.nlod; ++lv) {
    struct dimension lv_dims[MAX_ZARR_RANK];
    level_dimensions(lv_dims, cfg, plan, lv);

    struct strbuf level_prefix = { 0 };
    int lp_rc = (prefix && prefix[0])
                  ? strbuf_appendf(&level_prefix, "%s/%d", prefix, lv)
                  : strbuf_appendf(&level_prefix, "%d", lv);
    if (lp_rc) {
      strbuf_free(&level_prefix);
      goto Fail_levels;
    }

    int mkrc = store->mkdirs(store, strbuf_cstr(&level_prefix));

    struct zarr_array_config acfg = {
      .data_type = cfg->data_type,
      .rank = cfg->rank,
      .dimensions = lv_dims,
      .codec = cfg->codec,
    };

    ms->levels[lv] =
      mkrc == 0 ? zarr_array_create_with_pool(
                    store, pool, slot_base, strbuf_cstr(&level_prefix), &acfg)
                : NULL;
    strbuf_free(&level_prefix);
    CHECK(Fail_levels, ms->levels[lv]);

    slot_base += dims_slot_count(lv_dims, cfg->rank);
  }

  CHECK(Fail_levels, write_ngff_group_metadata(ms) == 0);

  lod_plan_free(plan);
  return ms;

Fail_levels:
  for (int i = 0; i < plan->levels.nlod; ++i) {
    if (ms->levels[i])
      zarr_array_destroy(ms->levels[i]);
  }
  free(ms->levels);
Fail_ms:
  ngff_axes_free_units(ms->axes, cfg->rank);
  strbuf_free(&ms->prefix);
  free(ms);
Fail_plan:
  lod_plan_free(plan);
  return NULL;
}

// --- Private API ---

uint64_t
ngff_multiscale_slot_count(const struct ngff_multiscale_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  struct lod_plan plan = { 0 };
  CHECK(Fail, plan_from_config(&plan, cfg) == 0);

  uint64_t total = 0;
  for (int lv = 0; lv < plan.levels.nlod; ++lv) {
    struct dimension lv_dims[MAX_ZARR_RANK];
    level_dimensions(lv_dims, cfg, &plan, lv);
    total += dims_slot_count(lv_dims, cfg->rank);
  }
  lod_plan_free(&plan);
  return total;

Fail:
  return 0;
}

struct ngff_multiscale*
ngff_multiscale_create_with_pool(struct store* store,
                                 struct shard_pool* pool,
                                 uint64_t slot_base,
                                 const char* prefix,
                                 const struct ngff_multiscale_config* cfg)
{
  CHECK(Fail, store);
  CHECK(Fail, pool);
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  struct lod_plan plan = { 0 };
  CHECK(Fail, plan_from_config(&plan, cfg) == 0);

  return ngff_multiscale_init(store, pool, slot_base, prefix, cfg, &plan);

Fail:
  return NULL;
}

// --- Public API ---

struct ngff_multiscale*
ngff_multiscale_create(struct store* store,
                       const char* prefix,
                       const struct ngff_multiscale_config* cfg)
{
  CHECK(Fail, store);

  uint64_t total_slots = ngff_multiscale_slot_count(cfg);
  CHECK(Fail, total_slots > 0);

  struct shard_pool* pool = store->create_pool(store, total_slots);
  CHECK(Fail, pool);

  struct ngff_multiscale* ms =
    ngff_multiscale_create_with_pool(store, pool, 0, prefix, cfg);
  if (!ms) {
    shard_pool_destroy(pool);
    return NULL;
  }
  ms->owns_pool = 1;
  return ms;

Fail:
  return NULL;
}

void
ngff_multiscale_destroy(struct ngff_multiscale* ms)
{
  if (!ms)
    return;
  if (ms->attrs.dirty && write_ngff_group_metadata(ms))
    log_error("ngff_multiscale: metadata flush failed during destroy");
  for (int i = 0; i < ms->nlod; ++i)
    zarr_array_destroy(ms->levels[i]);
  free(ms->levels);
  attr_set_destroy(&ms->attrs);
  ngff_axes_free_units(ms->axes, ms->rank);
  strbuf_free(&ms->prefix);
  struct shard_pool* pool = ms->owns_pool ? ms->pool : NULL;
  free(ms);
  shard_pool_destroy(pool);
}

struct shard_sink*
ngff_multiscale_as_shard_sink(struct ngff_multiscale* ms)
{
  return ms ? &ms->base : NULL;
}

int
ngff_multiscale_flush(struct ngff_multiscale* ms)
{
  return ms ? ms->pool->flush(ms->pool) : 0;
}

int
ngff_multiscale_has_error(const struct ngff_multiscale* ms)
{
  return ms ? ms->pool->has_error(ms->pool) : 0;
}

uint64_t
ngff_multiscale_pending_bytes(const struct ngff_multiscale* ms)
{
  return ms ? ms->pool->pending_bytes(ms->pool) : 0;
}

void
ngff_multiscale_io_stats(const struct ngff_multiscale* ms,
                         struct shard_pool_io_stats* out)
{
  shard_pool_io_stats(ms ? ms->pool : NULL, out);
}

int
ngff_multiscale_set_attribute(struct ngff_multiscale* ms,
                              const char* attr_key,
                              const char* json_value)
{
  CHECK(Fail, ms);
  return attr_set_upsert(&ms->attrs, attr_key, json_value);
Fail:
  return 1;
}

int
ngff_multiscale_flush_metadata(struct ngff_multiscale* ms)
{
  CHECK(Fail, ms);
  if (!ms->attrs.dirty)
    return 0;
  return write_ngff_group_metadata(ms);
Fail:
  return 1;
}

struct zarr_array*
ngff_multiscale_level(const struct ngff_multiscale* ms, int level)
{
  if (!ms || level < 0 || level >= ms->nlod)
    return NULL;
  return ms->levels[level];
}
