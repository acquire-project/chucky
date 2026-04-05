#include "zarr_s3_sink.h"
#include "defs.limits.h"
#include "dimension.h"
#include "dtype.h"
#include "lod/lod_plan.h"
#include "ngff/ngff_multiscale.h"
#include "util/prelude.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"
#include "zarr/store_s3.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_group.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

// --- set_defaults / validate (kept from original) ---

static void
s3_transport_set_defaults(size_t* part_size, double* throughput_gbps)
{
  if (*part_size == 0)
    *part_size = S3_DEFAULT_PART_SIZE;
  if (*throughput_gbps == 0.0)
    *throughput_gbps = S3_DEFAULT_THROUGHPUT_GBPS;
}

static int
s3_validate_part_count(uint8_t rank,
                       const struct dimension* dimensions,
                       enum dtype data_type,
                       size_t part_size)
{
  size_t bytes_per_element = dtype_bpe(data_type);
  if (bytes_per_element == 0)
    return 1;

  uint64_t shard_elements = 1;
  uint64_t chunks_per_shard_total = 1;
  for (int d = 0; d < rank; ++d) {
    uint64_t cps = dimensions[d].chunks_per_shard;
    if (cps == 0)
      cps = dimensions[d].size == 0
              ? 1
              : ceildiv(dimensions[d].size, dimensions[d].chunk_size);
    chunks_per_shard_total *= cps;
    shard_elements *= dimensions[d].chunk_size * cps;
  }

  uint64_t shard_data_bytes = shard_elements * bytes_per_element;
  uint64_t index_bytes = chunks_per_shard_total * 16 + 4;
  uint64_t max_shard_bytes = shard_data_bytes + index_bytes;
  uint64_t max_parts = ceildiv(max_shard_bytes, part_size);

  if (max_parts > S3_MAX_PARTS) {
    log_error("shard too large for S3 multipart upload: "
              "%llu bytes (%llu parts with %zu-byte parts, limit %d). "
              "Increase part_size or reduce shard dimensions.",
              (unsigned long long)max_shard_bytes,
              (unsigned long long)max_parts,
              part_size,
              S3_MAX_PARTS);
    return 1;
  }
  return 0;
}

void
zarr_s3_config_set_defaults(struct zarr_s3_config* cfg)
{
  if (!cfg)
    return;
  s3_transport_set_defaults(&cfg->part_size, &cfg->throughput_gbps);
}

static int
s3_valid_key_part(const char* s)
{
  if (!s || s[0] == '\0' || s[0] == '/')
    return 0;
  size_t len = strlen(s);
  return s[len - 1] != '/';
}

int
zarr_s3_config_validate(const struct zarr_s3_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, s3_valid_key_part(cfg->bucket));
  CHECK(Fail, s3_valid_key_part(cfg->prefix));
  CHECK(Fail, s3_valid_key_part(cfg->array_name));
  CHECK(Fail, cfg->region && cfg->region[0]);
  CHECK(Fail, cfg->endpoint && cfg->endpoint[0]);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);
  CHECK(Fail, dims_validate(cfg->dimensions, cfg->rank) == 0);
  CHECK(Fail, dtype_bpe(cfg->data_type) > 0);
  CHECK(Fail, cfg->part_size > 0);
  CHECK(Fail,
        strlen(cfg->prefix) + 1 + strlen(cfg->array_name) +
            sizeof("/zarr.json") <
          4096);
  CHECK(Fail,
        s3_validate_part_count(
          cfg->rank, cfg->dimensions, cfg->data_type, cfg->part_size) == 0);
  return 0;
Fail:
  return 1;
}

void
zarr_s3_multiscale_config_set_defaults(struct zarr_s3_multiscale_config* cfg)
{
  if (!cfg)
    return;
  s3_transport_set_defaults(&cfg->part_size, &cfg->throughput_gbps);
}

int
zarr_s3_multiscale_config_validate(const struct zarr_s3_multiscale_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, s3_valid_key_part(cfg->bucket));
  CHECK(Fail, s3_valid_key_part(cfg->prefix));
  CHECK(Fail, !cfg->array_name || s3_valid_key_part(cfg->array_name));
  CHECK(Fail, cfg->region && cfg->region[0]);
  CHECK(Fail, cfg->endpoint && cfg->endpoint[0]);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);
  CHECK(Fail, dims_validate(cfg->dimensions, cfg->rank) == 0);
  CHECK(Fail, dtype_bpe(cfg->data_type) > 0);
  CHECK(Fail, cfg->part_size > 0);
  if (cfg->array_name)
    CHECK(Fail,
          strlen(cfg->prefix) + 1 + strlen(cfg->array_name) +
              sizeof("/zarr.json") <
            4096);
  CHECK(Fail,
        s3_validate_part_count(
          cfg->rank, cfg->dimensions, cfg->data_type, cfg->part_size) == 0);
  return 0;
Fail:
  return 1;
}

// --- Helper: write root + intermediate groups ---

struct s3_intermediate_ctx
{
  struct store* store;
};

static int
write_s3_intermediate(const char* partial, void* ctx)
{
  const struct s3_intermediate_ctx* c = (const struct s3_intermediate_ctx*)ctx;
  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", partial);
  return zarr_write_group(c->store, key, NULL);
}

static int
write_root_and_intermediates(struct store* store, const char* array_name)
{
  CHECK(Fail, zarr_write_group(store, "zarr.json", NULL) == 0);
  if (array_name) {
    struct s3_intermediate_ctx ictx = { .store = store };
    CHECK(Fail,
          zarr_for_each_intermediate(
            array_name, write_s3_intermediate, &ictx) == 0);
  }
  return 0;
Fail:
  return 1;
}

// --- Single-array sink ---

struct zarr_s3_sink
{
  struct store* store;
  struct shard_pool* pool;
  struct zarr_array* array;
};

struct zarr_s3_sink*
zarr_s3_sink_create(struct zarr_s3_config* cfg)
{
  zarr_s3_config_set_defaults(cfg);
  CHECK(Fail, zarr_s3_config_validate(cfg) == 0);

  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)calloc(1, sizeof(*zs));
  CHECK(Fail, zs);

  struct store_s3_config scfg = {
    .bucket = cfg->bucket,
    .prefix = cfg->prefix,
    .region = cfg->region,
    .endpoint = cfg->endpoint,
    .part_size = cfg->part_size,
    .throughput_gbps = cfg->throughput_gbps,
    .max_retries = cfg->max_retries,
    .backoff_scale_ms = cfg->backoff_scale_ms,
    .max_backoff_secs = cfg->max_backoff_secs,
    .timeout_ns = cfg->timeout_ns,
  };
  zs->store = store_s3_create(&scfg);
  CHECK(Fail_alloc, zs->store);

  // Compute geometry
  uint64_t shard_counts[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_inner_count = dims_compute_shard_geometry(
    cfg->dimensions, cfg->rank, shard_counts, chunks_per_shard);

  zs->pool = zs->store->create_pool(zs->store, shard_inner_count);
  CHECK(Fail_store, zs->pool);

  // Write root + intermediate groups
  CHECK(Fail_pool,
        write_root_and_intermediates(zs->store, cfg->array_name) == 0);

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

  zs->array = zarr_array_create(zs->store, zs->pool, cfg->array_name, &acfg);
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

int
zarr_s3_sink_has_error(const struct zarr_s3_sink* s)
{
  return s ? s->pool->has_error(s->pool) : 0;
}

int
zarr_s3_sink_flush(struct zarr_s3_sink* s)
{
  return s ? s->pool->flush(s->pool) : 0;
}

int
zarr_s3_sink_destroy(struct zarr_s3_sink* s)
{
  if (!s)
    return 0;
  int err = s->pool->flush(s->pool);
  zarr_array_destroy(s->array);
  s->pool->destroy(s->pool);
  s->store->destroy(s->store);
  free(s);
  return err;
}

struct shard_sink*
zarr_s3_sink_as_shard_sink(struct zarr_s3_sink* s)
{
  return s ? zarr_array_as_shard_sink(s->array) : NULL;
}

// --- Multiscale sink ---

struct zarr_s3_multiscale_sink
{
  struct store* store;
  struct shard_pool* pool;
  struct ngff_multiscale* ms;
};

struct zarr_s3_multiscale_sink*
zarr_s3_multiscale_sink_create(struct zarr_s3_multiscale_config* cfg)
{
  zarr_s3_multiscale_config_set_defaults(cfg);
  CHECK(Fail, zarr_s3_multiscale_config_validate(cfg) == 0);

  // Build group prefix for the store's key space
  char group_name[4096] = "";
  if (cfg->array_name)
    snprintf(group_name, sizeof(group_name), "%s", cfg->array_name);

  struct zarr_s3_multiscale_sink* s =
    (struct zarr_s3_multiscale_sink*)calloc(1, sizeof(*s));
  CHECK(Fail, s);

  struct store_s3_config scfg = {
    .bucket = cfg->bucket,
    .prefix = cfg->prefix,
    .region = cfg->region,
    .endpoint = cfg->endpoint,
    .part_size = cfg->part_size,
    .throughput_gbps = cfg->throughput_gbps,
    .max_retries = cfg->max_retries,
    .backoff_scale_ms = cfg->backoff_scale_ms,
    .max_backoff_secs = cfg->max_backoff_secs,
    .timeout_ns = cfg->timeout_ns,
  };
  s->store = store_s3_create(&scfg);
  CHECK(Fail_alloc, s->store);

  // Compute L0 geometry for pool sizing
  uint64_t shard_counts[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_inner_count = dims_compute_shard_geometry(
    cfg->dimensions, cfg->rank, shard_counts, chunks_per_shard);

  s->pool = s->store->create_pool(s->store, shard_inner_count);
  CHECK(Fail_store, s->pool);

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

  s->ms = ngff_multiscale_create(s->store, s->pool, group_name, &mscfg);
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

int
zarr_s3_multiscale_sink_flush(struct zarr_s3_multiscale_sink* s)
{
  return s ? s->pool->flush(s->pool) : 0;
}

int
zarr_s3_multiscale_sink_destroy(struct zarr_s3_multiscale_sink* s)
{
  if (!s)
    return 0;
  int err = s->pool->flush(s->pool);
  ngff_multiscale_destroy(s->ms);
  s->pool->destroy(s->pool);
  s->store->destroy(s->store);
  free(s);
  return err;
}

struct shard_sink*
zarr_s3_multiscale_sink_as_shard_sink(struct zarr_s3_multiscale_sink* s)
{
  return s ? ngff_multiscale_as_shard_sink(s->ms) : NULL;
}
