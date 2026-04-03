#include "zarr_s3_sink.h"
#include "defs.limits.h"
#include "dimension.h"
#include "dtype.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"
#include "zarr/s3_client.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

// --- set_defaults / validate ---

// Shared: fill transport defaults for any S3 config.
static void
s3_transport_set_defaults(size_t* part_size, double* throughput_gbps)
{
  if (*part_size == 0)
    *part_size = S3_DEFAULT_PART_SIZE;
  if (*throughput_gbps == 0.0)
    *throughput_gbps = S3_DEFAULT_THROUGHPUT_GBPS;
}

// Shared: validate shard size vs part count.
// Uses uncompressed size as an upper bound (compression can only shrink).
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

// Reject empty strings or leading/trailing slashes in key components.
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
  // L0 is the largest level; if it fits, all LOD levels fit.
  CHECK(Fail,
        s3_validate_part_count(
          cfg->rank, cfg->dimensions, cfg->data_type, cfg->part_size) == 0);
  return 0;
Fail:
  return 1;
}

struct s3_intermediate_ctx
{
  struct s3_client* s3;
  const char* bucket;
  const char* prefix;
  const char* metadata;
  size_t metadata_len;
};

static int
put_s3_intermediate(const char* partial, void* ctx)
{
  const struct s3_intermediate_ctx* c = (const struct s3_intermediate_ctx*)ctx;
  char key[4096];
  snprintf(key, sizeof(key), "%s/%s/zarr.json", c->prefix, partial);
  return s3_client_put(c->s3, c->bucket, key, c->metadata, c->metadata_len);
}

// --- S3 shard writer ---

// --- Zarr S3 sink ---

struct zarr_s3_sink
{
  struct shard_sink base;
  struct s3_client* s3;
  int owns_s3; // 1 if we created the client, 0 if borrowed
  char bucket[256];

  // Geometry
  uint8_t rank;
  uint64_t chunk_count[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_count[MAX_ZARR_RANK];
  uint64_t shard_inner_count;

  // Key prefix: "{prefix}/{array_name}"
  char array_prefix[4096];

  // Writer pool
  struct s3_shard_writer* writers;
  uint64_t num_writers;

  // Fence tracking
  uint64_t finalize_seq; // monotonic count of finalizes issued
  int finalize_err;      // sticky error from any finalize

  // Metadata (for update_append)
  struct dimension dimensions[MAX_ZARR_RANK];
  enum dtype data_type;
  double fill_value;
  struct codec_config codec;
};

// --- S3 shard writer ---

struct s3_shard_writer
{
  struct shard_writer base;
  struct zarr_s3_sink* parent;
  struct s3_upload* upload;         // active upload (receiving writes)
  struct s3_upload* pending_upload; // previous upload completing async
  int pending_eof_err;              // EOF send error from finish_async
  int write_err;                    // set on part write failure
};

static int
s3_shard_write(struct shard_writer* self,
               uint64_t offset,
               const void* beg,
               const void* end)
{
  (void)offset; // writes are sequential; CRT handles ordering
  struct s3_shard_writer* w = (struct s3_shard_writer*)self;
  if (!w->upload)
    return 1; // already aborted
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;
  if (s3_upload_write(w->upload, beg, nbytes)) {
    w->write_err = 1;
    w->parent->finalize_err = 1;
    return 1;
  }
  return 0;
}

static int
s3_wait_pending(struct s3_shard_writer* w)
{
  if (!w->pending_upload)
    return 0;
  int rc = w->pending_eof_err;
  rc |= s3_upload_wait(w->pending_upload);
  s3_upload_destroy(w->pending_upload);
  w->pending_upload = NULL;
  w->pending_eof_err = 0;
  if (rc)
    w->parent->finalize_err = 1;
  return rc;
}

static int
s3_shard_finalize(struct shard_writer* self)
{
  struct s3_shard_writer* w = (struct s3_shard_writer*)self;
  if (!w->upload)
    return 0;

  // A part write failed; abort the upload rather than completing it.
  if (w->write_err) {
    s3_upload_abort(w->upload);
    w->upload = NULL;
    w->write_err = 0;
    w->parent->finalize_err = 1;
    ++w->parent->finalize_seq;
    return 1;
  }

  int eof_err = s3_upload_finish_async(w->upload);
  w->pending_upload = w->upload;
  w->pending_eof_err = eof_err;
  w->upload = NULL;
  ++w->parent->finalize_seq;
  return 0; // non-blocking: errors surface at next open() or wait_fence()
}

// --- shard_sink vtable ---

static struct io_event
s3_sink_record_fence(struct shard_sink* self, uint8_t level)
{
  (void)level;
  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)self;
  return (struct io_event){ .seq = zs->finalize_seq };
}

static void
s3_sink_wait_fence(struct shard_sink* self, uint8_t level, struct io_event ev)
{
  (void)level;
  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)self;
  // Wait for all pending uploads to complete.
  // Each writer has at most one pending upload; drain all of them
  // when a fence is requested.
  if (ev.seq == 0)
    return;
  for (uint64_t i = 0; i < zs->num_writers; ++i)
    s3_wait_pending(&zs->writers[i]);
}

static struct shard_writer*
s3_sink_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)self;

  uint64_t inner = shard_index % zs->shard_inner_count;
  struct s3_shard_writer* w = &zs->writers[inner];

  // Wait for previous upload on this writer slot to complete
  s3_wait_pending(w);

  char suffix[4096];
  if (zarr_shard_key(
        suffix, sizeof(suffix), zs->rank, zs->shard_count, shard_index) != 0) {
    log_error("s3_sink_open: key too long for shard %llu",
              (unsigned long long)shard_index);
    return NULL;
  }
  char key[4096];
  snprintf(key, sizeof(key), "%s/%s", zs->array_prefix, suffix);

  w->upload = s3_upload_begin(zs->s3, zs->bucket, key);
  if (!w->upload) {
    log_error(
      "s3_sink_open: failed to begin upload for %s/%s", zs->bucket, key);
    return NULL;
  }

  return &w->base;
}

// --- update_append ---

static int
s3_sink_update_append(struct shard_sink* self,
                      uint8_t level,
                      uint8_t n_append,
                      const uint64_t* append_sizes)
{
  (void)level;
  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)self;
  if (n_append == 0 || n_append > zs->rank)
    return 1;
  int changed = 0;
  for (uint8_t d = 0; d < n_append; ++d) {
    if (zs->dimensions[d].size != append_sizes[d]) {
      changed = 1;
      break;
    }
  }
  if (!changed)
    return 0;
  for (uint8_t d = 0; d < n_append; ++d)
    zs->dimensions[d].size = append_sizes[d];

  char buf[4096];
  int len = zarr_array_json(buf,
                            sizeof(buf),
                            zs->rank,
                            zs->dimensions,
                            zs->data_type,
                            zs->fill_value,
                            zs->chunks_per_shard,
                            zs->codec);
  if (len < 0) {
    log_error("s3_sink_update_append: failed to generate zarr.json for %s",
              zs->array_prefix);
    return 1;
  }
  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", zs->array_prefix);
  if (s3_client_put(zs->s3, zs->bucket, key, buf, (size_t)len)) {
    log_error("s3_sink_update_append: failed to write zarr.json for %s",
              zs->array_prefix);
    return 1;
  }
  return 0;
}

// --- Create / Destroy ---

// Internal: create a sink that borrows an existing client (does not own it).
// When skip_group_metadata is set, root and intermediate group zarr.json
// writes are skipped (the caller handles them, e.g. multiscale).
static struct zarr_s3_sink*
zarr_s3_sink_create_with_client(const struct zarr_s3_config* cfg,
                                struct s3_client* client,
                                int skip_group_metadata)
{
  CHECK(Fail, cfg);
  CHECK(Fail, client);

  struct zarr_s3_sink* zs =
    (struct zarr_s3_sink*)calloc(1, sizeof(struct zarr_s3_sink));
  CHECK(Fail, zs);

  zs->base.open = s3_sink_open;
  zs->base.update_append = s3_sink_update_append;
  zs->base.record_fence = s3_sink_record_fence;
  zs->base.wait_fence = s3_sink_wait_fence;
  zs->rank = cfg->rank;
  zs->data_type = cfg->data_type;
  zs->fill_value = cfg->fill_value;
  zs->codec = cfg->codec;

  snprintf(zs->bucket, sizeof(zs->bucket), "%s", cfg->bucket);

  for (int d = 0; d < cfg->rank; ++d)
    zs->dimensions[d] = cfg->dimensions[d];

  zs->s3 = client;
  zs->owns_s3 = 0;

  // Compute geometry
  {
    uint64_t shape[HALF_MAX_RANK], cs[HALF_MAX_RANK], cps[HALF_MAX_RANK];
    for (int d = 0; d < cfg->rank; ++d) {
      shape[d] = cfg->dimensions[d].size;
      cs[d] = cfg->dimensions[d].chunk_size;
      cps[d] = cfg->dimensions[d].chunks_per_shard;
    }
    const uint8_t na = dims_n_append(cfg->dimensions, cfg->rank);
    struct shard_geometry g;
    shard_geometry_compute(&g, cfg->rank, na, shape, cs, cps);
    memcpy(zs->chunk_count, g.chunk_count, cfg->rank * sizeof(uint64_t));
    memcpy(
      zs->chunks_per_shard, g.chunks_per_shard, cfg->rank * sizeof(uint64_t));
    memcpy(zs->shard_count, g.shard_count, cfg->rank * sizeof(uint64_t));
    zs->shard_inner_count = g.shard_inner_count;
  }

  // Build array prefix key
  snprintf(zs->array_prefix,
           sizeof(zs->array_prefix),
           "%s/%s",
           cfg->prefix,
           cfg->array_name);

  // Write root + intermediate group metadata (skipped for sub-sinks)
  if (!skip_group_metadata) {
    char buf[256];
    int len = zarr_root_json(buf, sizeof(buf));
    CHECK(Fail_alloc, len >= 0);
    char key[4096];
    snprintf(key, sizeof(key), "%s/zarr.json", cfg->prefix);
    CHECK(Fail_alloc,
          s3_client_put(zs->s3, zs->bucket, key, buf, (size_t)len) == 0);
    struct s3_intermediate_ctx ictx = {
      .s3 = zs->s3,
      .bucket = zs->bucket,
      .prefix = cfg->prefix,
      .metadata = buf,
      .metadata_len = (size_t)len,
    };
    CHECK(Fail_alloc,
          zarr_for_each_intermediate(
            cfg->array_name, put_s3_intermediate, &ictx) == 0);
  }

  // Write array metadata
  {
    char buf[4096];
    int len = zarr_array_json(buf,
                              sizeof(buf),
                              zs->rank,
                              zs->dimensions,
                              zs->data_type,
                              zs->fill_value,
                              zs->chunks_per_shard,
                              zs->codec);
    CHECK(Fail_alloc, len >= 0);
    char key[4096];
    snprintf(key, sizeof(key), "%s/zarr.json", zs->array_prefix);
    CHECK(Fail_alloc,
          s3_client_put(zs->s3, zs->bucket, key, buf, (size_t)len) == 0);
  }

  // Allocate writer pool
  zs->num_writers = zs->shard_inner_count;
  zs->writers = (struct s3_shard_writer*)calloc(zs->num_writers,
                                                sizeof(struct s3_shard_writer));
  CHECK(Fail_alloc, zs->writers);

  for (uint64_t i = 0; i < zs->num_writers; ++i) {
    zs->writers[i].base.write = s3_shard_write;
    zs->writers[i].base.write_direct = NULL; // no zero-copy for S3
    zs->writers[i].base.finalize = s3_shard_finalize;
    zs->writers[i].parent = zs;
  }

  return zs;

Fail_alloc:
  free(zs);
Fail:
  return NULL;
}

struct zarr_s3_sink*
zarr_s3_sink_create(struct zarr_s3_config* cfg)
{
  zarr_s3_config_set_defaults(cfg);
  CHECK(Fail, zarr_s3_config_validate(cfg) == 0);

  struct s3_client_config s3cfg = {
    .region = cfg->region,
    .endpoint = cfg->endpoint,
    .part_size = cfg->part_size,
    .throughput_gbps = cfg->throughput_gbps,
    .max_retries = cfg->max_retries,
    .backoff_scale_ms = cfg->backoff_scale_ms,
    .max_backoff_secs = cfg->max_backoff_secs,
    .timeout_ns = cfg->timeout_ns,
  };
  struct s3_client* client = s3_client_create(&s3cfg);
  CHECK(Fail, client);

  struct zarr_s3_sink* zs = zarr_s3_sink_create_with_client(cfg, client, 0);
  CHECK(Fail_client, zs);

  zs->owns_s3 = 1;
  return zs;

Fail_client:
  s3_client_destroy(client);
Fail:
  return NULL;
}

int
zarr_s3_sink_has_error(const struct zarr_s3_sink* s)
{
  return s ? s->finalize_err : 0;
}

int
zarr_s3_sink_flush(struct zarr_s3_sink* s)
{
  if (!s)
    return 0;
  for (uint64_t i = 0; i < s->num_writers; ++i)
    s3_wait_pending(&s->writers[i]);
  return s->finalize_err;
}

int
zarr_s3_sink_destroy(struct zarr_s3_sink* s)
{
  if (!s)
    return 0;

  if (s->writers) {
    for (uint64_t i = 0; i < s->num_writers; ++i) {
      // Drain any pending upload
      s3_wait_pending(&s->writers[i]);
      // Abort any active (non-finalized) upload
      if (s->writers[i].upload) {
        s3_upload_abort(s->writers[i].upload);
        s->writers[i].upload = NULL;
      }
    }
    free(s->writers);
  }
  int err = s->finalize_err;
  if (s->owns_s3)
    s3_client_destroy(s->s3);
  free(s);
  return err;
}

struct shard_sink*
zarr_s3_sink_as_shard_sink(struct zarr_s3_sink* s)
{
  return &s->base;
}

// --- Multiscale sink ---

struct zarr_s3_multiscale_sink
{
  struct shard_sink base;
  struct zarr_s3_sink** levels;
  int nlod;

  // Shared S3 client (owned by this struct, borrowed by levels)
  struct s3_client* s3;
  char bucket[256];
  char group_prefix[4096];
  uint8_t rank;
};

static struct io_event
s3_multiscale_record_fence(struct shard_sink* self, uint8_t level)
{
  struct zarr_s3_multiscale_sink* ms = (struct zarr_s3_multiscale_sink*)self;
  if (level >= ms->nlod)
    return (struct io_event){ 0 };
  return s3_sink_record_fence(&ms->levels[level]->base, level);
}

static void
s3_multiscale_wait_fence(struct shard_sink* self,
                         uint8_t level,
                         struct io_event ev)
{
  struct zarr_s3_multiscale_sink* ms = (struct zarr_s3_multiscale_sink*)self;
  if (level >= ms->nlod)
    return;
  s3_sink_wait_fence(&ms->levels[level]->base, level, ev);
}

static struct shard_writer*
s3_multiscale_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct zarr_s3_multiscale_sink* ms = (struct zarr_s3_multiscale_sink*)self;
  CHECK(Fail, level < ms->nlod);
  return ms->levels[level]->base.open(
    &ms->levels[level]->base, level, shard_index);
Fail:
  return NULL;
}

static int
put_multiscale_group_metadata(const struct zarr_s3_multiscale_sink* ms)
{
  const struct dimension* level_ptrs[LOD_MAX_LEVELS];
  for (int lv = 0; lv < ms->nlod; ++lv)
    level_ptrs[lv] = ms->levels[lv]->dimensions;

  char buf[8192];
  int len = zarr_multiscale_group_json(
    buf, sizeof(buf), ms->rank, ms->nlod, level_ptrs);
  if (len < 0)
    return -1;

  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", ms->group_prefix);
  return s3_client_put(ms->s3, ms->bucket, key, buf, (size_t)len);
}

static int
s3_multiscale_update_append(struct shard_sink* self,
                            uint8_t level,
                            uint8_t n_append,
                            const uint64_t* append_sizes)
{
  struct zarr_s3_multiscale_sink* ms = (struct zarr_s3_multiscale_sink*)self;
  if (level >= ms->nlod)
    return 1;

  // Skip if dim 0 is unchanged. Only dim 0 can be unbounded (invariant
  // enforced by dims_n_append), so it is the only append dim whose size
  // changes at runtime. Bounded append dims (1..n_append-1) keep their
  // declared size from creation.
  uint64_t old = ms->levels[level]->dimensions[0].size;
  if (s3_sink_update_append(
        &ms->levels[level]->base, level, n_append, append_sizes))
    return 1;
  if (old == append_sizes[0])
    return 0;

  if (put_multiscale_group_metadata(ms)) {
    log_error(
      "s3_multiscale_update_append: failed to rewrite group zarr.json for %s",
      ms->group_prefix);
    return 1;
  }
  return 0;
}

struct zarr_s3_multiscale_sink*
zarr_s3_multiscale_sink_create(struct zarr_s3_multiscale_config* cfg)
{
  zarr_s3_multiscale_config_set_defaults(cfg);
  CHECK(Fail, zarr_s3_multiscale_config_validate(cfg) == 0);

  // Build group prefix
  char group_prefix[4096];
  if (cfg->array_name)
    snprintf(group_prefix,
             sizeof(group_prefix),
             "%s/%s",
             cfg->prefix,
             cfg->array_name);
  else
    snprintf(group_prefix, sizeof(group_prefix), "%s", cfg->prefix);

  struct lod_plan plan = { 0 };
  int max_lev = cfg->nlod > 0 ? cfg->nlod : LOD_MAX_LEVELS;
  CHECK(Fail,
        lod_plan_init_from_dims(&plan, cfg->dimensions, cfg->rank, max_lev) ==
          0);

  struct zarr_s3_multiscale_sink* ms =
    (struct zarr_s3_multiscale_sink*)calloc(1, sizeof(*ms));
  CHECK(Fail_plan, ms);

  ms->base.open = s3_multiscale_open;
  ms->base.update_append = s3_multiscale_update_append;
  ms->base.record_fence = s3_multiscale_record_fence;
  ms->base.wait_fence = s3_multiscale_wait_fence;
  ms->nlod = plan.nlod;
  ms->rank = cfg->rank;
  snprintf(ms->bucket, sizeof(ms->bucket), "%s", cfg->bucket);
  snprintf(ms->group_prefix, sizeof(ms->group_prefix), "%s", group_prefix);

  // Create shared S3 client
  {
    struct s3_client_config s3cfg = {
      .region = cfg->region,
      .endpoint = cfg->endpoint,
      .part_size = cfg->part_size,
      .throughput_gbps = cfg->throughput_gbps,
      .max_retries = cfg->max_retries,
      .backoff_scale_ms = cfg->backoff_scale_ms,
      .max_backoff_secs = cfg->max_backoff_secs,
      .timeout_ns = cfg->timeout_ns,
    };
    ms->s3 = s3_client_create(&s3cfg);
    CHECK(Fail_ms, ms->s3);
  }

  ms->levels = (struct zarr_s3_sink**)calloc((size_t)plan.nlod,
                                             sizeof(struct zarr_s3_sink*));
  CHECK(Fail_s3, ms->levels);

  // Write root + intermediate metadata before creating levels
  if (cfg->array_name) {
    char buf[256];
    int len = zarr_root_json(buf, sizeof(buf));
    CHECK(Fail_levels, len >= 0);
    char key[4096];
    snprintf(key, sizeof(key), "%s/zarr.json", cfg->prefix);
    CHECK(Fail_levels,
          s3_client_put(ms->s3, ms->bucket, key, buf, (size_t)len) == 0);
    struct s3_intermediate_ctx ictx = {
      .s3 = ms->s3,
      .bucket = ms->bucket,
      .prefix = cfg->prefix,
      .metadata = buf,
      .metadata_len = (size_t)len,
    };
    CHECK(Fail_levels,
          zarr_for_each_intermediate(
            cfg->array_name, put_s3_intermediate, &ictx) == 0);
  }

  // Create one zarr_s3_sink per level, all borrowing the shared client
  for (int lv = 0; lv < plan.nlod; ++lv) {
    struct dimension lv_dims[MAX_ZARR_RANK];
    for (int d = 0; d < cfg->rank; ++d) {
      lv_dims[d] = cfg->dimensions[d];
      if (d == 0 && cfg->dimensions[0].size == 0)
        lv_dims[d].size = 0;
      else
        lv_dims[d].size = plan.levels.shapes[lv][d];
      lv_dims[d].chunk_size = plan.levels.chunk_sizes[lv][d];
      lv_dims[d].chunks_per_shard = plan.levels.chunks_per_shard[lv][d];
    }

    char name[8];
    snprintf(name, sizeof(name), "%d", lv);

    struct zarr_s3_config zcfg = {
      .bucket = cfg->bucket,
      .prefix = group_prefix,
      .array_name = name,
      .region = cfg->region,
      .endpoint = cfg->endpoint,
      .data_type = cfg->data_type,
      .fill_value = cfg->fill_value,
      .rank = cfg->rank,
      .dimensions = lv_dims,
      .codec = cfg->codec,
    };

    ms->levels[lv] = zarr_s3_sink_create_with_client(&zcfg, ms->s3, 1);
    CHECK(Fail_levels, ms->levels[lv]);
  }

  // Write OME-NGFF group metadata
  CHECK(Fail_levels, put_multiscale_group_metadata(ms) == 0);

  lod_plan_free(&plan);
  return ms;

Fail_levels:
  for (int i = 0; i < plan.nlod; ++i) {
    if (ms->levels[i])
      zarr_s3_sink_destroy(ms->levels[i]);
  }
  free(ms->levels);
Fail_s3:
  s3_client_destroy(ms->s3);
Fail_ms:
  free(ms);
Fail_plan:
  lod_plan_free(&plan);
Fail:
  return NULL;
}

int
zarr_s3_multiscale_sink_flush(struct zarr_s3_multiscale_sink* s)
{
  if (!s)
    return 0;
  int err = 0;
  for (int i = 0; i < s->nlod; ++i)
    err |= zarr_s3_sink_flush(s->levels[i]);
  return err;
}

int
zarr_s3_multiscale_sink_destroy(struct zarr_s3_multiscale_sink* s)
{
  if (!s)
    return 0;
  int err = 0;
  for (int i = 0; i < s->nlod; ++i)
    err |= zarr_s3_sink_destroy(s->levels[i]);
  free(s->levels);
  s3_client_destroy(s->s3);
  free(s);
  return err;
}

struct shard_sink*
zarr_s3_multiscale_sink_as_shard_sink(struct zarr_s3_multiscale_sink* s)
{
  return &s->base;
}
