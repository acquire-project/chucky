#include "zarr_s3_sink.h"
#include "defs.limits.h"
#include "json_writer.h"
#include "lod_plan.h"
#include "prelude.h"
#include "s3_client.h"

#include <stdlib.h>
#include <string.h>

// --- S3 shard writer ---

struct s3_shard_writer
{
  struct shard_writer base;
  struct zarr_s3_sink* parent;
  struct s3_upload* upload; // active streaming upload, NULL when idle
};

static int
s3_shard_write(struct shard_writer* self,
               uint64_t offset,
               const void* beg,
               const void* end)
{
  (void)offset; // writes are sequential; CRT handles ordering
  struct s3_shard_writer* w = (struct s3_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;
  return s3_upload_write(w->upload, beg, nbytes);
}

static int
s3_shard_finalize(struct shard_writer* self)
{
  struct s3_shard_writer* w = (struct s3_shard_writer*)self;
  if (!w->upload)
    return 0;
  int rc = s3_upload_finish(w->upload);
  w->upload = NULL;
  return rc;
}

// --- Zarr S3 sink ---

struct zarr_s3_sink
{
  struct shard_sink base;
  struct s3_client* s3;
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

  // Metadata (for update_dim0)
  struct dimension dimensions[MAX_ZARR_RANK];
  enum zarr_dtype data_type;
  double fill_value;
  enum compression_codec codec;
};

// --- Key computation ---

static int
shard_key(char* buf,
          size_t cap,
          const char* array_prefix,
          uint8_t rank,
          const uint64_t* shard_count,
          uint64_t flat)
{
  int pos = snprintf(buf, cap, "%s/c", array_prefix);
  if (pos < 0 || (size_t)pos >= cap)
    return -1;

  uint64_t coords[MAX_ZARR_RANK];
  uint64_t rem = flat;
  for (int d = rank - 1; d >= 0; --d) {
    if (d == 0) {
      coords[d] = rem;
    } else {
      coords[d] = rem % shard_count[d];
      rem /= shard_count[d];
    }
  }

  for (int d = 0; d < rank; ++d) {
    int n = snprintf(
      buf + pos, cap - (size_t)pos, "/%llu", (unsigned long long)coords[d]);
    if (n < 0 || (size_t)(pos + n) >= cap)
      return -1;
    pos += n;
  }
  return 0;
}

// --- shard_sink vtable ---

static struct io_event
s3_sink_record_fence(struct shard_sink* self, uint8_t level)
{
  (void)self;
  (void)level;
  // CRT manages async I/O internally; no io_queue fence needed.
  // finalize() blocks until upload completes, providing natural backpressure.
  return (struct io_event){ 0 };
}

static void
s3_sink_wait_fence(struct shard_sink* self,
                   uint8_t level,
                   struct io_event ev)
{
  (void)self;
  (void)level;
  (void)ev;
}

static struct shard_writer*
s3_sink_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)self;

  uint64_t inner = shard_index % zs->shard_inner_count;
  struct s3_shard_writer* w = &zs->writers[inner];

  char key[4096];
  if (shard_key(key,
                sizeof(key),
                zs->array_prefix,
                zs->rank,
                zs->shard_count,
                shard_index) != 0) {
    log_error("s3_sink_open: key too long for shard %llu",
              (unsigned long long)shard_index);
    return NULL;
  }

  w->upload = s3_upload_begin(zs->s3, zs->bucket, key);
  if (!w->upload) {
    log_error("s3_sink_open: failed to begin upload for %s/%s", zs->bucket, key);
    return NULL;
  }

  return &w->base;
}

// --- Metadata ---

static const char*
zarr_dtype_string(enum zarr_dtype dt)
{
  switch (dt) {
    case zarr_dtype_uint8:
      return "uint8";
    case zarr_dtype_uint16:
      return "uint16";
    case zarr_dtype_uint32:
      return "uint32";
    case zarr_dtype_uint64:
      return "uint64";
    case zarr_dtype_int8:
      return "int8";
    case zarr_dtype_int16:
      return "int16";
    case zarr_dtype_int32:
      return "int32";
    case zarr_dtype_int64:
      return "int64";
    case zarr_dtype_float16:
      return "float16";
    case zarr_dtype_float32:
      return "float32";
    case zarr_dtype_float64:
      return "float64";
  }
  return "unknown";
}

static int
put_root_metadata(struct s3_client* s3,
                  const char* bucket,
                  const char* prefix)
{
  char buf[256];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "group");
  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;

  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", prefix);
  return s3_client_put(s3, bucket, key, buf, jw_length(&jw));
}

static int
put_array_metadata(struct s3_client* s3,
                   const char* bucket,
                   const char* array_prefix,
                   uint8_t rank,
                   const struct dimension* dimensions,
                   enum zarr_dtype data_type,
                   double fill_value,
                   const uint64_t* chunks_per_shard,
                   enum compression_codec codec)
{
  char buf[4096];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "array");

  jw_key(&jw, "shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].size);
  jw_array_end(&jw);

  jw_key(&jw, "data_type");
  jw_string(&jw, zarr_dtype_string(data_type));

  // chunk_grid: shard = chunk_size * chunks_per_shard
  jw_key(&jw, "chunk_grid");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "regular");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].chunk_size * chunks_per_shard[d]);
  jw_array_end(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  jw_key(&jw, "chunk_key_encoding");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "default");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "separator");
  jw_string(&jw, "/");
  jw_object_end(&jw);
  jw_object_end(&jw);

  // codecs: sharding_indexed
  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "sharding_indexed");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);

  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].chunk_size);
  jw_array_end(&jw);

  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  if (codec != CODEC_NONE) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    jw_string(&jw, codec == CODEC_LZ4 ? "lz4" : "zstd");
    jw_key(&jw, "configuration");
    jw_object_begin(&jw);
    jw_object_end(&jw);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "index_codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "crc32c");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_array_end(&jw);

  jw_key(&jw, "index_location");
  jw_string(&jw, "end");

  jw_object_end(&jw); // configuration
  jw_object_end(&jw); // sharding_indexed
  jw_array_end(&jw);  // codecs

  jw_key(&jw, "fill_value");
  jw_float(&jw, fill_value);

  // dimension_names
  {
    int has_names = 0;
    for (int d = 0; d < rank; ++d) {
      if (dimensions[d].name) {
        has_names = 1;
        break;
      }
    }
    if (has_names) {
      jw_key(&jw, "dimension_names");
      jw_array_begin(&jw);
      for (int d = 0; d < rank; ++d) {
        if (dimensions[d].name)
          jw_string(&jw, dimensions[d].name);
        else
          jw_null(&jw);
      }
      jw_array_end(&jw);
    }
  }

  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;

  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", array_prefix);
  return s3_client_put(s3, bucket, key, buf, jw_length(&jw));
}

// --- update_dim0 ---

static void
s3_sink_update_dim0(struct shard_sink* self,
                    uint8_t level,
                    uint64_t dim0_size)
{
  (void)level;
  struct zarr_s3_sink* zs = (struct zarr_s3_sink*)self;
  if (zs->dimensions[0].size == dim0_size)
    return;
  zs->dimensions[0].size = dim0_size;
  if (put_array_metadata(zs->s3,
                         zs->bucket,
                         zs->array_prefix,
                         zs->rank,
                         zs->dimensions,
                         zs->data_type,
                         zs->fill_value,
                         zs->chunks_per_shard,
                         zs->codec))
    log_error("s3_sink_update_dim0: failed to rewrite zarr.json for %s",
              zs->array_prefix);
}

// --- Create / Destroy ---

struct zarr_s3_sink*
zarr_s3_sink_create(const struct zarr_s3_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->bucket);
  CHECK(Fail, cfg->prefix);
  CHECK(Fail, cfg->array_name);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  struct zarr_s3_sink* zs =
    (struct zarr_s3_sink*)calloc(1, sizeof(struct zarr_s3_sink));
  CHECK(Fail, zs);

  zs->base.open = s3_sink_open;
  zs->base.update_dim0 = s3_sink_update_dim0;
  zs->base.record_fence = s3_sink_record_fence;
  zs->base.wait_fence = s3_sink_wait_fence;
  zs->rank = cfg->rank;
  zs->data_type = cfg->data_type;
  zs->fill_value = cfg->fill_value;
  zs->codec = cfg->codec;

  snprintf(zs->bucket, sizeof(zs->bucket), "%s", cfg->bucket);

  for (int d = 0; d < cfg->rank; ++d)
    zs->dimensions[d] = cfg->dimensions[d];

  // S3 client
  struct s3_client_config s3cfg = {
    .region = cfg->region,
    .endpoint = cfg->endpoint,
  };
  zs->s3 = s3_client_create(&s3cfg);
  CHECK(Fail_alloc, zs->s3);

  // Compute geometry
  zs->shard_inner_count = 1;
  for (int d = 0; d < cfg->rank; ++d) {
    zs->chunk_count[d] =
      (cfg->dimensions[d].size == 0)
        ? 1
        : ceildiv(cfg->dimensions[d].size, cfg->dimensions[d].chunk_size);
    uint64_t cps = cfg->dimensions[d].chunks_per_shard;
    zs->chunks_per_shard[d] = (cps == 0) ? zs->chunk_count[d] : cps;
    zs->shard_count[d] = ceildiv(zs->chunk_count[d], zs->chunks_per_shard[d]);
    if (d > 0)
      zs->shard_inner_count *= zs->shard_count[d];
  }

  // Build array prefix key
  snprintf(zs->array_prefix,
           sizeof(zs->array_prefix),
           "%s/%s",
           cfg->prefix,
           cfg->array_name);

  // Write metadata
  CHECK(Fail_s3, put_root_metadata(zs->s3, zs->bucket, cfg->prefix) == 0);
  CHECK(Fail_s3,
        put_array_metadata(zs->s3,
                           zs->bucket,
                           zs->array_prefix,
                           zs->rank,
                           zs->dimensions,
                           zs->data_type,
                           zs->fill_value,
                           zs->chunks_per_shard,
                           zs->codec) == 0);

  // Allocate writer pool
  zs->num_writers = zs->shard_inner_count;
  zs->writers = (struct s3_shard_writer*)calloc(zs->num_writers,
                                                sizeof(struct s3_shard_writer));
  CHECK(Fail_s3, zs->writers);

  for (uint64_t i = 0; i < zs->num_writers; ++i) {
    zs->writers[i].base.write = s3_shard_write;
    zs->writers[i].base.write_direct = NULL; // no zero-copy for S3
    zs->writers[i].base.finalize = s3_shard_finalize;
    zs->writers[i].parent = zs;
  }

  return zs;

Fail_s3:
  s3_client_destroy(zs->s3);
Fail_alloc:
  free(zs->writers);
  free(zs);
Fail:
  return NULL;
}

void
zarr_s3_sink_flush(struct zarr_s3_sink* s)
{
  if (!s)
    return;
  // All uploads complete synchronously in finalize().
  // Nothing to wait for here.
}

void
zarr_s3_sink_destroy(struct zarr_s3_sink* s)
{
  if (!s)
    return;

  zarr_s3_sink_flush(s);

  if (s->writers) {
    for (uint64_t i = 0; i < s->num_writers; ++i) {
      if (s->writers[i].upload)
        s3_upload_abort(s->writers[i].upload);
    }
    free(s->writers);
  }
  s3_client_destroy(s->s3);
  free(s);
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

  // For group metadata
  struct s3_client* s3; // shared ref (owned by levels[0])
  char bucket[256];
  char group_prefix[4096];
  uint8_t rank;
  struct dimension dimensions[MAX_ZARR_RANK];
};

static struct io_event
s3_multiscale_record_fence(struct shard_sink* self, uint8_t level)
{
  (void)self;
  (void)level;
  return (struct io_event){ 0 };
}

static void
s3_multiscale_wait_fence(struct shard_sink* self,
                         uint8_t level,
                         struct io_event ev)
{
  (void)self;
  (void)level;
  (void)ev;
}

static struct shard_writer*
s3_multiscale_open(struct shard_sink* self,
                   uint8_t level,
                   uint64_t shard_index)
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
  char buf[8192];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "group");

  jw_key(&jw, "attributes");
  jw_object_begin(&jw);

  jw_key(&jw, "ome");
  jw_object_begin(&jw);
  jw_key(&jw, "version");
  jw_string(&jw, "0.5");

  jw_key(&jw, "multiscales");
  jw_array_begin(&jw);
  jw_object_begin(&jw);

  jw_key(&jw, "axes");
  jw_array_begin(&jw);
  for (int d = 0; d < ms->rank; ++d) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    if (ms->dimensions[d].name)
      jw_string(&jw, ms->dimensions[d].name);
    else {
      char name[8];
      snprintf(name, sizeof(name), "d%d", d);
      jw_string(&jw, name);
    }
    jw_key(&jw, "type");
    {
      const char* n = ms->dimensions[d].name;
      const char* type = "space";
      if (n && (n[0] == 't' || n[0] == 'T') && n[1] == '\0')
        type = "time";
      else if (n && (n[0] == 'c' || n[0] == 'C') && n[1] == '\0')
        type = "channel";
      jw_string(&jw, type);
    }
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "datasets");
  jw_array_begin(&jw);
  for (int lv = 0; lv < ms->nlod; ++lv) {
    jw_object_begin(&jw);
    jw_key(&jw, "path");
    char lvstr[8];
    snprintf(lvstr, sizeof(lvstr), "%d", lv);
    jw_string(&jw, lvstr);

    jw_key(&jw, "coordinateTransformations");
    jw_array_begin(&jw);
    // scale
    jw_object_begin(&jw);
    jw_key(&jw, "type");
    jw_string(&jw, "scale");
    jw_key(&jw, "scale");
    jw_array_begin(&jw);
    for (int d = 0; d < ms->rank; ++d) {
      double scale = 1.0;
      if (ms->dimensions[d].downsample &&
          ms->levels[lv]->dimensions[d].size > 0) {
        if (ms->dimensions[d].size == 0)
          scale = (double)(1u << lv);
        else
          scale = (double)ms->dimensions[d].size /
                  (double)ms->levels[lv]->dimensions[d].size;
      }
      jw_float(&jw, scale);
    }
    jw_array_end(&jw);
    jw_object_end(&jw);
    // translation
    jw_object_begin(&jw);
    jw_key(&jw, "type");
    jw_string(&jw, "translation");
    jw_key(&jw, "translation");
    jw_array_begin(&jw);
    for (int d = 0; d < ms->rank; ++d) {
      double t = 0.0;
      if (ms->dimensions[d].downsample &&
          ms->levels[lv]->dimensions[d].size > 0) {
        double factor;
        if (ms->dimensions[d].size == 0)
          factor = (double)(1u << lv);
        else
          factor = (double)ms->dimensions[d].size /
                   (double)ms->levels[lv]->dimensions[d].size;
        t = 0.5 * (factor - 1.0);
      }
      jw_float(&jw, t);
    }
    jw_array_end(&jw);
    jw_object_end(&jw);
    jw_array_end(&jw);

    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_object_end(&jw); // multiscales[0]
  jw_array_end(&jw);  // multiscales

  jw_object_end(&jw); // ome
  jw_object_end(&jw); // attributes
  jw_object_end(&jw); // root

  if (jw_error(&jw))
    return -1;

  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", ms->group_prefix);
  return s3_client_put(ms->s3, ms->bucket, key, buf, jw_length(&jw));
}

static void
s3_multiscale_update_dim0(struct shard_sink* self,
                          uint8_t level,
                          uint64_t dim0_size)
{
  struct zarr_s3_multiscale_sink* ms = (struct zarr_s3_multiscale_sink*)self;
  if (level >= ms->nlod)
    return;

  uint64_t old = ms->levels[level]->dimensions[0].size;
  s3_sink_update_dim0(&ms->levels[level]->base, level, dim0_size);
  if (old == dim0_size)
    return;

  if (put_multiscale_group_metadata(ms))
    log_error(
      "s3_multiscale_update_dim0: failed to rewrite group zarr.json for %s",
      ms->group_prefix);
}

struct zarr_s3_multiscale_sink*
zarr_s3_multiscale_sink_create(const struct zarr_s3_multiscale_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->bucket);
  CHECK(Fail, cfg->prefix);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

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

  // LOD plan
  struct lod_plan plan = { 0 };
  uint64_t shape[MAX_ZARR_RANK];
  uint64_t chunk_shape[MAX_ZARR_RANK];
  for (int d = 0; d < cfg->rank; ++d) {
    shape[d] = (cfg->dimensions[d].size == 0) ? cfg->dimensions[d].chunk_size
                                               : cfg->dimensions[d].size;
    chunk_shape[d] = cfg->dimensions[d].chunk_size;
  }

  uint32_t lod_mask = 0;
  for (int d = 0; d < cfg->rank; ++d)
    if (cfg->dimensions[d].downsample)
      lod_mask |= (1u << d);
  if (cfg->rank > 0 && cfg->dimensions[0].size == 0)
    lod_mask &= ~1u;

  int max_lev = cfg->nlod > 0 ? cfg->nlod : LOD_MAX_LEVELS;
  CHECK(Fail,
        lod_plan_init_shapes(
          &plan, cfg->rank, shape, chunk_shape, lod_mask, max_lev) == 0);

  struct zarr_s3_multiscale_sink* ms =
    (struct zarr_s3_multiscale_sink*)calloc(1, sizeof(*ms));
  CHECK(Fail_plan, ms);

  ms->base.open = s3_multiscale_open;
  ms->base.update_dim0 = s3_multiscale_update_dim0;
  ms->base.record_fence = s3_multiscale_record_fence;
  ms->base.wait_fence = s3_multiscale_wait_fence;
  ms->nlod = plan.nlod;
  ms->rank = cfg->rank;
  snprintf(ms->bucket, sizeof(ms->bucket), "%s", cfg->bucket);
  snprintf(ms->group_prefix, sizeof(ms->group_prefix), "%s", group_prefix);
  for (int d = 0; d < cfg->rank; ++d)
    ms->dimensions[d] = cfg->dimensions[d];

  ms->levels = (struct zarr_s3_sink**)calloc((size_t)plan.nlod,
                                              sizeof(struct zarr_s3_sink*));
  CHECK(Fail_ms, ms->levels);

  // Create one zarr_s3_sink per level
  for (int lv = 0; lv < plan.nlod; ++lv) {
    struct dimension lv_dims[MAX_ZARR_RANK];
    for (int d = 0; d < cfg->rank; ++d) {
      lv_dims[d] = cfg->dimensions[d];
      if (d == 0 && cfg->dimensions[0].size == 0)
        lv_dims[d].size = 0;
      else
        lv_dims[d].size = plan.shapes[lv][d];
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

    ms->levels[lv] = zarr_s3_sink_create(&zcfg);
    CHECK(Fail_levels, ms->levels[lv]);
  }

  // Use the first level's s3 client for group metadata
  ms->s3 = ms->levels[0]->s3;

  // Write root metadata (at prefix level, above group)
  if (cfg->array_name) {
    CHECK(Fail_levels,
          put_root_metadata(ms->s3, ms->bucket, cfg->prefix) == 0);
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
Fail_ms:
  free(ms);
Fail_plan:
  lod_plan_free(&plan);
Fail:
  return NULL;
}

void
zarr_s3_multiscale_sink_flush(struct zarr_s3_multiscale_sink* s)
{
  if (!s)
    return;
  for (int i = 0; i < s->nlod; ++i)
    zarr_s3_sink_flush(s->levels[i]);
}

void
zarr_s3_multiscale_sink_destroy(struct zarr_s3_multiscale_sink* s)
{
  if (!s)
    return;
  for (int i = 0; i < s->nlod; ++i)
    zarr_s3_sink_destroy(s->levels[i]);
  free(s->levels);
  free(s);
}

struct shard_sink*
zarr_s3_multiscale_sink_as_shard_sink(struct zarr_s3_multiscale_sink* s)
{
  return &s->base;
}
