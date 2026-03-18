#include "defs.limits.h"
#include "io_queue.h"
#include "json_writer.h"
#include "lod_plan.h"
#include "platform.h"
#include "platform_io.h"
#include "prelude.h"
#include "zarr_sink.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// --- Writer for a single shard file ---

struct zarr_shard_writer
{
  struct shard_writer base;
  platform_fd fd;
  struct io_queue* queue;
  size_t alignment; // 0 = normal malloc, >0 = page-aligned allocation
  _Atomic uint64_t* retired_bytes; // points to zarr_sink.retired_bytes
  uint64_t* queued_bytes;          // points to zarr_sink.queued_bytes
  _Atomic int* io_error;           // points to zarr_sink.io_error
};

struct pwrite_job
{
  platform_fd fd;
  uint64_t offset;
  size_t nbytes;
  size_t data_off;                 // byte offset from start of struct to data
  _Atomic uint64_t* retired_bytes; // written by io worker after pwrite
  _Atomic int* io_error;           // set on write failure
  uint8_t data[]; // used when data_off == sizeof(struct pwrite_job)
};

static void
pwrite_fn(void* arg)
{
  struct pwrite_job* j = (struct pwrite_job*)arg;
  const void* data = (const char*)j + j->data_off;
  if (platform_pwrite(j->fd, data, j->nbytes, j->offset) != 0) {
    log_error("zarr pwrite failed");
    atomic_store(j->io_error, 1);
  }
  atomic_fetch_add(j->retired_bytes, j->nbytes);
}

static int
zarr_shard_write(struct shard_writer* self,
                 uint64_t offset,
                 const void* beg,
                 const void* end)
{
  struct zarr_shard_writer* w = (struct zarr_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);

  if (w->queue) {
    struct pwrite_job* j;
    void (*job_free)(void*) = free;
    if (w->alignment > 0) {
      // Unbuffered IO: buffer must be page-aligned.
      size_t hdr = align_up(sizeof(struct pwrite_job), w->alignment);
      j =
        (struct pwrite_job*)platform_aligned_alloc(w->alignment, hdr + nbytes);
      CHECK(Error, j);
      j->data_off = hdr; // data lives at aligned offset
      job_free = platform_aligned_free;
    } else {
      j = (struct pwrite_job*)malloc(sizeof(struct pwrite_job) + nbytes);
      CHECK(Error, j);
      j->data_off = sizeof(struct pwrite_job);
    }
    j->fd = w->fd;
    j->offset = offset;
    j->nbytes = nbytes;
    j->retired_bytes = w->retired_bytes;
    j->io_error = w->io_error;
    memcpy((char*)j + j->data_off, beg, nbytes);
    if (io_queue_post(w->queue, pwrite_fn, j, job_free)) {
      job_free(j);
      goto Error;
    }
    *w->queued_bytes += nbytes;
  } else {
    CHECK(Error, platform_pwrite(w->fd, beg, nbytes, offset) == 0);
  }
  return 0;

Error:
  return 1;
}

// Zero-copy pwrite: data points into pinned memory, NOT owned.
struct pwrite_ref_job
{
  platform_fd fd;
  uint64_t offset;
  size_t nbytes;
  const void* data;                // NOT owned — points into pinned memory
  _Atomic uint64_t* retired_bytes; // written by io worker after pwrite
  _Atomic int* io_error;           // set on write failure
};

static void
pwrite_ref_fn(void* arg)
{
  struct pwrite_ref_job* j = (struct pwrite_ref_job*)arg;
  if (platform_pwrite(j->fd, j->data, j->nbytes, j->offset) != 0) {
    log_error("zarr pwrite_ref failed");
    atomic_store(j->io_error, 1);
  }
  atomic_fetch_add(j->retired_bytes, j->nbytes);
}

static int
zarr_shard_write_direct(struct shard_writer* self,
                        uint64_t offset,
                        const void* beg,
                        const void* end)
{
  struct zarr_shard_writer* w = (struct zarr_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;

  if (w->queue) {
    struct pwrite_ref_job* j =
      (struct pwrite_ref_job*)malloc(sizeof(struct pwrite_ref_job));
    CHECK(Error, j);
    j->fd = w->fd;
    j->offset = offset;
    j->nbytes = nbytes;
    j->data = beg;
    j->retired_bytes = w->retired_bytes;
    j->io_error = w->io_error;
    if (io_queue_post(w->queue, pwrite_ref_fn, j, free)) {
      free(j);
      goto Error;
    }
    *w->queued_bytes += nbytes;
  } else {
    CHECK(Error, platform_pwrite(w->fd, beg, nbytes, offset) == 0);
  }
  return 0;

Error:
  return 1;
}

struct close_job
{
  platform_fd fd;
};

static void
close_fn(void* arg)
{
  struct close_job* j = (struct close_job*)arg;
  platform_close(j->fd);
}

static int
zarr_shard_finalize(struct shard_writer* self)
{
  struct zarr_shard_writer* w = (struct zarr_shard_writer*)self;
  if (w->fd == PLATFORM_FD_INVALID)
    return 0;

  if (w->queue) {
    struct close_job* j = (struct close_job*)malloc(sizeof(struct close_job));
    CHECK(Error, j);
    j->fd = w->fd;
    if (io_queue_post(w->queue, close_fn, j, free)) {
      free(j);
      goto Error;
    }
  } else {
    platform_close(w->fd);
  }

  w->fd = PLATFORM_FD_INVALID;
  return 0;

Error:
  return 1;
}

// --- Zarr sink ---

struct zarr_sink
{
  struct shard_sink base;
  struct io_queue* queue;
  int unbuffered;
  uint64_t queued_bytes;          // written by main thread only
  _Atomic uint64_t retired_bytes; // written by io worker (atomic)
  _Atomic int io_error;           // set by io worker on write failure

  // Geometry
  uint8_t rank;
  uint64_t chunk_count[MAX_ZARR_RANK];
  uint64_t chunks_per_shard[MAX_ZARR_RANK];
  uint64_t shard_count[MAX_ZARR_RANK];
  uint64_t shard_inner_count; // prod(shard_count[d] for d > 0)

  // Paths
  char array_dir[4096]; // "{store_path}/{array_name}"

  // Writer pool: one per inner shard, reused across shard epochs
  struct zarr_shard_writer* writers;
  uint64_t num_writers;

  // For metadata updates (stored copy of config)
  struct dimension dimensions[MAX_ZARR_RANK];
  enum zarr_dtype data_type;
  double fill_value;
  enum compression_codec codec;
};

// --- Path computation ---

static int
shard_path(char* buf,
           size_t cap,
           const char* array_dir,
           uint8_t rank,
           const uint64_t* shard_count,
           uint64_t flat)
{
  int pos = snprintf(buf, cap, "%s/c", array_dir);
  if (pos < 0 || (size_t)pos >= cap)
    return -1;

  // Unravel flat index into coordinates (row-major).
  // Dim 0 coordinate is just the remainder after extracting inner dims,
  // which works for both bounded and unbounded (shard_count[0] unknown).
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

// --- shard_sink fence ---

static struct io_event
zarr_sink_record_fence(struct shard_sink* self, uint8_t level)
{
  (void)level;
  struct zarr_sink* zs = (struct zarr_sink*)self;
  return io_queue_record(zs->queue);
}

static void
zarr_sink_wait_fence(struct shard_sink* self, uint8_t level, struct io_event ev)
{
  (void)level;
  struct zarr_sink* zs = (struct zarr_sink*)self;
  io_event_wait(zs->queue, ev);
}

// --- shard_sink open ---

static struct shard_writer*
zarr_sink_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct zarr_sink* zs = (struct zarr_sink*)self;

  // Map flat shard_index to inner writer index
  // flat = s0 * shard_inner_count + inner_flat
  uint64_t inner = shard_index % zs->shard_inner_count;
  struct zarr_shard_writer* w = &zs->writers[inner];

  // Build path
  char path[4096];
  if (shard_path(path,
                 sizeof(path),
                 zs->array_dir,
                 zs->rank,
                 zs->shard_count,
                 shard_index) != 0) {
    log_error("zarr_sink_open: path too long for shard %llu",
              (unsigned long long)shard_index);
    return NULL;
  }

  int flags = zs->unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0;
  w->fd = platform_open_write(path, flags);
  if (w->fd == PLATFORM_FD_INVALID) {
    // Directory may not exist yet (unbounded dim0) — create and retry
    char dir[4096];
    size_t pathlen = strlen(path);
    memcpy(dir, path, pathlen + 1);
    char* last_slash = strrchr(dir, '/');
    if (last_slash) {
      *last_slash = '\0';
      if (platform_mkdirp(dir) == 0)
        w->fd = platform_open_write(path, flags);
    }
    if (w->fd == PLATFORM_FD_INVALID) {
      log_error("zarr_sink_open: open(%s) failed", path);
      return NULL;
    }
  }

  return &w->base;
}

// --- Metadata writing ---

static int
write_file(const char* path, const char* data, size_t len)
{
  platform_fd fd = platform_open_write(path, 0);
  if (fd == PLATFORM_FD_INVALID)
    return -1;
  int rc = platform_write(fd, data, len);
  platform_close(fd);
  return rc;
}

static int
write_root_metadata(const char* store_path)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", store_path);

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
  return write_file(path, buf, jw_length(&jw));
}

static const char*
zarr_dtype_string(enum zarr_dtype dt)
{
#define CAT(x, y) x##y
#define CASE(T)                                                                \
  case CAT(zarr_dtype_, T):                                                    \
    return #T

  switch (dt) {
    CASE(uint8);
    CASE(uint16);
    CASE(uint32);
    CASE(uint64);
    CASE(int8);
    CASE(int16);
    CASE(int32);
    CASE(int64);
    CASE(float16);
    CASE(float32);
    CASE(float64);
  }
#undef CAT
#undef CASE
  return "unknown";
}

static int
write_array_metadata(const char* array_dir,
                     uint8_t rank,
                     const struct dimension* dimensions,
                     enum zarr_dtype data_type,
                     double fill_value,
                     const uint64_t* chunks_per_shard,
                     enum compression_codec codec)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", array_dir);

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

  // chunk_grid (zarr spec): each zarr outer "chunk" is a shard = chunk_size *
  // chunks_per_shard
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

  // chunk_shape (zarr spec): inner chunk = chunk shape
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
  jw_object_end(&jw); // sharding_indexed codec
  jw_array_end(&jw);  // codecs

  jw_key(&jw, "fill_value");
  jw_float(&jw, fill_value);

  // dimension_names (optional)
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
  return write_file(path, buf, jw_length(&jw));
}

// --- Metadata update ---

static void
zarr_sink_update_dim0(struct shard_sink* self,
                      uint8_t level,
                      uint64_t dim0_size)
{
  (void)level;
  struct zarr_sink* zs = (struct zarr_sink*)self;
  if (zs->dimensions[0].size == dim0_size)
    return;
  zs->dimensions[0].size = dim0_size;
  if (write_array_metadata(zs->array_dir,
                           zs->rank,
                           zs->dimensions,
                           zs->data_type,
                           zs->fill_value,
                           zs->chunks_per_shard,
                           zs->codec))
    log_error("zarr_sink_update_dim0: failed to rewrite zarr.json for %s",
              zs->array_dir);
}

// --- Create / Destroy ---

struct zarr_sink*
zarr_sink_create(const struct zarr_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->store_path);
  CHECK(Fail, cfg->array_name);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  struct zarr_sink* zs = (struct zarr_sink*)calloc(1, sizeof(*zs));
  CHECK(Fail, zs);

  zs->base.open = zarr_sink_open;
  zs->base.update_dim0 = zarr_sink_update_dim0;
  zs->base.record_fence = zarr_sink_record_fence;
  zs->base.wait_fence = zarr_sink_wait_fence;
  zs->queue = io_queue_create();
  CHECK(Fail_alloc, zs->queue);
  zs->unbuffered = cfg->unbuffered;
  zs->rank = cfg->rank;
  zs->data_type = cfg->data_type;
  zs->fill_value = cfg->fill_value;
  zs->codec = cfg->codec;

  // Store dimension config for metadata updates
  for (int d = 0; d < cfg->rank; ++d)
    zs->dimensions[d] = cfg->dimensions[d];

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

  // Build array directory path
  snprintf(zs->array_dir,
           sizeof(zs->array_dir),
           "%s/%s",
           cfg->store_path,
           cfg->array_name);

  // Create directory tree: ensure shard directories exist.
  // When dim0 is unbounded (size=0), only pre-create inner dirs for
  // shard_epoch=0; further dirs are created on-demand in zarr_sink_open.
  {
    uint64_t total_shards =
      zs->shard_inner_count; // just inner dims for epoch 0
    if (cfg->dimensions[0].size > 0)
      total_shards *= zs->shard_count[0]; // bounded: create all

    for (uint64_t flat = 0; flat < total_shards; ++flat) {
      char path[4096];
      if (shard_path(path,
                     sizeof(path),
                     zs->array_dir,
                     zs->rank,
                     zs->shard_count,
                     flat) != 0)
        goto Fail_alloc;

      // Get the directory portion (everything up to last '/')
      char* last_slash = strrchr(path, '/');
      if (last_slash) {
        *last_slash = '\0';
        if (platform_mkdirp(path) != 0) {
          log_error("zarr_sink: platform_mkdirp(%s) failed", path);
          goto Fail_alloc;
        }
      }
    }
  }

  // Write metadata
  CHECK(Fail_alloc, write_root_metadata(cfg->store_path) == 0);
  CHECK(Fail_alloc,
        write_array_metadata(zs->array_dir,
                             zs->rank,
                             zs->dimensions,
                             zs->data_type,
                             zs->fill_value,
                             zs->chunks_per_shard,
                             zs->codec) == 0);

  // Allocate writer pool
  zs->num_writers = zs->shard_inner_count;
  zs->writers = (struct zarr_shard_writer*)calloc(
    zs->num_writers, sizeof(struct zarr_shard_writer));
  CHECK(Fail_alloc, zs->writers);

  for (uint64_t i = 0; i < zs->num_writers; ++i) {
    zs->writers[i].base.write = zarr_shard_write;
    zs->writers[i].base.write_direct = zarr_shard_write_direct;
    zs->writers[i].base.finalize = zarr_shard_finalize;
    zs->writers[i].fd = PLATFORM_FD_INVALID;
    zs->writers[i].queue = zs->queue;
    zs->writers[i].alignment = zs->unbuffered ? platform_page_size() : 0;
    zs->writers[i].retired_bytes = &zs->retired_bytes;
    zs->writers[i].queued_bytes = &zs->queued_bytes;
    zs->writers[i].io_error = &zs->io_error;
  }

  return zs;

Fail_alloc:
  free(zs->writers);
  io_queue_destroy(zs->queue);
  free(zs);
Fail:
  return NULL;
}

size_t
zarr_sink_pending_bytes(struct zarr_sink* s)
{
  if (!s || !s->queue)
    return 0;
  return (size_t)(s->queued_bytes - atomic_load(&s->retired_bytes));
}

void
zarr_sink_flush(struct zarr_sink* s)
{
  if (!s || !s->queue)
    return;
  struct io_event ev = io_queue_record(s->queue);
  io_event_wait(s->queue, ev);
}

void
zarr_sink_destroy(struct zarr_sink* s)
{
  if (!s)
    return;

  zarr_sink_flush(s);

  if (s->writers) {
    for (uint64_t i = 0; i < s->num_writers; ++i) {
      if (s->writers[i].fd != PLATFORM_FD_INVALID)
        platform_close(s->writers[i].fd);
    }
    free(s->writers);
  }
  io_queue_destroy(s->queue);
  free(s);
}

struct shard_sink*
zarr_sink_as_shard_sink(struct zarr_sink* s)
{
  return &s->base;
}

// --- Multiscale sink ---

struct zarr_multiscale_sink
{
  struct shard_sink base;
  struct zarr_sink** levels; // array of nlod zarr_sink*
  int nlod;

  // For group metadata regeneration
  char group_path[4096];
  uint8_t rank;
  struct dimension
    dimensions[MAX_ZARR_RANK]; // L0 dims (names, downsample flags)
};

static struct io_event
zarr_multiscale_record_fence(struct shard_sink* self, uint8_t level)
{
  struct zarr_multiscale_sink* ms = (struct zarr_multiscale_sink*)self;
  if (level >= ms->nlod)
    return (struct io_event){ 0 };
  return io_queue_record(ms->levels[level]->queue);
}

static void
zarr_multiscale_wait_fence(struct shard_sink* self,
                           uint8_t level,
                           struct io_event ev)
{
  struct zarr_multiscale_sink* ms = (struct zarr_multiscale_sink*)self;
  if (level < ms->nlod)
    io_event_wait(ms->levels[level]->queue, ev);
}

static struct shard_writer*
zarr_multiscale_open(struct shard_sink* self,
                     uint8_t level,
                     uint64_t shard_index)
{
  struct zarr_multiscale_sink* ms = (struct zarr_multiscale_sink*)self;
  CHECK(Fail, level < ms->nlod);
  return ms->levels[level]->base.open(
    &ms->levels[level]->base, level, shard_index);
Fail:
  return NULL;
}

// Regenerate OME-NGFF group metadata from live per-level shapes.
static int
write_multiscale_group_metadata(const struct zarr_multiscale_sink* ms)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", ms->group_path);

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
      const char* type;
      switch (ms->dimensions[d].axis_type) {
        case dimension_axis_time: type = "time"; break;
        case dimension_axis_channel: type = "channel"; break;
        case dimension_axis_other: type = "custom"; break;
        default: type = "space"; break;
      }
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
  return write_file(path, buf, jw_length(&jw));
}

static void
zarr_multiscale_update_dim0(struct shard_sink* self,
                            uint8_t level,
                            uint64_t dim0_size)
{
  struct zarr_multiscale_sink* ms = (struct zarr_multiscale_sink*)self;
  if (level >= ms->nlod)
    return;

  // Skip if unchanged
  uint64_t old = ms->levels[level]->dimensions[0].size;
  zarr_sink_update_dim0(&ms->levels[level]->base, level, dim0_size);
  if (old == dim0_size)
    return;

  // Regenerate group metadata
  if (write_multiscale_group_metadata(ms))
    log_error(
      "zarr_multiscale_update_dim0: failed to rewrite group zarr.json for %s",
      ms->group_path);
}

struct zarr_multiscale_sink*
zarr_multiscale_sink_create(const struct zarr_multiscale_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->store_path);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  // Build group path: store_path/array_name when array_name is set
  char group_path[4096];
  if (cfg->array_name)
    snprintf(group_path,
             sizeof(group_path),
             "%s/%s",
             cfg->store_path,
             cfg->array_name);
  else
    snprintf(group_path, sizeof(group_path), "%s", cfg->store_path);

  // Compute LOD plan for shapes.
  // When dim0 is unbounded (size=0), use chunk_size as a placeholder
  // to get valid inner shapes. Dim0 shape will be updated dynamically.
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

  struct zarr_multiscale_sink* ms =
    (struct zarr_multiscale_sink*)calloc(1, sizeof(*ms));
  CHECK(Fail_plan, ms);

  ms->base.open = zarr_multiscale_open;
  ms->base.update_dim0 = zarr_multiscale_update_dim0;
  ms->base.record_fence = zarr_multiscale_record_fence;
  ms->base.wait_fence = zarr_multiscale_wait_fence;
  ms->nlod = plan.nlod;
  ms->rank = cfg->rank;
  snprintf(ms->group_path, sizeof(ms->group_path), "%s", group_path);
  for (int d = 0; d < cfg->rank; ++d)
    ms->dimensions[d] = cfg->dimensions[d];

  ms->levels =
    (struct zarr_sink**)calloc((size_t)plan.nlod, sizeof(struct zarr_sink*));
  CHECK(Fail_ms, ms->levels);

  // Ensure directories exist before writing metadata
  if (cfg->array_name) {
    CHECK(Fail_ms, platform_mkdirp(cfg->store_path) == 0);
    CHECK(Fail_ms, platform_mkdirp(group_path) == 0);
    CHECK(Fail_ms, write_root_metadata(cfg->store_path) == 0);
  } else {
    CHECK(Fail_ms, platform_mkdirp(group_path) == 0);
  }

  // Create one zarr_sink per level
  for (int lv = 0; lv < plan.nlod; ++lv) {
    // Build per-level dimensions with downsampled sizes.
    // When dim0 is unbounded (size=0), set level shape[0]=0 (will grow).
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

    struct zarr_config zcfg = {
      .store_path = group_path,
      .array_name = name,
      .data_type = cfg->data_type,
      .fill_value = cfg->fill_value,
      .rank = cfg->rank,
      .dimensions = lv_dims,
      .unbuffered = cfg->unbuffered,
      .codec = cfg->codec,
    };

    ms->levels[lv] = zarr_sink_create(&zcfg);
    CHECK(Fail_levels, ms->levels[lv]);
  }

  // Write OME-NGFF multiscales group metadata (overwrites group zarr.json)
  CHECK(Fail_levels, write_multiscale_group_metadata(ms) == 0);

  lod_plan_free(&plan);
  return ms;

Fail_levels:
  for (int i = 0; i < plan.nlod; ++i) {
    if (ms->levels[i])
      zarr_sink_destroy(ms->levels[i]);
  }
  free(ms->levels);
Fail_ms:
  free(ms);
Fail_plan:
  lod_plan_free(&plan);
Fail:
  return NULL;
}

size_t
zarr_multiscale_sink_pending_bytes(struct zarr_multiscale_sink* s)
{
  if (!s)
    return 0;
  size_t total = 0;
  for (int i = 0; i < s->nlod; ++i)
    total += zarr_sink_pending_bytes(s->levels[i]);
  return total;
}

void
zarr_multiscale_sink_flush(struct zarr_multiscale_sink* s)
{
  if (!s)
    return;
  for (int i = 0; i < s->nlod; ++i)
    zarr_sink_flush(s->levels[i]);
}

void
zarr_multiscale_sink_destroy(struct zarr_multiscale_sink* s)
{
  if (!s)
    return;
  for (int i = 0; i < s->nlod; ++i)
    zarr_sink_destroy(s->levels[i]);
  free(s->levels);
  free(s);
}

struct shard_sink*
zarr_multiscale_sink_as_shard_sink(struct zarr_multiscale_sink* s)
{
  return &s->base;
}
