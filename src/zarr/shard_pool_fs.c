#include "zarr/shard_pool_fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "stream/host_output_pool.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/filesystem_write.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_scheduler.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// --- Pool ---

struct fs_slot;

struct prepared_file
{
  char* path;
  unsigned char* created_dirs;
  platform_fd fd;
  uint64_t capacity;
  struct io_event ready;
  int owned;
};

struct shard_pool_fs
{
  struct shard_pool base;
  struct io_backend_fs* backend;
  struct io_scheduler* queue;
  struct io_scheduler* prepare_queue;
  struct io_backend prepare_backend;
  struct prepared_file* prepared;
  struct fs_slot* slots;
  uint64_t nslots;
  int unbuffered;
  struct strbuf root; // owned
  _Atomic int io_error;
};

// --- Writer slot for a single shard file ---

struct fs_slot
{
  struct shard_writer base;
  struct io_file_token token; // zero generation means no file is open here
  struct io_scheduler* queue;
  size_t alignment; // 0 = normal malloc, >0 = page-aligned allocation
  int presize;      // set the file's size up front
  uint64_t prepared_capacity;
};

static int
fs_slot_write(struct shard_writer* self,
              uint64_t offset,
              const void* beg,
              const void* end)
{
  struct fs_slot* w = (struct fs_slot*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;

  // Debug-build watchdog: under O_DIRECT, length and offset must both be
  // multiples of the device alignment (source pointer alignment is handled
  // by the aligned_alloc + memcpy below).
  CHECK(Error, w->alignment == 0 || nbytes % w->alignment == 0);
  CHECK(Error, w->alignment == 0 || offset % w->alignment == 0);

  struct io_request req = {
    .op = IO_OP_WRITE,
    .file = w->token,
    .nbytes = nbytes,
    .offset = offset,
  };

  void* buf;
  void (*buf_free)(void*);
  if (w->alignment > 0) {
    buf = platform_aligned_alloc(w->alignment, nbytes);
    buf_free = platform_aligned_free;
  } else {
    buf = malloc(nbytes);
    buf_free = free;
  }
  CHECK(Error, buf);
  memcpy(buf, beg, nbytes);

  req.payload = buf;
  req.owned = buf;
  req.owned_free = buf_free;
  if (io_scheduler_post(w->queue, req)) {
    buf_free(buf);
    goto Error;
  }
  return 0;

Error:
  return 1;
}

// The payload points into pinned memory the caller keeps alive.
static int
fs_slot_write_direct(struct shard_writer* self,
                     uint64_t offset,
                     const void* beg,
                     const void* end)
{
  struct fs_slot* w = (struct fs_slot*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;

  return io_scheduler_post(w->queue,
                           (struct io_request){
                             .op = IO_OP_WRITE,
                             .file = w->token,
                             .payload = beg,
                             .nbytes = nbytes,
                             .offset = offset,
                           });
}

static void
output_write_finished(void* ctx)
{
  host_output_group_complete((struct host_output_group*)ctx);
}

static int
fs_slot_write_from_output(struct shard_writer* self,
                          uint64_t offset,
                          const void* beg,
                          const void* end,
                          struct host_output_group* group)
{
  struct fs_slot* w = (struct fs_slot*)self;
  const uint64_t nbytes = (uint64_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;

  const struct filesystem_write write = {
    .offset = offset,
    .nbytes = nbytes,
    .alignment = w->alignment,
  };
  const uint64_t count = filesystem_write_count(&write);
  for (uint64_t i = 0; i < count; ++i) {
    struct filesystem_write_part part;
    CHECK(Error, filesystem_write_at(&write, i, &part) == 0);
    CHECK(Error, host_output_group_retain(group) == 0);
    if (io_scheduler_post(
          w->queue,
          (struct io_request){
            .op = IO_OP_WRITE,
            .file = w->token,
            .payload = (const char*)beg + (part.offset - offset),
            .nbytes = part.nbytes,
            .offset = part.offset,
            .finished_ctx = group,
            .finished = output_write_finished,
          })) {
      host_output_group_complete(group);
      goto Error;
    }
  }
  return 0;

Error:
  return 1;
}

// Growing the file is a barrier, so the writes posted behind it wait for it
// and then run inside a file that no longer has to be extended.
static int
fs_slot_presize(struct shard_writer* self, uint64_t nbytes)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->token.generation == 0 || !w->presize || nbytes <= w->prepared_capacity)
    return 0;

  return io_scheduler_post(w->queue,
                           (struct io_request){
                             .op = IO_OP_TRUNCATE,
                             .file = w->token,
                             .logical_size = nbytes,
                           });
}

static int
fs_slot_truncate(struct shard_writer* self, uint64_t logical_size)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->token.generation == 0)
    return 0;

  return io_scheduler_post(w->queue,
                           (struct io_request){
                             .op = IO_OP_TRUNCATE,
                             .file = w->token,
                             .logical_size = logical_size,
                           });
}

static int
fs_slot_finalize(struct shard_writer* self)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->token.generation == 0)
    return 0;

  if (io_scheduler_post(
        w->queue, (struct io_request){ .op = IO_OP_CLOSE, .file = w->token }))
    return 1;

  w->token = (struct io_file_token){ 0 };
  w->prepared_capacity = 0;
  return 0;
}

static void
prepared_file_clear(struct prepared_file* f)
{
  free(f->created_dirs);
  free(f->path);
  *f = (struct prepared_file){ .fd = PLATFORM_FD_INVALID };
}

static void
prepare_file(void* ctx, const struct io_request* req)
{
  struct shard_pool_fs* p = ctx;
  struct prepared_file* f = (struct prepared_file*)req->payload;
  char* directory = strdup(f->path);
  CHECK(Fail, directory);
  const int flags =
    PLATFORM_OPEN_EXCLUSIVE | (p->unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0);
  f->fd = platform_open_write(f->path, flags);
  if (f->fd == PLATFORM_FD_INVALID) {
    const size_t len = strlen(directory);
    for (size_t i = 1; i < len; ++i) {
      if (directory[i] != '/' && directory[i] != '\\')
        continue;
      if (i == 2 && directory[1] == ':')
        continue;
      const char saved = directory[i];
      directory[i] = 0;
      const int made = platform_mkdir_new(directory);
      directory[i] = saved;
      CHECK(Fail, made >= 0);
      f->created_dirs[i] = made != 0;
    }
    f->fd = platform_open_write(f->path, flags);
  }
  CHECK(Fail, f->fd != PLATFORM_FD_INVALID);
  f->owned = 1;
  if (f->capacity)
    CHECK(Fail, platform_ftruncate(f->fd, f->capacity) == 0);
  free(directory);
  return;
Fail:
  log_error("shard preparation failed: %s", f->path);
  atomic_store(&p->io_error, 1);
  free(directory);
}

static int
pool_fs_prepare(struct shard_pool* self,
                uint64_t slot,
                const char* key,
                uint64_t capacity)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  struct strbuf path = { 0 };
  CHECK(Fail, slot < p->nslots && key && !atomic_load(&p->io_error));
  if (!p->prepare_queue) {
    CHECK(Fail, p->nslots <= SIZE_MAX / sizeof(*p->prepared));
    p->prepared = calloc((size_t)p->nslots, sizeof(*p->prepared));
    CHECK(Fail, p->prepared);
    for (uint64_t i = 0; i < p->nslots; ++i)
      p->prepared[i].fd = PLATFORM_FD_INVALID;
    p->prepare_queue = io_scheduler_create(
      p->prepare_backend,
      (struct io_scheduler_limits){ .workers = 1, .max_requests = p->nslots });
    CHECK(Fail, p->prepare_queue);
  }
  struct prepared_file* f = &p->prepared[slot];
  CHECK(Fail, !f->path);
  CHECK(Fail, strbuf_appendf(&path, "%s/%s", strbuf_cstr(&p->root), key) == 0);
  f->path = strdup(strbuf_cstr(&path));
  f->created_dirs = calloc(strbuf_len(&path) + 1, 1);
  if (!f->path || !f->created_dirs) {
    prepared_file_clear(f);
    goto Fail;
  }
  f->capacity = p->slots[slot].presize ? capacity : 0;
  if (io_scheduler_post(p->prepare_queue,
                        (struct io_request){ .op = IO_OP_NOOP,
                                             .payload = f,
                                             .path = f->path,
                                             .logical_size = f->capacity })) {
    prepared_file_clear(f);
    goto Fail;
  }
  f->ready = io_scheduler_record(p->prepare_queue);
  strbuf_free(&path);
  return 0;
Fail:
  strbuf_free(&path);
  atomic_store(&p->io_error, 1);
  return 1;
}

static int
pool_fs_wait_prepared(struct shard_pool* self, uint64_t slot)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  if (slot >= p->nslots)
    return 1;
  if (p->prepare_queue && p->prepared[slot].path)
    io_event_wait(p->prepare_queue, p->prepared[slot].ready);
  return atomic_load(&p->io_error);
}

static int
pool_fs_cancel_prepared(struct shard_pool* self, uint64_t first, uint64_t count)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  if (first > p->nslots || count > p->nslots - first)
    return 1;
  if (!p->prepared)
    return 0;
  int failed = 0;
  size_t longest = 0;
  for (uint64_t i = first; i < first + count; ++i) {
    struct prepared_file* f = &p->prepared[i];
    if (!f->path)
      continue;
    if (p->prepare_queue)
      io_event_wait(p->prepare_queue, f->ready);
    if (f->fd != PLATFORM_FD_INVALID) {
      platform_close(f->fd);
      f->fd = PLATFORM_FD_INVALID;
    }
    if (f->owned) {
      if (platform_remove_file(f->path) == 0 ||
          platform_path_exists(f->path) == 0)
        f->owned = 0;
      else {
        log_error("unused shard cleanup failed: %s", f->path);
        failed = 1;
      }
    }
    const size_t len = strlen(f->path);
    if (len > longest)
      longest = len;
  }
  if (failed) {
    atomic_store(&p->io_error, 1);
    return 1;
  }
  for (size_t pos = longest; pos > 0; --pos) {
    for (uint64_t i = first; i < first + count; ++i) {
      struct prepared_file* f = &p->prepared[i];
      if (!f->path || strlen(f->path) <= pos || !f->created_dirs[pos])
        continue;
      char saved = f->path[pos];
      f->path[pos] = 0;
      if (platform_remove_empty_directory(f->path)) {
        log_error("unused shard directory cleanup failed: %s", f->path);
        failed = 1;
      } else {
        f->created_dirs[pos] = 0;
      }
      f->path[pos] = saved;
    }
  }
  if (!failed) {
    for (uint64_t i = first; i < first + count; ++i)
      prepared_file_clear(&p->prepared[i]);
  } else {
    atomic_store(&p->io_error, 1);
  }
  return failed;
}

static struct shard_writer*
pool_fs_open(struct shard_pool* self, uint64_t slot, const char* key)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  struct strbuf path = { 0 };
  char* owned_path = NULL;
  CHECK(Fail, slot < p->nslots);

  struct fs_slot* w = &p->slots[slot];

  if (w->token.generation != 0)
    CHECK(Fail, fs_slot_finalize(&w->base) == 0);

  if (strbuf_appendf(&path, "%s/%s", strbuf_cstr(&p->root), key))
    goto Fail;

  struct prepared_file* f = p->prepared ? &p->prepared[slot] : NULL;
  if (f && f->path) {
    CHECK(Fail, strcmp(f->path, strbuf_cstr(&path)) == 0);
    CHECK(Fail, pool_fs_wait_prepared(self, slot) == 0);
    const struct io_file_token token =
      io_backend_fs_adopt_file(p->backend, f->fd);
    CHECK(Fail, token.generation != 0);
    w->token = token;
    w->prepared_capacity = f->capacity;
    prepared_file_clear(f);
    strbuf_free(&path);
    return &w->base;
  }

  const size_t path_bytes = strbuf_len(&path) + 1;
  owned_path = (char*)malloc(path_bytes);
  CHECK(Fail, owned_path);
  memcpy(owned_path, strbuf_cstr(&path), path_bytes);

  const struct io_file_token token = io_backend_fs_reserve_file(p->backend);
  CHECK(Fail, token.generation != 0);

  if (io_scheduler_post(p->queue,
                        (struct io_request){ .op = IO_OP_OPEN,
                                             .file = token,
                                             .path = owned_path,
                                             .owned = owned_path,
                                             .owned_free = free })) {
    io_backend_fs_cancel_file(p->backend, token);
    goto Fail;
  }

  w->token = token;
  strbuf_free(&path);
  return &w->base;

Fail:
  free(owned_path);
  strbuf_free(&path);
  return NULL;
}

static struct io_event
pool_fs_record_fence(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  return io_scheduler_record(p->queue);
}

static void
pool_fs_wait_fence(struct shard_pool* self, struct io_event ev)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  io_event_wait(p->queue, ev);
}

static int
pool_fs_queue_metadata(struct shard_pool* self,
                       const char* key,
                       const void* data,
                       size_t len)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  struct strbuf path = { 0 };
  char* owned = NULL;
  CHECK(Fail, key && (data || len == 0));
  CHECK(Fail, !atomic_load(&p->io_error));
  CHECK(Fail, strbuf_appendf(&path, "%s/%s", strbuf_cstr(&p->root), key) == 0);
  const size_t path_bytes = strbuf_len(&path) + 1;
  CHECK(Fail, len <= SIZE_MAX - path_bytes);
  owned = (char*)malloc(path_bytes + len);
  CHECK(Fail, owned);
  memcpy(owned, strbuf_cstr(&path), path_bytes);
  if (len)
    memcpy(owned + path_bytes, data, len);

  CHECK(Fail,
        io_scheduler_post(p->queue,
                          (struct io_request){
                            .op = IO_OP_REPLACE,
                            .path = owned,
                            .payload = owned + path_bytes,
                            .nbytes = len,
                            .owned = owned,
                            .owned_free = free,
                          }) == 0);
  strbuf_free(&path);
  return 0;

Fail:
  atomic_store(&p->io_error, 1);
  free(owned);
  strbuf_free(&path);
  return 1;
}

static int
pool_fs_flush(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  struct io_event ev = io_scheduler_record(p->queue);
  io_event_wait(p->queue, ev);
  return atomic_load(&p->io_error);
}

static int
pool_fs_has_error(const struct shard_pool* self)
{
  const struct shard_pool_fs* p =
    container_of(self, struct shard_pool_fs, base);
  return atomic_load(&p->io_error);
}

static uint64_t
pool_fs_pending_bytes(const struct shard_pool* self)
{
  const struct shard_pool_fs* p =
    container_of(self, struct shard_pool_fs, base);
  return io_scheduler_pending_bytes(p->queue);
}

static size_t
pool_fs_required_shard_alignment(const struct shard_pool* self)
{
  const struct shard_pool_fs* p =
    container_of(self, struct shard_pool_fs, base);
  return p->unbuffered ? platform_page_alignment() : 0;
}

static void
pool_fs_destroy(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);

  if (pool_fs_cancel_prepared(self, 0, p->nslots))
    log_error("shard pool cleanup failed during destroy");
  io_scheduler_destroy(p->prepare_queue);
  if (p->prepared) {
    for (uint64_t i = 0; i < p->nslots; ++i)
      prepared_file_clear(&p->prepared[i]);
    free(p->prepared);
  }

  // Finalize any open slots
  for (uint64_t i = 0; i < p->nslots; ++i) {
    if (p->slots[i].token.generation != 0)
      fs_slot_finalize(&p->slots[i].base);
  }

  // The worker has to be gone before the backend holding its descriptors is.
  io_scheduler_destroy(p->queue);
  io_backend_fs_destroy(p->backend);

  free(p->slots);
  strbuf_free(&p->root);
  free(p);
}

void
shard_pool_fs_set_error(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  atomic_store(&p->io_error, 1);
}

#ifndef CHUCKY_IO_WORKERS
#define CHUCKY_IO_WORKERS 32u
#endif
#define DEFAULT_MAX_IN_FLIGHT_PER_FILE 4u

#define DEFAULT_MAX_QUEUED_BYTES (2ull << 30)

static struct io_scheduler_limits
resolve_limits(const struct io_scheduler_limits* limits)
{
  struct io_scheduler_limits resolved = limits ? *limits
                                               : (struct io_scheduler_limits){
                                                   0,
                                                 };
  if (!resolved.max_bytes)
    resolved.max_bytes = DEFAULT_MAX_QUEUED_BYTES;
  if (!resolved.workers)
    resolved.workers = CHUCKY_IO_WORKERS;
  if (!resolved.max_in_flight_per_file)
    resolved.max_in_flight_per_file = DEFAULT_MAX_IN_FLIGHT_PER_FILE;
  return resolved;
}

struct shard_pool*
shard_pool_fs_create_wrapped(const char* root,
                             uint64_t nslots,
                             int unbuffered,
                             const struct io_scheduler_limits* requested_limits,
                             struct shard_pool_fs_wrapper wrapper)
{
  CHECK(Fail, root);
  CHECK(Fail, nslots > 0);

  const struct io_scheduler_limits limits = resolve_limits(requested_limits);

  struct shard_pool_fs* p =
    (struct shard_pool_fs*)calloc(1, sizeof(struct shard_pool_fs));
  CHECK(Fail, p);

  p->base.open = pool_fs_open;
  p->base.prepare = pool_fs_prepare;
  p->base.wait_prepared = pool_fs_wait_prepared;
  p->base.cancel_prepared = pool_fs_cancel_prepared;
  p->prepare_backend = (struct io_backend){ .ctx = p, .execute = prepare_file };
  if (wrapper.wrap_prepare)
    p->prepare_backend = wrapper.wrap_prepare(wrapper.ctx, p->prepare_backend);
  p->base.record_fence = pool_fs_record_fence;
  p->base.wait_fence = pool_fs_wait_fence;
  p->base.queue_metadata = pool_fs_queue_metadata;
  p->base.flush = pool_fs_flush;
  p->base.has_error = pool_fs_has_error;
  p->base.pending_bytes = pool_fs_pending_bytes;
  p->base.required_shard_alignment = pool_fs_required_shard_alignment;
  p->base.destroy = pool_fs_destroy;
  p->nslots = nslots;
  p->unbuffered = unbuffered;
  CHECK(Fail_alloc, strbuf_set(&p->root, root) == 0);

  p->backend = io_backend_fs_create(&p->io_error,
                                    unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0);
  CHECK(Fail_alloc, p->backend);

  struct io_backend backend = io_backend_fs_as_backend(p->backend);
  if (wrapper.wrap)
    backend = wrapper.wrap(wrapper.ctx, backend);

  p->queue = io_scheduler_create(backend, limits);
  CHECK(Fail_backend, p->queue);

  p->slots = (struct fs_slot*)calloc((size_t)nslots, sizeof(struct fs_slot));
  CHECK(Fail_queue, p->slots);

  size_t page_size = unbuffered ? platform_page_size() : 0;
  for (uint64_t i = 0; i < nslots; ++i) {
    struct fs_slot* s = &p->slots[i];
    s->base.write = fs_slot_write;
    s->base.write_direct = fs_slot_write_direct;
    s->base.write_from_output = fs_slot_write_from_output;
    s->base.presize = fs_slot_presize;
    s->base.truncate = fs_slot_truncate;
    s->base.finalize = fs_slot_finalize;
    s->queue = p->queue;
    s->alignment = page_size;
    s->presize =
      limits.max_in_flight_per_file > 1 && platform_should_presize_shard();
  }

  if (wrapper.queue)
    *wrapper.queue = p->queue;

  return &p->base;

Fail_queue:
  io_scheduler_destroy(p->queue);
Fail_backend:
  io_backend_fs_destroy(p->backend);
Fail_alloc:
  strbuf_free(&p->root);
  free(p);
Fail:
  return NULL;
}

struct shard_pool*
shard_pool_fs_create(const char* root, uint64_t nslots, int unbuffered)
{
  return shard_pool_fs_create_wrapped(
    root, nslots, unbuffered, NULL, (struct shard_pool_fs_wrapper){ 0 });
}
