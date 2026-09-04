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

struct shard_pool_fs
{
  struct shard_pool base;
  struct io_backend_fs* backend;
  struct io_scheduler* queue;
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
  if (w->token.generation == 0 || !w->presize || nbytes == 0)
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
  return 0;
}

static struct shard_writer*
pool_fs_open(struct shard_pool* self, uint64_t slot, const char* key)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  CHECK(Fail, slot < p->nslots);

  struct fs_slot* w = &p->slots[slot];

  if (w->token.generation != 0)
    CHECK(Fail, fs_slot_finalize(&w->base) == 0);

  struct strbuf path = { 0 };
  if (strbuf_appendf(&path, "%s/%s", strbuf_cstr(&p->root), key))
    goto Fail;

  const size_t path_bytes = strbuf_len(&path) + 1;
  char* owned_path = (char*)malloc(path_bytes);
  CHECK(Fail, owned_path);
  memcpy(owned_path, strbuf_cstr(&path), path_bytes);

  const struct io_file_token token = io_backend_fs_reserve_file(p->backend);
  CHECK(Fail_path, token.generation != 0);

  if (io_scheduler_post(
        p->queue,
        (struct io_request){
          .op = IO_OP_OPEN,
          .file = token,
          .path = owned_path,
          .open_flags = p->unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0,
          .owned = owned_path,
          .owned_free = free,
        })) {
    io_backend_fs_cancel_file(p->backend, token);
    goto Fail_path;
  }

  w->token = token;
  strbuf_free(&path);
  return &w->base;

Fail_path:
  free(owned_path);
Fail:
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

#define DEFAULT_WORKERS 8u
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
    resolved.workers = DEFAULT_WORKERS;
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
  p->base.record_fence = pool_fs_record_fence;
  p->base.wait_fence = pool_fs_wait_fence;
  p->base.flush = pool_fs_flush;
  p->base.has_error = pool_fs_has_error;
  p->base.pending_bytes = pool_fs_pending_bytes;
  p->base.required_shard_alignment = pool_fs_required_shard_alignment;
  p->base.destroy = pool_fs_destroy;
  p->nslots = nslots;
  p->unbuffered = unbuffered;
  CHECK(Fail_alloc, strbuf_set(&p->root, root) == 0);

  p->backend = io_backend_fs_create(&p->io_error);
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
