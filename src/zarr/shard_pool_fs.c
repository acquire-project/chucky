#include "zarr/shard_pool_fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// --- Pool ---

struct fs_slot;

struct shard_pool_fs
{
  struct shard_pool base;
  struct io_backend_fs* backend;
  struct io_queue* queue;
  struct fs_slot* slots;
  uint64_t nslots;
  int unbuffered;
  struct strbuf root; // owned
  _Atomic int io_error;
  // Test hook: one-shot, fail the next truncate.
  _Atomic int fail_next_truncate;
};

// --- Writer slot for a single shard file ---

struct fs_slot
{
  struct shard_writer base;
  struct io_file_token token; // zero generation means no file is open here
  struct io_queue* queue;
  size_t alignment;      // 0 = normal malloc, >0 = page-aligned allocation
  _Atomic int* io_error; // points to shard_pool_fs.io_error
  _Atomic int* fail_next_truncate; // points to shard_pool_fs.fail_next_truncate
};

static int
fs_slot_write(struct shard_writer* self,
              uint64_t offset,
              const void* beg,
              const void* end)
{
  struct fs_slot* w = (struct fs_slot*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);

  // Debug-build watchdog: under O_DIRECT, length and offset must both be
  // multiples of the device alignment (source pointer alignment is handled
  // by the aligned_alloc + memcpy below).
  CHECK(Error, w->alignment == 0 || nbytes % w->alignment == 0);
  CHECK(Error, w->alignment == 0 || offset % w->alignment == 0);

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

  if (io_queue_post(w->queue,
                    (struct io_request){
                      .op = IO_OP_WRITE,
                      .file = w->token,
                      .payload = buf,
                      .nbytes = nbytes,
                      .offset = offset,
                      .owned = buf,
                      .owned_free = buf_free,
                    })) {
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

  return io_queue_post(w->queue,
                       (struct io_request){
                         .op = IO_OP_WRITE,
                         .borrowed = 1,
                         .file = w->token,
                         .payload = beg,
                         .nbytes = nbytes,
                         .offset = offset,
                       });
}

static int
fs_slot_truncate(struct shard_writer* self, uint64_t logical_size)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->token.generation == 0)
    return 0;

  // Test hook: fail synchronously and mark the pool errored, so a footer write
  // already queued by the caller outlives the flush that bails on the error.
  if (w->fail_next_truncate && atomic_exchange(w->fail_next_truncate, 0)) {
    atomic_store(w->io_error, 1);
    return 1;
  }

  return io_queue_post(w->queue,
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

  if (io_queue_post(w->queue,
                    (struct io_request){ .op = IO_OP_CLOSE, .file = w->token }))
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

  // Finalize previous use of this slot if still open
  if (w->token.generation != 0)
    fs_slot_finalize(&w->base);

  struct strbuf path = { 0 };
  if (strbuf_appendf(&path, "%s/%s", strbuf_cstr(&p->root), key))
    goto Fail;

  int flags = p->unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0;
  platform_fd fd = platform_open_write(strbuf_cstr(&path), flags);
  if (fd == PLATFORM_FD_INVALID) {
    // Directory may not exist yet — create parent and retry.
    const char* path_cstr = strbuf_cstr(&path);
    const char* last_slash = strrchr(path_cstr, '/');
    if (last_slash) {
      struct strbuf dir = { 0 };
      if (strbuf_append(&dir, path_cstr, (size_t)(last_slash - path_cstr)) ==
            0 &&
          platform_mkdirp(strbuf_cstr(&dir)) == 0)
        fd = platform_open_write(strbuf_cstr(&path), flags);
      strbuf_free(&dir);
    }
    if (fd == PLATFORM_FD_INVALID) {
      log_error("shard_pool_fs: open(%s) failed", strbuf_cstr(&path));
      goto Fail;
    }
  }

  w->token = io_backend_fs_add_file(p->backend, fd);
  if (w->token.generation == 0) {
    platform_close(fd);
    goto Fail;
  }

  strbuf_free(&path);
  return &w->base;

Fail:
  strbuf_free(&path);
  return NULL;
}

static struct io_event
pool_fs_record_fence(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  return io_queue_record(p->queue);
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
  struct io_event ev = io_queue_record(p->queue);
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
  return io_queue_pending_bytes(p->queue);
}

static void
pool_fs_io_stats(const struct shard_pool* self, struct shard_pool_io_stats* out)
{
  const struct shard_pool_fs* p =
    container_of(self, struct shard_pool_fs, base);
  io_queue_get_stats(p->queue, &out->queue);
  out->files_opened = io_backend_fs_files_opened(p->backend);
  out->files_open_peak = io_backend_fs_files_open_peak(p->backend);
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

  // io_queue_destroy signals shutdown and joins the worker, which drains all
  // queued jobs first, so every outstanding write and close runs before the
  // backend that holds their descriptors goes away.
  io_queue_destroy(p->queue);
  io_backend_fs_destroy(p->backend);

  free(p->slots);
  strbuf_free(&p->root);
  free(p);
}

static void
fail_fn(void* arg)
{
  _Atomic int* io_error = (_Atomic int*)arg;
  log_error("shard_pool_fs: injected test failure");
  atomic_store(io_error, 1);
}

int
shard_pool_fs_inject_failing_job(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  return io_queue_post(
    p->queue, (struct io_request){ .fn = fail_fn, .ctx = (void*)&p->io_error });
}

int
shard_pool_fs_inject_failing_truncate(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  atomic_store(&p->fail_next_truncate, 1);
  return 0;
}

void
shard_pool_fs_set_error(struct shard_pool* self)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  atomic_store(&p->io_error, 1);
}

struct gate_ctx
{
  _Atomic int* gate;
  struct io_queue* queue;
};

static void
gate_fn(void* arg)
{
  struct gate_ctx* g = (struct gate_ctx*)arg;
  while (atomic_load(g->gate) == 0) {
    if (io_queue_is_shutdown(g->queue))
      return;
    platform_sleep_ns(1000000LL); // 1ms
  }
}

int
shard_pool_fs_inject_blocking_job(struct shard_pool* self, _Atomic int* gate)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  struct gate_ctx* g = (struct gate_ctx*)malloc(sizeof(*g));
  if (!g)
    return 1;
  g->gate = gate;
  g->queue = p->queue;
  if (io_queue_post(p->queue,
                    (struct io_request){
                      .fn = gate_fn, .ctx = (void*)g, .ctx_free = free })) {
    free(g);
    return 1;
  }
  return 0;
}

struct shard_pool*
shard_pool_fs_create(const char* root, uint64_t nslots, int unbuffered)
{
  CHECK(Fail, root);
  CHECK(Fail, nslots > 0);

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
  p->base.io_stats = pool_fs_io_stats;
  p->base.destroy = pool_fs_destroy;
  p->nslots = nslots;
  p->unbuffered = unbuffered;
  CHECK(Fail_alloc, strbuf_set(&p->root, root) == 0);

  p->backend = io_backend_fs_create(&p->io_error);
  CHECK(Fail_alloc, p->backend);

  p->queue = io_queue_create(io_backend_fs_as_backend(p->backend));
  CHECK(Fail_backend, p->queue);

  p->slots = (struct fs_slot*)calloc((size_t)nslots, sizeof(struct fs_slot));
  CHECK(Fail_queue, p->slots);

  size_t page_size = unbuffered ? platform_page_size() : 0;
  for (uint64_t i = 0; i < nslots; ++i) {
    struct fs_slot* s = &p->slots[i];
    s->base.write = fs_slot_write;
    s->base.write_direct = fs_slot_write_direct;
    s->base.truncate = fs_slot_truncate;
    s->base.finalize = fs_slot_finalize;
    s->queue = p->queue;
    s->alignment = page_size;
    s->io_error = &p->io_error;
    s->fail_next_truncate = &p->fail_next_truncate;
  }

  return &p->base;

Fail_queue:
  io_queue_destroy(p->queue);
Fail_backend:
  io_backend_fs_destroy(p->backend);
Fail_alloc:
  strbuf_free(&p->root);
  free(p);
Fail:
  return NULL;
}
