#include "zarr/shard_pool_fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_backend.uring.h"
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
  struct io_backend_uring* ring; // null unless the writes go to a ring
  struct io_queue* queue;
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
  struct io_queue* queue;
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

  // Room is claimed before the buffer is allocated, so the copy stays under
  // the ceiling on queued memory.
  CHECK_SILENT(Error, io_queue_reserve(w->queue, req) == 0);

  void* buf;
  void (*buf_free)(void*);
  if (w->alignment > 0) {
    buf = platform_aligned_alloc(w->alignment, nbytes);
    buf_free = platform_aligned_free;
  } else {
    buf = malloc(nbytes);
    buf_free = free;
  }
  CHECK(Release, buf);
  memcpy(buf, beg, nbytes);

  req.payload = buf;
  req.owned = buf;
  req.owned_free = buf_free;
  io_queue_commit(w->queue, req);
  return 0;

Release:
  io_queue_release(w->queue, nbytes);
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

// Growing the file is a barrier, so the writes posted behind it wait for it
// and then run inside a file that no longer has to be extended.
static int
fs_slot_presize(struct shard_writer* self, uint64_t nbytes)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->token.generation == 0 || !w->presize || nbytes == 0)
    return 0;

  return io_queue_post(w->queue,
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

  // The worker has to be gone before the backend holding its descriptors is.
  io_queue_destroy(p->queue);
  io_backend_uring_destroy(p->ring);
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

// Eight is where the sweep peaks on both an md RAID10 of eight drives (1.34x)
// and a two-drive mirror (1.21x); past it the rest of the pipeline, not the
// drive, is what is waited on, and throughput falls back. Four per file hides
// one write's latency behind the next on a shard file written by itself.
#define DEFAULT_WORKERS 8u
#define DEFAULT_WRITES_IN_FLIGHT 8u
#define DEFAULT_WRITES_IN_FLIGHT_PER_FILE 4u

// Room for the payloads the queue holds. The deepest backlog measured is
// 1392 MiB, from an uncompressed 256cube run with one write at a time, so
// this bounds a runaway rather than throttling work that is merely busy.
#define DEFAULT_MAX_QUEUED_BYTES (2ull << 30)

void
shard_pool_fs_scheduling_defaults(struct io_scheduling* io)
{
  if (!io->workers)
    io->workers = DEFAULT_WORKERS;
  if (!io->writes_in_flight)
    io->writes_in_flight = DEFAULT_WRITES_IN_FLIGHT;
  if (!io->writes_in_flight_per_file)
    io->writes_in_flight_per_file = DEFAULT_WRITES_IN_FLIGHT_PER_FILE;
  // Settled here, not in the pool, so what a caller records is what it got.
  if (io->backend != IO_BACKEND_URING)
    return;
  if (!io_backend_uring_supported())
    io->backend = IO_BACKEND_THREADS;
  else if (io->writes_in_flight > IO_BACKEND_URING_MAX_DEPTH)
    io->writes_in_flight = IO_BACKEND_URING_MAX_DEPTH;
}

static struct io_queue_limits
limits_from(const struct io_scheduling* resolved)
{
  return (struct io_queue_limits){
    .max_bytes = DEFAULT_MAX_QUEUED_BYTES,
    .workers = resolved->workers,
    .writes_in_flight = resolved->writes_in_flight,
    .writes_in_flight_per_file = resolved->writes_in_flight_per_file,
  };
}

struct shard_pool*
shard_pool_fs_create_wrapped(const char* root,
                             uint64_t nslots,
                             int unbuffered,
                             const struct io_scheduling* io,
                             struct shard_pool_fs_wrapper wrapper)
{
  CHECK(Fail, root);
  CHECK(Fail, nslots > 0);

  struct io_scheduling resolved = io ? *io : (struct io_scheduling){ 0 };
  shard_pool_fs_scheduling_defaults(&resolved);
  const struct io_queue_limits limits = limits_from(&resolved);

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

  struct io_backend backend = io_backend_fs_as_backend(p->backend);
  if (resolved.backend == IO_BACKEND_URING) {
    p->ring = io_backend_uring_create(
      p->backend, &p->io_error, limits.writes_in_flight);
    if (p->ring)
      backend = io_backend_uring_as_backend(p->ring);
    else
      log_error("shard_pool_fs: no ring for this pool; writing on the workers");
  }
  if (wrapper.wrap) {
    const struct io_backend inner = backend;
    backend = wrapper.wrap(wrapper.ctx, inner);
    // The queue only stops the outermost backend, and a ring left running
    // reads a queue that has already been freed.
    CHECK(Fail_backend, !inner.stop || backend.stop);
  }

  p->queue = io_queue_create(backend, limits);
  CHECK(Fail_backend, p->queue);
  CHECK(Fail_queue, !p->ring || io_backend_uring_start(p->ring, p->queue) == 0);

  p->slots = (struct fs_slot*)calloc((size_t)nslots, sizeof(struct fs_slot));
  CHECK(Fail_queue, p->slots);

  size_t page_size = unbuffered ? platform_page_size() : 0;
  for (uint64_t i = 0; i < nslots; ++i) {
    struct fs_slot* s = &p->slots[i];
    s->base.write = fs_slot_write;
    s->base.write_direct = fs_slot_write_direct;
    s->base.presize = fs_slot_presize;
    s->base.truncate = fs_slot_truncate;
    s->base.finalize = fs_slot_finalize;
    s->queue = p->queue;
    s->alignment = page_size;
    s->presize =
      limits.writes_in_flight_per_file > 1 && platform_should_presize_shard();
  }

  if (wrapper.queue)
    *wrapper.queue = p->queue;

  return &p->base;

Fail_queue:
  io_queue_destroy(p->queue);
Fail_backend:
  io_backend_uring_destroy(p->ring);
  io_backend_fs_destroy(p->backend);
Fail_alloc:
  strbuf_free(&p->root);
  free(p);
Fail:
  return NULL;
}

struct shard_pool*
shard_pool_fs_create(const char* root,
                     uint64_t nslots,
                     int unbuffered,
                     const struct io_scheduling* io)
{
  return shard_pool_fs_create_wrapped(
    root, nslots, unbuffered, io, (struct shard_pool_fs_wrapper){ 0 });
}
