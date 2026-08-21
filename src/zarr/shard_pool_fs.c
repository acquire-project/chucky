#include "zarr/shard_pool_fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// --- Pool ---

struct fs_slot;

struct shard_pool_fs
{
  struct shard_pool base;
  struct io_queue* queue;
  struct fs_slot* slots;
  uint64_t nslots;
  int unbuffered;
  struct strbuf root; // owned
  _Atomic int io_error;
  // Test hook: one-shot, fail the next truncate.
  _Atomic int fail_next_truncate;

  // A fresh number per open; a descriptor can be reused after close.
  _Atomic uint64_t files_opened;
  _Atomic uint64_t files_open_now;
  _Atomic uint64_t files_open_peak;
};

// --- Writer slot for a single shard file ---

struct fs_slot
{
  struct shard_writer base;
  platform_fd fd;
  uint64_t generation; // which open of a shard file
  struct io_queue* queue;
  size_t alignment;      // 0 = normal malloc, >0 = page-aligned allocation
  _Atomic int* io_error; // points to shard_pool_fs.io_error
  _Atomic int* fail_next_truncate; // points to shard_pool_fs.fail_next_truncate
  _Atomic uint64_t* files_open_now;
};

struct pwrite_job
{
  platform_fd fd;
  uint64_t offset;
  size_t nbytes;
  size_t data_off;       // byte offset from start of struct to data
  _Atomic int* io_error; // set on write failure
  uint8_t data[];        // used when data_off == sizeof(struct pwrite_job)
};

static void
pwrite_fn(void* arg)
{
  struct pwrite_job* j = (struct pwrite_job*)arg;
  const void* data = (const char*)j + j->data_off;
  if (platform_pwrite(j->fd, data, j->nbytes, j->offset) != 0) {
    log_error("shard_pool_fs pwrite failed");
    atomic_store(j->io_error, 1);
  }
}

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

  if (w->queue) {
    struct pwrite_job* j;
    void (*job_free)(void*) = free;
    if (w->alignment > 0) {
      size_t hdr = align_up(sizeof(struct pwrite_job), w->alignment);
      j =
        (struct pwrite_job*)platform_aligned_alloc(w->alignment, hdr + nbytes);
      CHECK(Error, j);
      j->data_off = hdr;
      job_free = platform_aligned_free;
    } else {
      j = (struct pwrite_job*)malloc(sizeof(struct pwrite_job) + nbytes);
      CHECK(Error, j);
      j->data_off = sizeof(struct pwrite_job);
    }
    j->fd = w->fd;
    j->offset = offset;
    j->nbytes = nbytes;
    j->io_error = w->io_error;
    memcpy((char*)j + j->data_off, beg, nbytes);
    if (io_queue_post(w->queue,
                      (struct io_request){
                        .fn = pwrite_fn,
                        .ctx = j,
                        .ctx_free = job_free,
                        .nbytes = nbytes,
                        .file = { .generation = w->generation },
                      })) {
      job_free(j);
      goto Error;
    }
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
  const void* data;      // NOT owned — points into pinned memory
  _Atomic int* io_error; // set on write failure
};

static void
pwrite_ref_fn(void* arg)
{
  struct pwrite_ref_job* j = (struct pwrite_ref_job*)arg;
  if (platform_pwrite(j->fd, j->data, j->nbytes, j->offset) != 0) {
    log_error("shard_pool_fs pwrite_ref failed");
    atomic_store(j->io_error, 1);
  }
}

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

  if (w->queue) {
    struct pwrite_ref_job* j =
      (struct pwrite_ref_job*)malloc(sizeof(struct pwrite_ref_job));
    CHECK(Error, j);
    j->fd = w->fd;
    j->offset = offset;
    j->nbytes = nbytes;
    j->data = beg;
    j->io_error = w->io_error;
    if (io_queue_post(w->queue,
                      (struct io_request){
                        .fn = pwrite_ref_fn,
                        .ctx = j,
                        .ctx_free = free,
                        .nbytes = nbytes,
                        .file = { .generation = w->generation },
                        .borrowed = 1,
                      })) {
      free(j);
      goto Error;
    }
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
  _Atomic uint64_t* files_open_now;
};

static void
close_fn(void* arg)
{
  struct close_job* j = (struct close_job*)arg;
  platform_close(j->fd);
  atomic_fetch_sub(j->files_open_now, 1);
}

struct truncate_job
{
  platform_fd fd;
  uint64_t logical_size;
  _Atomic int* io_error;
};

static void
truncate_fn(void* arg)
{
  struct truncate_job* j = (struct truncate_job*)arg;
  if (platform_ftruncate(j->fd, j->logical_size) != 0) {
    log_error("shard_pool_fs ftruncate failed");
    atomic_store(j->io_error, 1);
  }
}

static int
fs_slot_truncate(struct shard_writer* self, uint64_t logical_size)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->fd == PLATFORM_FD_INVALID)
    return 0;

  // Test hook: fail synchronously and mark the pool errored, so a footer write
  // already queued by the caller outlives the flush that bails on the error.
  if (w->fail_next_truncate && atomic_exchange(w->fail_next_truncate, 0)) {
    atomic_store(w->io_error, 1);
    return 1;
  }

  if (w->queue) {
    struct truncate_job* j =
      (struct truncate_job*)malloc(sizeof(struct truncate_job));
    if (!j)
      return 1;
    j->fd = w->fd;
    j->logical_size = logical_size;
    j->io_error = w->io_error;
    if (io_queue_post(w->queue,
                      (struct io_request){
                        .fn = truncate_fn,
                        .ctx = j,
                        .ctx_free = free,
                        .file = { .generation = w->generation },
                      })) {
      free(j);
      return 1;
    }
    return 0;
  }
  return platform_ftruncate(w->fd, logical_size) == 0 ? 0 : 1;
}

static int
fs_slot_finalize(struct shard_writer* self)
{
  struct fs_slot* w = (struct fs_slot*)self;
  if (w->fd == PLATFORM_FD_INVALID)
    return 0;

  if (w->queue) {
    struct close_job* j = (struct close_job*)malloc(sizeof(struct close_job));
    CHECK(Error, j);
    j->fd = w->fd;
    j->files_open_now = w->files_open_now;
    if (io_queue_post(w->queue,
                      (struct io_request){
                        .fn = close_fn,
                        .ctx = j,
                        .ctx_free = free,
                        .file = { .generation = w->generation },
                      })) {
      free(j);
      goto Error;
    }
  } else {
    platform_close(w->fd);
    atomic_fetch_sub(w->files_open_now, 1);
  }

  w->fd = PLATFORM_FD_INVALID;
  return 0;

Error:
  return 1;
}

static struct shard_writer*
pool_fs_open(struct shard_pool* self, uint64_t slot, const char* key)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  CHECK(Fail, slot < p->nslots);

  struct fs_slot* w = &p->slots[slot];

  // Finalize previous use of this slot if still open
  if (w->fd != PLATFORM_FD_INVALID)
    fs_slot_finalize(&w->base);

  struct strbuf path = { 0 };
  if (strbuf_appendf(&path, "%s/%s", strbuf_cstr(&p->root), key))
    goto Fail;

  int flags = p->unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0;
  w->fd = platform_open_write(strbuf_cstr(&path), flags);
  if (w->fd == PLATFORM_FD_INVALID) {
    // Directory may not exist yet — create parent and retry.
    const char* path_cstr = strbuf_cstr(&path);
    const char* last_slash = strrchr(path_cstr, '/');
    if (last_slash) {
      struct strbuf dir = { 0 };
      if (strbuf_append(&dir, path_cstr, (size_t)(last_slash - path_cstr)) ==
            0 &&
          platform_mkdirp(strbuf_cstr(&dir)) == 0)
        w->fd = platform_open_write(strbuf_cstr(&path), flags);
      strbuf_free(&dir);
    }
    if (w->fd == PLATFORM_FD_INVALID) {
      log_error("shard_pool_fs: open(%s) failed", strbuf_cstr(&path));
      goto Fail;
    }
  }

  w->generation = atomic_fetch_add(&p->files_opened, 1) + 1;
  const uint64_t open_now = atomic_fetch_add(&p->files_open_now, 1) + 1;
  uint64_t peak = atomic_load(&p->files_open_peak);
  while (open_now > peak &&
         !atomic_compare_exchange_weak(&p->files_open_peak, &peak, open_now)) {
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
  out->files_opened = atomic_load(&p->files_opened);
  out->files_open_peak = atomic_load(&p->files_open_peak);
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
    if (p->slots[i].fd != PLATFORM_FD_INVALID)
      fs_slot_finalize(&p->slots[i].base);
  }

  // Tear down the queue. io_queue_destroy signals shutdown and joins the
  // worker; the worker drains all queued jobs before exiting, so any
  // outstanding pwrite/close jobs run before destroy returns.
  if (p->queue)
    io_queue_destroy(p->queue);

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

  p->queue = io_queue_create();
  CHECK(Fail_alloc, p->queue);

  p->slots = (struct fs_slot*)calloc((size_t)nslots, sizeof(struct fs_slot));
  CHECK(Fail_queue, p->slots);

  size_t page_size = unbuffered ? platform_page_size() : 0;
  for (uint64_t i = 0; i < nslots; ++i) {
    struct fs_slot* s = &p->slots[i];
    s->base.write = fs_slot_write;
    s->base.write_direct = fs_slot_write_direct;
    s->base.truncate = fs_slot_truncate;
    s->base.finalize = fs_slot_finalize;
    s->fd = PLATFORM_FD_INVALID;
    s->queue = p->queue;
    s->alignment = page_size;
    s->io_error = &p->io_error;
    s->fail_next_truncate = &p->fail_next_truncate;
    s->files_open_now = &p->files_open_now;
  }

  return &p->base;

Fail_queue:
  io_queue_destroy(p->queue);
Fail_alloc:
  strbuf_free(&p->root);
  free(p);
Fail:
  return NULL;
}
