#include "zarr/shard_pool_fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "zarr/io_queue.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// --- Writer slot for a single shard file ---

struct fs_slot
{
  struct shard_writer base;
  platform_fd fd;
  struct io_queue* queue;
  size_t alignment; // 0 = normal malloc, >0 = page-aligned allocation
  _Atomic uint64_t* retired_bytes; // points to shard_pool_fs.retired_bytes
  uint64_t* queued_bytes;          // points to shard_pool_fs.queued_bytes
  _Atomic int* io_error;           // points to shard_pool_fs.io_error
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
    log_error("shard_pool_fs pwrite failed");
    atomic_store(j->io_error, 1);
  }
  atomic_fetch_add(j->retired_bytes, j->nbytes);
}

static int
fs_slot_write(struct shard_writer* self,
              uint64_t offset,
              const void* beg,
              const void* end)
{
  struct fs_slot* w = (struct fs_slot*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);

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
    log_error("shard_pool_fs pwrite_ref failed");
    atomic_store(j->io_error, 1);
  }
  atomic_fetch_add(j->retired_bytes, j->nbytes);
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
fs_slot_finalize(struct shard_writer* self)
{
  struct fs_slot* w = (struct fs_slot*)self;
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

// --- Pool ---

struct shard_pool_fs
{
  struct shard_pool base;
  struct io_queue* queue;
  struct fs_slot* slots;
  uint64_t nslots;
  int unbuffered;
  char root[4096];
  uint64_t queued_bytes;
  _Atomic uint64_t retired_bytes;
  _Atomic int io_error;
};

static struct shard_writer*
pool_fs_open(struct shard_pool* self, uint64_t slot, const char* key)
{
  struct shard_pool_fs* p = container_of(self, struct shard_pool_fs, base);
  CHECK(Fail, slot < p->nslots);

  struct fs_slot* w = &p->slots[slot];

  // Finalize previous use of this slot if still open
  if (w->fd != PLATFORM_FD_INVALID)
    fs_slot_finalize(&w->base);

  // Build full path
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", p->root, key);

  int flags = p->unbuffered ? PLATFORM_OPEN_UNBUFFERED : 0;
  w->fd = platform_open_write(path, flags);
  if (w->fd == PLATFORM_FD_INVALID) {
    // Directory may not exist yet — create parent and retry
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
      log_error("shard_pool_fs: open(%s) failed", path);
      goto Fail;
    }
  }

  return &w->base;

Fail:
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

static size_t
pool_fs_pending_bytes(const struct shard_pool* self)
{
  const struct shard_pool_fs* p =
    container_of(self, struct shard_pool_fs, base);
  return p->queued_bytes - atomic_load(&p->retired_bytes);
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

  // Flush remaining I/O
  if (p->queue) {
    struct io_event ev = io_queue_record(p->queue);
    io_event_wait(p->queue, ev);
    io_queue_destroy(p->queue);
  }

  free(p->slots);
  free(p);
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
  p->base.destroy = pool_fs_destroy;
  p->nslots = nslots;
  p->unbuffered = unbuffered;
  snprintf(p->root, sizeof(p->root), "%s", root);

  p->queue = io_queue_create();
  CHECK(Fail_alloc, p->queue);

  p->slots = (struct fs_slot*)calloc((size_t)nslots, sizeof(struct fs_slot));
  CHECK(Fail_queue, p->slots);

  size_t page_size = unbuffered ? platform_page_size() : 0;
  for (uint64_t i = 0; i < nslots; ++i) {
    struct fs_slot* s = &p->slots[i];
    s->base.write = fs_slot_write;
    s->base.write_direct = fs_slot_write_direct;
    s->base.finalize = fs_slot_finalize;
    s->fd = PLATFORM_FD_INVALID;
    s->queue = p->queue;
    s->alignment = page_size;
    s->retired_bytes = &p->retired_bytes;
    s->queued_bytes = &p->queued_bytes;
    s->io_error = &p->io_error;
  }

  return &p->base;

Fail_queue:
  io_queue_destroy(p->queue);
Fail_alloc:
  free(p);
Fail:
  return NULL;
}
