#include "zarr/io_backend.fs.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>

#define FREE_LIST_END UINT32_MAX

struct file_entry
{
  platform_fd fd;
  uint64_t generation;
  uint32_t next_free;
};

struct io_backend_fs
{
  struct platform_mutex* mutex;
  struct file_entry* files;
  uint32_t files_cap;
  uint32_t free_head;

  _Atomic int* io_error;
  _Atomic uint64_t files_opened;
  _Atomic uint64_t files_open_now;
  _Atomic uint64_t files_open_peak;

  _Atomic int fail_next_noop;
  _Atomic int fail_next_truncate;
  _Atomic int block_next_noop;
  _Atomic int* block_gate;
  _Atomic int stopped;
};

static int
grow_locked(struct io_backend_fs* b)
{
  const uint32_t cap = b->files_cap ? b->files_cap * 2 : 16;
  struct file_entry* files = (struct file_entry*)realloc(
    b->files, (size_t)cap * sizeof(struct file_entry));
  if (!files) {
    log_error("io_backend_fs: failed to grow the file registry");
    return 1;
  }

  b->files = files;
  for (uint32_t i = cap; i > b->files_cap; --i) {
    struct file_entry* e = &b->files[i - 1];
    e->fd = PLATFORM_FD_INVALID;
    e->generation = 0;
    e->next_free = b->free_head;
    b->free_head = i - 1;
  }
  b->files_cap = cap;
  return 0;
}

struct io_backend_fs*
io_backend_fs_create(_Atomic int* io_error)
{
  struct io_backend_fs* b =
    (struct io_backend_fs*)calloc(1, sizeof(struct io_backend_fs));
  CHECK(Fail, b);

  b->io_error = io_error;
  b->free_head = FREE_LIST_END;
  b->mutex = platform_mutex_new();
  CHECK(Fail_alloc, b->mutex);
  CHECK(Fail_mutex, grow_locked(b) == 0);

  return b;

Fail_mutex:
  platform_mutex_free(b->mutex);
Fail_alloc:
  free(b);
Fail:
  return NULL;
}

void
io_backend_fs_destroy(struct io_backend_fs* b)
{
  if (!b)
    return;

  for (uint32_t i = 0; i < b->files_cap; ++i) {
    if (b->files[i].generation != 0)
      platform_close(b->files[i].fd);
  }

  free(b->files);
  platform_mutex_free(b->mutex);
  free(b);
}

struct io_file_token
io_backend_fs_add_file(struct io_backend_fs* b, platform_fd fd)
{
  struct io_file_token token = { 0 };

  platform_mutex_lock(b->mutex);
  if (b->free_head == FREE_LIST_END && grow_locked(b)) {
    platform_mutex_unlock(b->mutex);
    return token;
  }

  const uint32_t index = b->free_head;
  struct file_entry* e = &b->files[index];
  b->free_head = e->next_free;
  e->fd = fd;
  e->generation = atomic_fetch_add(&b->files_opened, 1) + 1;

  const uint64_t open_now = atomic_fetch_add(&b->files_open_now, 1) + 1;
  uint64_t peak = atomic_load(&b->files_open_peak);
  while (open_now > peak &&
         !atomic_compare_exchange_weak(&b->files_open_peak, &peak, open_now)) {
  }

  token.generation = e->generation;
  token.index = index;
  platform_mutex_unlock(b->mutex);
  return token;
}

uint64_t
io_backend_fs_files_opened(const struct io_backend_fs* b)
{
  return atomic_load(&b->files_opened);
}

uint64_t
io_backend_fs_files_open_peak(const struct io_backend_fs* b)
{
  return atomic_load(&b->files_open_peak);
}

void
io_backend_fs_inject_failure(struct io_backend_fs* b)
{
  atomic_store(&b->fail_next_noop, 1);
}

void
io_backend_fs_inject_failing_truncate(struct io_backend_fs* b)
{
  atomic_store(&b->fail_next_truncate, 1);
}

void
io_backend_fs_inject_block(struct io_backend_fs* b, _Atomic int* gate)
{
  b->block_gate = gate;
  atomic_store(&b->block_next_noop, 1);
}

void
io_backend_fs_stop(struct io_backend_fs* b)
{
  atomic_store(&b->stopped, 1);
}

static int
resolve(struct io_backend_fs* b, struct io_file_token file, platform_fd* fd)
{
  int found = 0;
  platform_mutex_lock(b->mutex);
  if (file.generation != 0 && file.index < b->files_cap &&
      b->files[file.index].generation == file.generation) {
    *fd = b->files[file.index].fd;
    found = 1;
  }
  platform_mutex_unlock(b->mutex);
  return found;
}

static void
release(struct io_backend_fs* b, struct io_file_token file)
{
  platform_mutex_lock(b->mutex);
  struct file_entry* e = &b->files[file.index];
  e->fd = PLATFORM_FD_INVALID;
  e->generation = 0;
  e->next_free = b->free_head;
  b->free_head = file.index;
  platform_mutex_unlock(b->mutex);

  atomic_fetch_sub(&b->files_open_now, 1);
}

static void
record_failure(struct io_backend_fs* b,
               struct io_completion* out,
               const char* message)
{
  log_error("%s", message);
  atomic_store(b->io_error, 1);
  out->nbytes = 0;
  out->status = IO_FAILED;
}

static int
fs_execute(void* ctx,
           const struct io_request* req,
           uint64_t seq,
           struct io_completion* out)
{
  struct io_backend_fs* b = (struct io_backend_fs*)ctx;
  out->seq = seq;

  if (req->op == IO_OP_NOOP) {
    if (atomic_exchange(&b->fail_next_noop, 0)) {
      record_failure(b, out, "io_backend_fs: injected test failure");
      return IO_DONE;
    }
    if (atomic_exchange(&b->block_next_noop, 0)) {
      while (atomic_load(b->block_gate) == 0 && !atomic_load(&b->stopped))
        platform_sleep_ns(1000000LL);
    }
    return IO_DONE;
  }

  platform_fd fd = PLATFORM_FD_INVALID;
  if (!resolve(b, req->file, &fd)) {
    record_failure(b, out, "io_backend_fs: stale file token");
    return IO_DONE;
  }

  switch (req->op) {
    case IO_OP_WRITE:
      if (platform_pwrite(fd, req->payload, (size_t)req->nbytes, req->offset))
        record_failure(b, out, "io_backend_fs: pwrite failed");
      break;
    case IO_OP_TRUNCATE:
      if (atomic_exchange(&b->fail_next_truncate, 0))
        record_failure(b, out, "io_backend_fs: injected truncate failure");
      else if (platform_ftruncate(fd, req->logical_size))
        record_failure(b, out, "io_backend_fs: ftruncate failed");
      break;
    case IO_OP_CLOSE:
      platform_close(fd);
      release(b, req->file);
      break;
    default:
      break;
  }
  return IO_DONE;
}

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b)
{
  return (struct io_backend){ .ctx = b, .execute = fs_execute };
}
