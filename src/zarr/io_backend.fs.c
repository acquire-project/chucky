#include "zarr/io_backend.fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#include <stdlib.h>
#include <string.h>

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
  uint32_t handle_count;
  uint32_t peak_handle_count;
  uint64_t next_generation;
  int open_flags;
  _Atomic int* io_error;
};

static int
grow_locked(struct io_backend_fs* b)
{
  CHECK(Fail, b->files_cap <= UINT32_MAX / 2);
  const uint32_t cap = b->files_cap ? b->files_cap * 2 : 16;
  CHECK(Fail, sizeof(struct file_entry) <= SIZE_MAX / cap);
  struct file_entry* files =
    (struct file_entry*)realloc(b->files, (size_t)cap * sizeof(*files));
  CHECK(Fail, files);

  b->files = files;
  for (uint32_t i = cap; i > b->files_cap; --i) {
    b->files[i - 1] = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                                           .next_free = b->free_head };
    b->free_head = i - 1;
  }
  b->files_cap = cap;
  return 0;
Fail:
  return 1;
}

struct io_backend_fs*
io_backend_fs_create(_Atomic int* io_error, int open_flags)
{
  CHECK(Fail, io_error);
  struct io_backend_fs* b = (struct io_backend_fs*)calloc(1, sizeof(*b));
  CHECK(Fail, b);
  b->io_error = io_error;
  b->open_flags = open_flags;
  b->free_head = FREE_LIST_END;
  b->mutex = platform_mutex_new();
  CHECK(Fail_alloc, b->mutex && grow_locked(b) == 0);
  return b;
Fail_alloc:
  io_backend_fs_destroy(b);
Fail:
  return NULL;
}

void
io_backend_fs_destroy(struct io_backend_fs* b)
{
  if (!b)
    return;
  for (uint32_t i = 0; i < b->files_cap; ++i) {
    if (b->files[i].fd != PLATFORM_FD_INVALID)
      platform_close(b->files[i].fd);
  }
  free(b->files);
  platform_mutex_free(b->mutex);
  free(b);
}

struct io_file_token
io_backend_fs_reserve_file(struct io_backend_fs* b)
{
  platform_mutex_lock(b->mutex);
  CHECK(Unlock, b->free_head != FREE_LIST_END || grow_locked(b) == 0);
  CHECK(Unlock, b->next_generation != UINT64_MAX);
  const uint32_t index = b->free_head;
  struct file_entry* e = &b->files[index];
  b->free_head = e->next_free;
  *e = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                            .generation = ++b->next_generation };
  const struct io_file_token token = { .generation = e->generation,
                                       .index = index };
  platform_mutex_unlock(b->mutex);
  return token;
Unlock:
  platform_mutex_unlock(b->mutex);
  return (struct io_file_token){ 0 };
}

static struct file_entry*
find_locked(struct io_backend_fs* b, struct io_file_token file)
{
  if (file.generation == 0 || file.index >= b->files_cap)
    return NULL;
  struct file_entry* e = &b->files[file.index];
  return e->generation == file.generation ? e : NULL;
}

static void
release_locked(struct io_backend_fs* b, struct io_file_token file)
{
  struct file_entry* e = &b->files[file.index];
  *e =
    (struct file_entry){ .fd = PLATFORM_FD_INVALID, .next_free = b->free_head };
  b->free_head = file.index;
}

void
io_backend_fs_cancel_file(struct io_backend_fs* b, struct io_file_token file)
{
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, file);
  if (e && e->fd == PLATFORM_FD_INVALID)
    release_locked(b, file);
  platform_mutex_unlock(b->mutex);
}

uint32_t
io_backend_fs_handle_count(const struct io_backend_fs* b)
{
  platform_mutex_lock(b->mutex);
  const uint32_t count = b->handle_count;
  platform_mutex_unlock(b->mutex);
  return count;
}

uint32_t
io_backend_fs_peak_handle_count(const struct io_backend_fs* b)
{
  platform_mutex_lock(b->mutex);
  const uint32_t count = b->peak_handle_count;
  platform_mutex_unlock(b->mutex);
  return count;
}

static void
record_failure(struct io_backend_fs* b, const char* message)
{
  log_error("%s", message);
  atomic_store(b->io_error, 1);
}

static platform_fd
open_write(const char* path, int flags)
{
  platform_fd fd = platform_open_write(path, flags);
  if (fd != PLATFORM_FD_INVALID)
    return fd;

  const char* last_slash = strrchr(path, '/');
  if (!last_slash)
    return PLATFORM_FD_INVALID;

  struct strbuf dir = { 0 };
  if (strbuf_append(&dir, path, (size_t)(last_slash - path)) == 0 &&
      platform_mkdirp(strbuf_cstr(&dir)) == 0)
    fd = platform_open_write(path, flags);
  strbuf_free(&dir);
  return fd;
}

static void
execute_open(struct io_backend_fs* b, const struct io_request* req)
{
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, req->file);
  if (!e || e->fd != PLATFORM_FD_INVALID || !req->path) {
    platform_mutex_unlock(b->mutex);
    record_failure(b, "io_backend_fs: invalid open request");
    return;
  }
  b->handle_count++;
  if (b->handle_count > b->peak_handle_count)
    b->peak_handle_count = b->handle_count;
  platform_mutex_unlock(b->mutex);

  const platform_fd fd = open_write(req->path, b->open_flags);
  if (fd == PLATFORM_FD_INVALID) {
    log_error("io_backend_fs: open(%s) failed", req->path);
    atomic_store(b->io_error, 1);
  }

  platform_mutex_lock(b->mutex);
  b->files[req->file.index].fd = fd;
  if (fd == PLATFORM_FD_INVALID)
    b->handle_count--;
  platform_mutex_unlock(b->mutex);
}

static void
fs_execute(void* ctx, const struct io_request* req)
{
  struct io_backend_fs* b = (struct io_backend_fs*)ctx;
  if (req->op == IO_OP_NOOP)
    return;
  if (req->op == IO_OP_OPEN) {
    execute_open(b, req);
    return;
  }

  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, req->file);
  const platform_fd fd = e ? e->fd : PLATFORM_FD_INVALID;
  const int found = e != NULL;
  platform_mutex_unlock(b->mutex);
  if (!found) {
    record_failure(b, "io_backend_fs: stale file token");
    return;
  }

  switch (req->op) {
    case IO_OP_WRITE:
      if (fd != PLATFORM_FD_INVALID &&
          platform_pwrite(fd, req->payload, (size_t)req->nbytes, req->offset))
        record_failure(b, "io_backend_fs: pwrite failed");
      break;
    case IO_OP_TRUNCATE:
      if (fd != PLATFORM_FD_INVALID &&
          platform_ftruncate(fd, req->logical_size))
        record_failure(b, "io_backend_fs: ftruncate failed");
      break;
    case IO_OP_CLOSE:
      if (fd != PLATFORM_FD_INVALID)
        platform_close(fd);
      platform_mutex_lock(b->mutex);
      if (fd != PLATFORM_FD_INVALID)
        b->handle_count--;
      release_locked(b, req->file);
      platform_mutex_unlock(b->mutex);
      break;
    default:
      record_failure(b, "io_backend_fs: unsupported request");
      break;
  }
}

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b)
{
  return (struct io_backend){ .ctx = b, .execute = fs_execute };
}
