#include "zarr/io_backend.fs.h"
#include "platform/platform.h"
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
  uint64_t next_generation;

  _Atomic int* io_error;
};

static int
grow_locked(struct io_backend_fs* b)
{
  if (b->files_cap > UINT32_MAX / 2) {
    log_error("io_backend_fs: file registry is full");
    return 1;
  }
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
  CHECK(Fail, io_error);
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
    if (b->files[i].generation != 0 && b->files[i].fd != PLATFORM_FD_INVALID)
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
  e->generation = ++b->next_generation;

  token.generation = e->generation;
  token.index = index;
  platform_mutex_unlock(b->mutex);
  return token;
}

struct io_file_token
io_backend_fs_reserve_file(struct io_backend_fs* b)
{
  return io_backend_fs_add_file(b, PLATFORM_FD_INVALID);
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
}

void
io_backend_fs_cancel_file(struct io_backend_fs* b, struct io_file_token file)
{
  platform_mutex_lock(b->mutex);
  if (file.generation != 0 && file.index < b->files_cap) {
    struct file_entry* e = &b->files[file.index];
    if (e->generation == file.generation && e->fd == PLATFORM_FD_INVALID) {
      e->generation = 0;
      e->next_free = b->free_head;
      b->free_head = file.index;
    }
  }
  platform_mutex_unlock(b->mutex);
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

static int
install_fd(struct io_backend_fs* b, struct io_file_token file, platform_fd fd)
{
  int installed = 0;
  platform_mutex_lock(b->mutex);
  if (file.generation != 0 && file.index < b->files_cap) {
    struct file_entry* e = &b->files[file.index];
    if (e->generation == file.generation && e->fd == PLATFORM_FD_INVALID) {
      e->fd = fd;
      installed = 1;
    }
  }
  platform_mutex_unlock(b->mutex);
  return installed;
}

static void
execute_open(struct io_backend_fs* b, const struct io_request* req)
{
  platform_fd existing = PLATFORM_FD_INVALID;
  if (!req->path || !resolve(b, req->file, &existing) ||
      existing != PLATFORM_FD_INVALID) {
    record_failure(b, "io_backend_fs: invalid open request");
    return;
  }

  platform_fd fd = open_write(req->path, req->open_flags);
  if (fd == PLATFORM_FD_INVALID) {
    log_error("io_backend_fs: open(%s) failed", req->path);
    atomic_store(b->io_error, 1);
    return;
  }
  if (!install_fd(b, req->file, fd)) {
    platform_close(fd);
    record_failure(b, "io_backend_fs: open token became stale");
  }
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

  platform_fd fd = PLATFORM_FD_INVALID;
  if (!resolve(b, req->file, &fd)) {
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
      release(b, req->file);
      break;
    default:
      break;
  }
}

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b)
{
  return (struct io_backend){ .ctx = b, .execute = fs_execute };
}
