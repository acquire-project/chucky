#include "zarr/io_backend.fs.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#include <stdlib.h>
#include <string.h>

#define FREE_LIST_END UINT32_MAX

enum file_state
{
  FILE_RESERVED = 0,
  FILE_READY,
  FILE_FAILED,
};

struct file_entry
{
  platform_fd fd;
  char* path;
  uint64_t generation;
  uint32_t next_free;
  uint32_t active_requests;
  int open_flags;
  uint8_t state;
  uint8_t opening;
  uint8_t closing;
};

struct handle_to_close
{
  struct io_file_token file;
  platform_fd fd;
};

struct io_backend_fs
{
  struct platform_mutex* mutex;
  struct platform_cond* handle_available;
  struct file_entry* files;
  uint32_t files_cap;
  uint32_t free_head;
  uint32_t next_to_close;
  uint32_t handle_count;
  uint32_t peak_handle_count;
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
  if (cap > SIZE_MAX / sizeof(struct file_entry)) {
    log_error("io_backend_fs: file registry is too large");
    return 1;
  }
  struct file_entry* files = (struct file_entry*)realloc(
    b->files, (size_t)cap * sizeof(struct file_entry));
  if (!files) {
    log_error("io_backend_fs: failed to grow the file registry");
    return 1;
  }

  b->files = files;
  for (uint32_t i = cap; i > b->files_cap; --i) {
    struct file_entry* e = &b->files[i - 1];
    *e = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                              .next_free = b->free_head };
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
  b->handle_available = platform_cond_new();
  CHECK(Fail_alloc, b->mutex && b->handle_available);
  CHECK(Fail_sync, grow_locked(b) == 0);

  return b;

Fail_sync:
Fail_alloc:
  platform_cond_free(b->handle_available);
  platform_mutex_free(b->mutex);
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
    if (b->files[i].fd != PLATFORM_FD_INVALID)
      platform_close(b->files[i].fd);
    free(b->files[i].path);
  }

  free(b->files);
  platform_cond_free(b->handle_available);
  platform_mutex_free(b->mutex);
  free(b);
}

struct io_file_token
io_backend_fs_reserve_file(struct io_backend_fs* b)
{
  struct io_file_token token = { 0 };

  platform_mutex_lock(b->mutex);
  if (b->free_head == FREE_LIST_END && grow_locked(b)) {
    platform_mutex_unlock(b->mutex);
    return token;
  }

  if (b->next_generation == UINT64_MAX) {
    platform_mutex_unlock(b->mutex);
    log_error("io_backend_fs: file generations are exhausted");
    return token;
  }
  const uint32_t index = b->free_head;
  struct file_entry* e = &b->files[index];
  b->free_head = e->next_free;
  *e = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                            .generation = ++b->next_generation };

  token.generation = e->generation;
  token.index = index;
  platform_mutex_unlock(b->mutex);
  return token;
}

static struct file_entry*
find_locked(struct io_backend_fs* b, struct io_file_token file)
{
  if (file.generation == 0 || file.index >= b->files_cap)
    return NULL;
  struct file_entry* e = &b->files[file.index];
  return e->generation == file.generation ? e : NULL;
}

void
io_backend_fs_cancel_file(struct io_backend_fs* b, struct io_file_token file)
{
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, file);
  if (e && e->state == FILE_RESERVED && e->fd == PLATFORM_FD_INVALID &&
      !e->opening && !e->closing && e->active_requests == 0) {
    *e = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                              .next_free = b->free_head };
    b->free_head = file.index;
  }
  platform_mutex_unlock(b->mutex);
}

uint32_t
io_backend_fs_handle_count(const struct io_backend_fs* b)
{
  struct io_backend_fs* mutable_b = (struct io_backend_fs*)b;
  platform_mutex_lock(mutable_b->mutex);
  const uint32_t count = mutable_b->handle_count;
  platform_mutex_unlock(mutable_b->mutex);
  return count;
}

uint32_t
io_backend_fs_peak_handle_count(const struct io_backend_fs* b)
{
  struct io_backend_fs* mutable_b = (struct io_backend_fs*)b;
  platform_mutex_lock(mutable_b->mutex);
  const uint32_t count = mutable_b->peak_handle_count;
  platform_mutex_unlock(mutable_b->mutex);
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

static char*
copy_path(const char* path)
{
  const size_t nbytes = strlen(path) + 1;
  char* copy = (char*)malloc(nbytes);
  if (copy)
    memcpy(copy, path, nbytes);
  return copy;
}

static struct handle_to_close
claim_handle_slot_locked(struct io_backend_fs* b)
{
  while (b->handle_count >= IO_BACKEND_FS_MAX_OPEN_FILES) {
    uint32_t index = b->next_to_close;
    for (uint32_t checked = 0; checked < b->files_cap; ++checked) {
      struct file_entry* e = &b->files[index];
      if (e->state == FILE_READY && e->fd != PLATFORM_FD_INVALID &&
          e->active_requests == 0 && !e->opening && !e->closing) {
        e->closing = 1;
        b->next_to_close = index + 1 == b->files_cap ? 0 : index + 1;
        struct handle_to_close handle = { .fd = e->fd };
        handle.file.generation = e->generation;
        handle.file.index = index;
        return handle;
      }
      if (++index == b->files_cap)
        index = 0;
    }
    platform_cond_wait(b->handle_available, b->mutex);
  }

  b->handle_count++;
  if (b->handle_count > b->peak_handle_count)
    b->peak_handle_count = b->handle_count;
  return (struct handle_to_close){ .fd = PLATFORM_FD_INVALID };
}

static int
close_handle_for_reuse(struct io_backend_fs* b, struct handle_to_close handle)
{
  if (handle.fd == PLATFORM_FD_INVALID)
    return 0;

  platform_close(handle.fd);

  int valid = 0;
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, handle.file);
  if (e && e->closing && e->fd == handle.fd) {
    e->fd = PLATFORM_FD_INVALID;
    e->closing = 0;
    valid = 1;
  }
  platform_cond_broadcast(b->handle_available);
  platform_mutex_unlock(b->mutex);

  if (!valid)
    record_failure(b, "io_backend_fs: invalid file close for reuse");
  return !valid;
}

static void
mark_open_failed(struct io_backend_fs* b, struct io_file_token file)
{
  char* path = NULL;
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, file);
  if (e && e->fd == PLATFORM_FD_INVALID && !e->closing &&
      e->active_requests == 0) {
    path = e->path;
    e->path = NULL;
    e->state = FILE_FAILED;
    if (e->opening) {
      e->opening = 0;
      if (b->handle_count > 0)
        b->handle_count--;
    }
  }
  platform_cond_broadcast(b->handle_available);
  platform_mutex_unlock(b->mutex);
  free(path);
}

static void
execute_open(struct io_backend_fs* b, const struct io_request* req)
{
  if (!req->path) {
    mark_open_failed(b, req->file);
    record_failure(b, "io_backend_fs: open request has no path");
    return;
  }

  char* path = copy_path(req->path);
  if (!path) {
    mark_open_failed(b, req->file);
    record_failure(b, "io_backend_fs: could not retain an open path");
    return;
  }

  int valid = 0;
  struct handle_to_close to_close = { .fd = PLATFORM_FD_INVALID };
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, req->file);
  if (e && e->state == FILE_RESERVED && e->fd == PLATFORM_FD_INVALID &&
      !e->opening && !e->closing && e->active_requests == 0) {
    e->opening = 1;
    to_close = claim_handle_slot_locked(b);
    valid = 1;
  }
  platform_mutex_unlock(b->mutex);

  if (!valid) {
    free(path);
    record_failure(b, "io_backend_fs: invalid open request");
    return;
  }
  if (close_handle_for_reuse(b, to_close)) {
    free(path);
    mark_open_failed(b, req->file);
    return;
  }

  platform_fd fd = open_write(req->path, req->open_flags);
  if (fd == PLATFORM_FD_INVALID) {
    log_error("io_backend_fs: open(%s) failed", req->path);
    atomic_store(b->io_error, 1);
    free(path);
    mark_open_failed(b, req->file);
    return;
  }

  platform_mutex_lock(b->mutex);
  e = find_locked(b, req->file);
  valid = e && e->state == FILE_RESERVED && e->opening &&
          e->fd == PLATFORM_FD_INVALID && !e->closing &&
          e->active_requests == 0;
  if (valid) {
    e->fd = fd;
    e->path = path;
    e->open_flags = req->open_flags & ~PLATFORM_OPEN_EXISTING;
    e->state = FILE_READY;
    e->opening = 0;
    platform_cond_broadcast(b->handle_available);
  }
  platform_mutex_unlock(b->mutex);

  if (!valid) {
    platform_close(fd);
    free(path);
    mark_open_failed(b, req->file);
    record_failure(b, "io_backend_fs: open token became stale");
  }
}

static int
acquire_file_handle(struct io_backend_fs* b,
                    struct io_file_token file,
                    platform_fd* out)
{
  for (;;) {
    const char* path = NULL;
    int flags = 0;
    struct handle_to_close to_close = { .fd = PLATFORM_FD_INVALID };

    platform_mutex_lock(b->mutex);
    struct file_entry* e = find_locked(b, file);
    if (!e) {
      platform_mutex_unlock(b->mutex);
      record_failure(b, "io_backend_fs: stale file token");
      return -1;
    }
    if (e->opening || e->closing) {
      platform_cond_wait(b->handle_available, b->mutex);
      platform_mutex_unlock(b->mutex);
      continue;
    }
    if (e->state == FILE_FAILED) {
      platform_mutex_unlock(b->mutex);
      return 0;
    }
    if (e->state != FILE_READY || !e->path) {
      platform_mutex_unlock(b->mutex);
      record_failure(b, "io_backend_fs: file was used before it opened");
      return -1;
    }
    if (e->fd != PLATFORM_FD_INVALID) {
      e->active_requests++;
      *out = e->fd;
      platform_mutex_unlock(b->mutex);
      return 1;
    }

    e->opening = 1;
    path = e->path;
    flags = e->open_flags | PLATFORM_OPEN_EXISTING;
    to_close = claim_handle_slot_locked(b);
    platform_mutex_unlock(b->mutex);

    if (close_handle_for_reuse(b, to_close)) {
      mark_open_failed(b, file);
      return -1;
    }

    platform_fd fd = platform_open_write(path, flags);
    if (fd == PLATFORM_FD_INVALID) {
      log_error("io_backend_fs: reopen(%s) failed", path);
      atomic_store(b->io_error, 1);
      mark_open_failed(b, file);
      return -1;
    }

    platform_mutex_lock(b->mutex);
    e = find_locked(b, file);
    const int valid = e && e->state == FILE_READY && e->opening &&
                      e->fd == PLATFORM_FD_INVALID && !e->closing &&
                      e->active_requests == 0 && e->path == path;
    if (valid) {
      e->fd = fd;
      e->active_requests = 1;
      e->opening = 0;
      platform_cond_broadcast(b->handle_available);
    }
    platform_mutex_unlock(b->mutex);

    if (valid) {
      *out = fd;
      return 1;
    }

    platform_close(fd);
    mark_open_failed(b, file);
    record_failure(b, "io_backend_fs: reopen token became stale");
    return -1;
  }
}

static void
finish_file_use(struct io_backend_fs* b,
                struct io_file_token file,
                platform_fd fd)
{
  int valid = 0;
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, file);
  if (e && e->fd == fd && e->active_requests > 0) {
    e->active_requests--;
    valid = 1;
    if (e->active_requests == 0)
      platform_cond_broadcast(b->handle_available);
  }
  platform_mutex_unlock(b->mutex);

  if (!valid)
    record_failure(b, "io_backend_fs: file use was already released");
}

static void
close_file(struct io_backend_fs* b, struct io_file_token file)
{
  platform_fd fd = PLATFORM_FD_INVALID;
  char* path = NULL;

  platform_mutex_lock(b->mutex);
  for (;;) {
    struct file_entry* e = find_locked(b, file);
    if (!e) {
      platform_mutex_unlock(b->mutex);
      record_failure(b, "io_backend_fs: stale close token");
      return;
    }
    if (e->opening || e->closing || e->active_requests > 0) {
      platform_cond_wait(b->handle_available, b->mutex);
      continue;
    }

    fd = e->fd;
    if (fd == PLATFORM_FD_INVALID) {
      path = e->path;
      *e = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                                .next_free = b->free_head };
      b->free_head = file.index;
      platform_cond_broadcast(b->handle_available);
      platform_mutex_unlock(b->mutex);
      free(path);
      return;
    }

    e->closing = 1;
    break;
  }
  platform_mutex_unlock(b->mutex);

  platform_close(fd);

  int valid = 0;
  platform_mutex_lock(b->mutex);
  struct file_entry* e = find_locked(b, file);
  if (e && e->closing && e->fd == fd && e->active_requests == 0) {
    path = e->path;
    *e = (struct file_entry){ .fd = PLATFORM_FD_INVALID,
                              .next_free = b->free_head };
    b->free_head = file.index;
    if (b->handle_count > 0) {
      b->handle_count--;
      valid = 1;
    }
  }
  platform_cond_broadcast(b->handle_available);
  platform_mutex_unlock(b->mutex);
  free(path);

  if (!valid)
    record_failure(b, "io_backend_fs: invalid close state");
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
  if (req->op == IO_OP_CLOSE) {
    close_file(b, req->file);
    return;
  }
  if (req->op != IO_OP_WRITE && req->op != IO_OP_TRUNCATE) {
    record_failure(b, "io_backend_fs: unsupported request");
    return;
  }

  platform_fd fd = PLATFORM_FD_INVALID;
  const int acquired = acquire_file_handle(b, req->file, &fd);
  if (acquired <= 0)
    return;

  if (req->op == IO_OP_WRITE) {
    if (platform_pwrite(fd, req->payload, (size_t)req->nbytes, req->offset))
      record_failure(b, "io_backend_fs: pwrite failed");
  } else if (platform_ftruncate(fd, req->logical_size)) {
    record_failure(b, "io_backend_fs: ftruncate failed");
  }

  finish_file_use(b, req->file, fd);
}

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b)
{
  return (struct io_backend){ .ctx = b, .execute = fs_execute };
}
