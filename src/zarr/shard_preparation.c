#include "zarr/shard_preparation.h"

#include "platform/platform.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_scheduler.h"

#include <stdlib.h>
#include <string.h>

struct prepared_file
{
  char* path;
  platform_fd fd;
  uint64_t capacity;
  struct io_event ready;
  int owned;
};

struct prepared_directory
{
  struct prepared_directory* next;
  size_t len;
  char path[];
};

struct shard_preparation
{
  struct io_scheduler* queue;
  struct prepared_file* files;
  uint64_t nslots;
  int open_flags;
  _Atomic int* io_error;

  // Shared by slots; retain nonempty directories across cancellation ranges.
  // New children precede their parents, so cleanup visits deepest first.
  // The lock also prevents cleanup from removing a directory between mkdir
  // and open on the worker. Sizing happens outside this lock.
  struct platform_mutex* directories_mutex;
  struct prepared_directory* directories;
};

static void
prepared_file_clear(struct prepared_file* f)
{
  free(f->path);
  *f = (struct prepared_file){ .fd = PLATFORM_FD_INVALID };
}

// Called with directories_mutex held. Allocate before mkdir so every new
// directory can be tracked even when a subsequent allocation fails.
static int
prepare_directory(struct shard_preparation* p, const char* path, size_t len)
{
  struct prepared_directory* d = malloc(sizeof(*d) + len + 1);
  if (!d)
    return 1;
  memcpy(d->path, path, len);
  d->path[len] = 0;
  d->len = len;
  const int made = platform_mkdir_new(d->path);
  if (made == 1) {
    d->next = p->directories;
    p->directories = d;
  } else {
    free(d);
  }
  return made < 0;
}

static void
prepare_file(void* ctx, const struct io_request* req)
{
  struct shard_preparation* p = ctx;
  struct prepared_file* f = (struct prepared_file*)req->payload;
  int failed = 0;
  platform_mutex_lock(p->directories_mutex);
  f->fd = platform_open_write(f->path, p->open_flags);
  if (f->fd == PLATFORM_FD_INVALID) {
    const size_t len = strlen(f->path);
    for (size_t i = 1; i < len; ++i) {
      if (f->path[i] != '/' && f->path[i] != '\\')
        continue;
      if (i == 2 && f->path[1] == ':')
        continue;
      if (prepare_directory(p, f->path, i)) {
        failed = 1;
        break;
      }
    }
    if (!failed)
      f->fd = platform_open_write(f->path, p->open_flags);
  }
  platform_mutex_unlock(p->directories_mutex);
  f->owned = f->fd != PLATFORM_FD_INVALID;
  if (failed || f->fd == PLATFORM_FD_INVALID ||
      (f->capacity && platform_ftruncate(f->fd, f->capacity))) {
    log_error("shard preparation failed: %s", f->path);
    atomic_store(p->io_error, 1);
  }
}

struct shard_preparation*
shard_preparation_create(uint64_t nslots,
                         int open_flags,
                         _Atomic int* io_error,
                         void* wrap_ctx,
                         struct io_backend (*wrap)(void*, struct io_backend))
{
  struct shard_preparation* p = calloc(1, sizeof(*p));
  CHECK(Fail, p);
  p->io_error = io_error;
  p->open_flags = open_flags | PLATFORM_OPEN_EXCLUSIVE;
  CHECK(Fail, nslots > 0 && nslots <= SIZE_MAX / sizeof(*p->files));
  p->files = calloc((size_t)nslots, sizeof(*p->files));
  CHECK(Fail, p->files);
  p->nslots = nslots;
  for (uint64_t i = 0; i < nslots; ++i)
    p->files[i].fd = PLATFORM_FD_INVALID;
  p->directories_mutex = platform_mutex_new();
  CHECK(Fail, p->directories_mutex);
  struct io_backend backend = { .ctx = p, .execute = prepare_file };
  if (wrap)
    backend = wrap(wrap_ctx, backend);
  p->queue = io_scheduler_create(
    backend,
    (struct io_scheduler_limits){ .workers = 1, .max_requests = nslots });
  CHECK(Fail, p->queue);
  return p;
Fail:
  shard_preparation_destroy(p);
  return NULL;
}

int
shard_preparation_post(struct shard_preparation* p,
                       uint64_t slot,
                       const char* path,
                       uint64_t capacity)
{
  CHECK(Fail, slot < p->nslots && path && !atomic_load(p->io_error));
  struct prepared_file* f = &p->files[slot];
  CHECK(Fail, !f->path);
  f->path = strdup(path);
  CHECK(Fail, f->path);
  f->capacity = capacity;
  if (io_scheduler_post(p->queue,
                        (struct io_request){
                          .op = IO_OP_NOOP, .payload = f, .path = f->path })) {
    prepared_file_clear(f);
    goto Fail;
  }
  f->ready = io_scheduler_record(p->queue);
  return 0;
Fail:
  atomic_store(p->io_error, 1);
  return 1;
}

int
shard_preparation_wait(struct shard_preparation* p, uint64_t slot)
{
  if (!p)
    return 0;
  if (slot >= p->nslots)
    return 1;
  if (p->files[slot].path)
    io_event_wait(p->queue, p->files[slot].ready);
  return atomic_load(p->io_error);
}

int
shard_preparation_adopt(struct shard_preparation* p,
                        uint64_t slot,
                        const char* path,
                        struct io_backend_fs* backend,
                        struct io_file_token* token,
                        uint64_t* capacity)
{
  if (!p)
    return 0;
  if (slot >= p->nslots)
    return -1;
  if (!p->files[slot].path)
    return 0;
  struct prepared_file* f = &p->files[slot];
  if (strcmp(f->path, path) || shard_preparation_wait(p, slot))
    return -1;
  *token = io_backend_fs_adopt_file(backend, f->fd);
  if (!token->generation)
    return -1;
  *capacity = f->capacity;

  // Adopted files make their ancestors permanent, including directories
  // created by another slot. Forget them to keep tracking bounded.
  platform_mutex_lock(p->directories_mutex);
  struct prepared_directory** link = &p->directories;
  const size_t len = strlen(path);
  while (*link) {
    struct prepared_directory* d = *link;
    if (len > d->len && !memcmp(path, d->path, d->len) &&
        (path[d->len] == '/' || path[d->len] == '\\')) {
      *link = d->next;
      free(d);
    } else {
      link = &d->next;
    }
  }
  platform_mutex_unlock(p->directories_mutex);
  prepared_file_clear(f);
  return 1;
}

int
shard_preparation_cancel(struct shard_preparation* p,
                         uint64_t first,
                         uint64_t count)
{
  if (!p)
    return 0;
  if (first > p->nslots || count > p->nslots - first)
    return 1;
  int failed = 0;
  for (uint64_t i = first; i < first + count; ++i) {
    struct prepared_file* f = &p->files[i];
    if (!f->path)
      continue;
    io_event_wait(p->queue, f->ready);
    if (f->fd != PLATFORM_FD_INVALID) {
      platform_close(f->fd);
      f->fd = PLATFORM_FD_INVALID;
    }
    if (f->owned && platform_remove_file(f->path) &&
        platform_path_exists(f->path) != 0) {
      log_error("unused shard cleanup failed: %s", f->path);
      failed = 1;
      continue;
    }
    prepared_file_clear(f);
  }
  if (p->directories_mutex) {
    platform_mutex_lock(p->directories_mutex);
    struct prepared_directory** link = &p->directories;
    while (*link) {
      struct prepared_directory* d = *link;
      const int rc = platform_remove_empty_directory(d->path);
      if (rc == 0) {
        *link = d->next;
        free(d);
      } else {
        if (rc < 0) {
          log_error("unused shard directory cleanup failed: %s", d->path);
          failed = 1;
        }
        link = &d->next;
      }
    }
    platform_mutex_unlock(p->directories_mutex);
  }
  if (failed)
    atomic_store(p->io_error, 1);
  return failed;
}

void
shard_preparation_destroy(struct shard_preparation* p)
{
  if (!p)
    return;
  if (shard_preparation_cancel(p, 0, p->nslots))
    log_error("shard preparation cleanup failed during destroy");
  io_scheduler_destroy(p->queue);
  for (uint64_t i = 0; i < p->nslots; ++i)
    prepared_file_clear(&p->files[i]);
  while (p->directories) {
    struct prepared_directory* next = p->directories->next;
    free(p->directories);
    p->directories = next;
  }
  platform_mutex_free(p->directories_mutex);
  free(p->files);
  free(p);
}
