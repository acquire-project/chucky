#include "zarr/store_fs.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/shard_pool_fs.h"

#include <stdlib.h>

struct store_fs
{
  struct store base;
  struct strbuf root; // owned
  int unbuffered;
};

// Build "<root>/<key>" into a fresh strbuf. Caller frees with strbuf_free.
// Returns 0 on success.
static int
fs_join(const struct store_fs* fs, const char* key, struct strbuf* out)
{
  return strbuf_appendf(out, "%s/%s", strbuf_cstr(&fs->root), key);
}

static int
fs_put(struct store* self, const char* key, const void* data, size_t len)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  struct strbuf path = { 0 };
  int rc = 1;
  if (fs_join(fs, key, &path))
    goto done;

  platform_fd fd = platform_open_write(strbuf_cstr(&path), 0);
  if (fd == PLATFORM_FD_INVALID)
    goto done;
  rc = platform_write(fd, data, len) != 0;
  platform_close(fd);
done:
  strbuf_free(&path);
  return rc;
}

static int
fs_mkdirs(struct store* self, const char* key)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  struct strbuf path = { 0 };
  int rc = 1;
  if (fs_join(fs, key, &path))
    goto done;
  rc = platform_mkdirp(strbuf_cstr(&path));
done:
  strbuf_free(&path);
  return rc;
}

static struct shard_pool*
fs_create_pool(struct store* self, uint64_t nslots)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  return shard_pool_fs_create(strbuf_cstr(&fs->root), nslots, fs->unbuffered);
}

static int
fs_has_existing_data(struct store* self)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  struct strbuf path = { 0 };
  int exists;
  if (strbuf_appendf(&path, "%s/zarr.json", strbuf_cstr(&fs->root))) {
    // Fail closed: assume data may exist rather than silently overwrite.
    log_error("store_fs: failed to build path for existence check");
    exists = 1;
  } else {
    exists = platform_path_exists(strbuf_cstr(&path));
  }
  strbuf_free(&path);
  return exists;
}

static void
fs_destroy(struct store* self)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  strbuf_free(&fs->root);
  free(fs);
}

struct store*
store_fs_create(const char* root, int unbuffered)
{
  CHECK(Fail, root);

  struct store_fs* fs = (struct store_fs*)calloc(1, sizeof(*fs));
  CHECK(Fail, fs);

  fs->base.put = fs_put;
  fs->base.mkdirs = fs_mkdirs;
  fs->base.create_pool = fs_create_pool;
  fs->base.has_existing_data = fs_has_existing_data;
  fs->base.destroy = fs_destroy;
  fs->unbuffered = unbuffered;
  CHECK(Fail_fs, strbuf_set(&fs->root, root) == 0);

  return &fs->base;

Fail_fs:
  strbuf_free(&fs->root);
  free(fs);
Fail:
  return NULL;
}
