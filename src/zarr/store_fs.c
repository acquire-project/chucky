#include "zarr/store_fs.h"
#include "platform/platform_io.h"
#include "util/prelude.h"
#include "zarr/shard_pool_fs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct store_fs
{
  struct store base;
  char root[4096];
  int unbuffered;
};

static int
fs_put(struct store* self, const char* key, const void* data, size_t len)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", fs->root, key);

  platform_fd fd = platform_open_write(path, 0);
  if (fd == PLATFORM_FD_INVALID)
    return 1;
  int rc = platform_write(fd, data, len);
  platform_close(fd);
  return rc != 0;
}

static int
fs_mkdirs(struct store* self, const char* key)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", fs->root, key);
  return platform_mkdirp(path);
}

static struct shard_pool*
fs_create_pool(struct store* self, uint64_t nslots)
{
  struct store_fs* fs = container_of(self, struct store_fs, base);
  return shard_pool_fs_create(fs->root, nslots, fs->unbuffered);
}

static void
fs_destroy(struct store* self)
{
  free(container_of(self, struct store_fs, base));
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
  fs->base.destroy = fs_destroy;
  fs->unbuffered = unbuffered;
  snprintf(fs->root, sizeof(fs->root), "%s", root);

  return &fs->base;

Fail:
  return NULL;
}
