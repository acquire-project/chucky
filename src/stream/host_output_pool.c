#include "stream/host_output_pool.h"

#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>

struct host_output_entry
{
  struct host_output_pool* pool;
  void* data;
  size_t borrowed;
  uint8_t in_use;
  uint8_t sealed;
};

struct host_output_pool
{
  struct platform_mutex* mutex;
  struct platform_cond* changed;
  struct host_output_entry* entries;
  size_t next;
  size_t waiters;
  size_t in_use;
  size_t output_bytes;
  struct host_output_allocator allocator;
  int closed;
};

struct host_output_group
{
  struct host_output_pool* pool;
};

static void*
default_allocate(void* ctx, size_t alignment, size_t bytes)
{
  (void)ctx;
  return platform_aligned_alloc(alignment, bytes);
}

static void
default_release(void* ctx, void* data)
{
  (void)ctx;
  platform_aligned_free(data);
}

static struct host_output_entry*
entry_from_group(struct host_output_group* group)
{
  return (struct host_output_entry*)group;
}

static void
release_entry_locked(struct host_output_entry* entry)
{
  struct host_output_pool* pool = entry->pool;
  CHECK(Return, entry->in_use && entry->sealed && entry->borrowed == 0);
  entry->in_use = 0;
  entry->sealed = 0;
  pool->in_use--;
  platform_cond_broadcast(pool->changed);
Return:
  return;
}

int
host_output_size(size_t required_bytes, size_t alignment, size_t* output_bytes)
{
  if (!output_bytes || required_bytes == 0 || alignment == 0)
    return 1;
  const size_t remainder = required_bytes % alignment;
  const size_t add = remainder == 0 ? 0 : alignment - remainder;
  if (required_bytes > SIZE_MAX - add)
    return 1;
  *output_bytes = required_bytes + add;
  return 0;
}

struct host_output_pool*
host_output_pool_create(size_t output_bytes,
                        size_t alignment,
                        struct host_output_allocator allocator)
{
  CHECK(Fail, output_bytes > 0 && alignment > 0);
  CHECK(Fail, output_bytes % alignment == 0);
  if (!allocator.allocate) {
    allocator.allocate = default_allocate;
    allocator.release = default_release;
  }
  CHECK(Fail, allocator.release);

  struct host_output_pool* pool =
    (struct host_output_pool*)calloc(1, sizeof(*pool));
  CHECK(Fail, pool);
  pool->output_bytes = output_bytes;
  pool->allocator = allocator;
  pool->mutex = platform_mutex_new();
  pool->changed = platform_cond_new();
  pool->entries = (struct host_output_entry*)calloc(HOST_OUTPUT_COUNT,
                                                    sizeof(*pool->entries));
  CHECK(Fail_pool, pool->mutex && pool->changed && pool->entries);

  for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i) {
    pool->entries[i].pool = pool;
    pool->entries[i].data =
      allocator.allocate(allocator.ctx, alignment, output_bytes);
    CHECK(Fail_pool, pool->entries[i].data);
    CHECK(Fail_pool, (uintptr_t)pool->entries[i].data % alignment == 0);
  }
  return pool;

Fail_pool:
  host_output_pool_destroy(pool);
Fail:
  return NULL;
}

void
host_output_pool_destroy(struct host_output_pool* pool)
{
  if (!pool)
    return;
  if (pool->mutex) {
    platform_mutex_lock(pool->mutex);
    pool->closed = 1;
    if (pool->changed) {
      platform_cond_broadcast(pool->changed);
      while (pool->in_use > 0 || pool->waiters > 0)
        platform_cond_wait(pool->changed, pool->mutex);
    }
    platform_mutex_unlock(pool->mutex);
  }
  if (pool->entries) {
    for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i)
      if (pool->entries[i].data)
        pool->allocator.release(pool->allocator.ctx, pool->entries[i].data);
  }
  free(pool->entries);
  platform_cond_free(pool->changed);
  platform_mutex_free(pool->mutex);
  free(pool);
}

int
host_output_pool_acquire(struct host_output_pool* pool,
                         struct host_output* output)
{
  if (!pool || !output)
    return 1;
  platform_mutex_lock(pool->mutex);

  struct host_output_entry* entry = NULL;
  for (;;) {
    if (pool->closed)
      break;
    for (size_t i = 0; i < HOST_OUTPUT_COUNT; ++i) {
      const size_t index = (pool->next + i) % HOST_OUTPUT_COUNT;
      if (!pool->entries[index].in_use) {
        entry = &pool->entries[index];
        pool->next = (index + 1) % HOST_OUTPUT_COUNT;
        break;
      }
    }
    if (entry)
      break;
    pool->waiters++;
    platform_cond_wait(pool->changed, pool->mutex);
    pool->waiters--;
    platform_cond_broadcast(pool->changed);
  }

  if (!entry) {
    platform_mutex_unlock(pool->mutex);
    return 1;
  }

  entry->borrowed = 0;
  entry->sealed = 0;
  entry->in_use = 1;
  pool->in_use++;

  *output = (struct host_output){
    .data = entry->data,
    .capacity = pool->output_bytes,
    .group = (struct host_output_group*)entry,
  };
  platform_mutex_unlock(pool->mutex);
  return 0;
}

int
host_output_group_retain(struct host_output_group* group)
{
  if (!group)
    return 1;
  struct host_output_entry* entry = entry_from_group(group);
  struct host_output_pool* pool = entry->pool;
  platform_mutex_lock(pool->mutex);
  const int refused = !entry->in_use || entry->sealed;
  if (!refused)
    entry->borrowed++;
  platform_mutex_unlock(pool->mutex);
  return refused;
}

void
host_output_group_complete(struct host_output_group* group)
{
  if (!group)
    return;
  struct host_output_entry* entry = entry_from_group(group);
  struct host_output_pool* pool = entry->pool;
  platform_mutex_lock(pool->mutex);
  CHECK(Unlock, entry->in_use && entry->borrowed > 0);
  entry->borrowed--;
  if (entry->sealed && entry->borrowed == 0)
    release_entry_locked(entry);
Unlock:
  platform_mutex_unlock(pool->mutex);
}

void
host_output_group_seal(struct host_output_group* group)
{
  if (!group)
    return;
  struct host_output_entry* entry = entry_from_group(group);
  struct host_output_pool* pool = entry->pool;
  platform_mutex_lock(pool->mutex);
  CHECK(Unlock, entry->in_use && !entry->sealed);
  entry->sealed = 1;
  if (entry->borrowed == 0)
    release_entry_locked(entry);
Unlock:
  platform_mutex_unlock(pool->mutex);
}
