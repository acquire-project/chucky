#include "stream/host_output_pool.h"

#include "platform/platform.h"
#include "types.stream.h"
#include "util/prelude.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct host_output_entry
{
  struct host_output_pool* pool;
  void* data;
  uint64_t borrowed;
  int64_t acquired_ns;
  uint8_t in_use;
  uint8_t sealed;
};

struct host_output_pool
{
  struct platform_mutex* mutex;
  struct platform_cond* changed;
  struct host_output_entry* entries;
  uint64_t count;
  uint64_t next;
  uint64_t waiters;
  size_t output_bytes;
  size_t alignment;
  struct host_output_allocator allocator;
  struct host_output_pool_stats stats;
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
record_sample(double value,
              uint64_t* count,
              double* total,
              double* best,
              double* maximum)
{
  if (*count == 0 || value < *best)
    *best = value;
  if (value > *maximum)
    *maximum = value;
  (*count)++;
  *total += value;
}

static void
release_entry_locked(struct host_output_entry* entry, int64_t now)
{
  struct host_output_pool* pool = entry->pool;
  CHECK(Return, entry->in_use && entry->sealed && entry->borrowed == 0);
  const double lifetime_ms = (double)(now - entry->acquired_ns) / 1e6;
  record_sample(lifetime_ms,
                &pool->stats.lifetime_count,
                &pool->stats.lifetime_ms_total,
                &pool->stats.lifetime_ms_best,
                &pool->stats.lifetime_ms_max);
  entry->in_use = 0;
  entry->sealed = 0;
  entry->acquired_ns = 0;
  pool->stats.buffers_in_use--;
  pool->stats.bytes_in_use -= pool->output_bytes;
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

int
host_output_count(uint64_t budget_bytes, size_t output_bytes, uint64_t* count)
{
  if (!count || output_bytes == 0)
    return 1;
  if (budget_bytes == 0) {
    *count = 2;
    return 0;
  }
  *count = budget_bytes / output_bytes;
  return *count == 0;
}

struct host_output_pool*
host_output_pool_create(uint64_t count,
                        size_t output_bytes,
                        size_t alignment,
                        struct host_output_allocator allocator)
{
  CHECK(Fail, count > 0 && output_bytes > 0 && alignment > 0);
  CHECK(Fail, output_bytes % alignment == 0);
  CHECK_MUL_OVERFLOW(Fail, count, sizeof(struct host_output_entry), SIZE_MAX);
  CHECK_MUL_OVERFLOW(Fail, count, output_bytes, UINT64_MAX);
  if (!allocator.allocate) {
    allocator.allocate = default_allocate;
    allocator.release = default_release;
  }
  CHECK(Fail, allocator.release);

  struct host_output_pool* pool =
    (struct host_output_pool*)calloc(1, sizeof(*pool));
  CHECK(Fail, pool);
  pool->count = count;
  pool->output_bytes = output_bytes;
  pool->alignment = alignment;
  pool->allocator = allocator;
  pool->mutex = platform_mutex_new();
  pool->changed = platform_cond_new();
  pool->entries =
    (struct host_output_entry*)calloc((size_t)count, sizeof(*pool->entries));
  CHECK(Fail_pool, pool->mutex && pool->changed && pool->entries);

  for (uint64_t i = 0; i < count; ++i) {
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
host_output_pool_close(struct host_output_pool* pool)
{
  if (!pool)
    return;
  platform_mutex_lock(pool->mutex);
  pool->closed = 1;
  platform_cond_broadcast(pool->changed);
  platform_mutex_unlock(pool->mutex);
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
      while (pool->stats.buffers_in_use > 0 || pool->waiters > 0)
        platform_cond_wait(pool->changed, pool->mutex);
    }
    platform_mutex_unlock(pool->mutex);
  }
  if (pool->entries) {
    for (uint64_t i = 0; i < pool->count; ++i)
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
  const int64_t start = platform_monotonic_ns();
  platform_mutex_lock(pool->mutex);
  pool->stats.wait_calls++;

  struct host_output_entry* entry = NULL;
  for (;;) {
    if (pool->closed)
      break;
    for (uint64_t i = 0; i < pool->count; ++i) {
      const uint64_t index = (pool->next + i) % pool->count;
      if (!pool->entries[index].in_use) {
        entry = &pool->entries[index];
        pool->next = (index + 1) % pool->count;
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

  const double wait_ms = (double)(platform_monotonic_ns() - start) / 1e6;
  record_sample(wait_ms,
                &pool->stats.wait_count,
                &pool->stats.wait_ms_total,
                &pool->stats.wait_ms_best,
                &pool->stats.wait_ms_max);
  if (!entry) {
    platform_mutex_unlock(pool->mutex);
    return 1;
  }

  entry->borrowed = 0;
  entry->sealed = 0;
  entry->in_use = 1;
  entry->acquired_ns = platform_monotonic_ns();
  pool->stats.buffers_in_use++;
  pool->stats.bytes_in_use += pool->output_bytes;
  if (pool->stats.buffers_in_use > pool->stats.buffers_in_use_peak)
    pool->stats.buffers_in_use_peak = pool->stats.buffers_in_use;
  if (pool->stats.bytes_in_use > pool->stats.bytes_in_use_peak)
    pool->stats.bytes_in_use_peak = pool->stats.bytes_in_use;

  *output = (struct host_output){
    .data = entry->data,
    .capacity = pool->output_bytes,
    .group = (struct host_output_group*)entry,
  };
  platform_mutex_unlock(pool->mutex);
  return 0;
}

void
host_output_pool_get_stats(const struct host_output_pool* pool,
                           struct host_output_pool_stats* stats)
{
  if (!stats)
    return;
  memset(stats, 0, sizeof(*stats));
  if (!pool)
    return;
  struct host_output_pool* mutable_pool = (struct host_output_pool*)pool;
  platform_mutex_lock(mutable_pool->mutex);
  *stats = mutable_pool->stats;
  platform_mutex_unlock(mutable_pool->mutex);
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
    release_entry_locked(entry, platform_monotonic_ns());
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
    release_entry_locked(entry, platform_monotonic_ns());
Unlock:
  platform_mutex_unlock(pool->mutex);
}

static void
accumulate_samples(struct stream_metric* metric,
                   uint64_t count,
                   double total_ms,
                   double best_ms,
                   double max_ms)
{
  if (count == 0)
    return;
  metric->ms += (float)total_ms;
  metric->max_ms = metric->max_ms > max_ms ? metric->max_ms : (float)max_ms;
  if (metric->count == 0 || best_ms < metric->best_ms)
    metric->best_ms = (float)best_ms;
  if (count > (uint64_t)(INT_MAX - metric->count))
    metric->count = INT_MAX;
  else
    metric->count += (int)count;
}

void
host_output_pool_accumulate_metrics(const struct host_output_pool* pool,
                                    struct stream_metrics* metrics)
{
  if (!pool || !metrics)
    return;
  struct host_output_pool_stats stats;
  host_output_pool_get_stats(pool, &stats);
  accumulate_samples(&metrics->host_output_wait,
                     stats.wait_count,
                     stats.wait_ms_total,
                     stats.wait_ms_best,
                     stats.wait_ms_max);
  metrics->host_output_wait.wait_calls += stats.wait_calls;
  accumulate_samples(&metrics->host_output_lifetime,
                     stats.lifetime_count,
                     stats.lifetime_ms_total,
                     stats.lifetime_ms_best,
                     stats.lifetime_ms_max);
  metrics->host_output_buffers_in_use += stats.buffers_in_use;
  metrics->host_output_bytes_in_use += stats.bytes_in_use;
  metrics->host_output_buffers_in_use_peak += stats.buffers_in_use_peak;
  metrics->host_output_bytes_in_use_peak += stats.bytes_in_use_peak;
}
