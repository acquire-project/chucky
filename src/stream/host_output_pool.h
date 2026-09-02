#pragma once

#include <stddef.h>
#include <stdint.h>

struct host_output_pool;
struct host_output_group;
struct stream_metrics;

struct host_output_allocator
{
  void* ctx;
  void* (*allocate)(void* ctx, size_t alignment, size_t bytes);
  void (*release)(void* ctx, void* data);
};

struct host_output
{
  void* data;
  size_t capacity;
  struct host_output_group* group;
};

struct host_output_pool_stats
{
  uint64_t buffers_in_use;
  uint64_t bytes_in_use;
  uint64_t buffers_in_use_peak;
  uint64_t bytes_in_use_peak;
  uint64_t wait_calls;
  uint64_t wait_count;
  double wait_ms_total;
  double wait_ms_best;
  double wait_ms_max;
  uint64_t lifetime_count;
  double lifetime_ms_total;
  double lifetime_ms_best;
  double lifetime_ms_max;
};

int
host_output_size(size_t required_bytes, size_t alignment, size_t* output_bytes);

int
host_output_count(uint64_t budget_bytes, size_t output_bytes, uint64_t* count);

struct host_output_pool*
host_output_pool_create(uint64_t count,
                        size_t output_bytes,
                        size_t alignment,
                        struct host_output_allocator allocator);

void
host_output_pool_close(struct host_output_pool* pool);

void
host_output_pool_destroy(struct host_output_pool* pool);

int
host_output_pool_acquire(struct host_output_pool* pool,
                         struct host_output* output);

void
host_output_pool_get_stats(const struct host_output_pool* pool,
                           struct host_output_pool_stats* stats);

int
host_output_group_retain(struct host_output_group* group);

void
host_output_group_complete(struct host_output_group* group);

void
host_output_group_seal(struct host_output_group* group);

void
host_output_pool_accumulate_metrics(const struct host_output_pool* pool,
                                    struct stream_metrics* metrics);
