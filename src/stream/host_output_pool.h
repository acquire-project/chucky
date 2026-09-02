#pragma once

#include <stddef.h>

struct host_output_pool;
struct host_output_group;

#define HOST_OUTPUT_COUNT 2u

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

int
host_output_size(size_t required_bytes, size_t alignment, size_t* output_bytes);

struct host_output_pool*
host_output_pool_create(size_t output_bytes,
                        size_t alignment,
                        struct host_output_allocator allocator);

void
host_output_pool_destroy(struct host_output_pool* pool);

int
host_output_pool_acquire(struct host_output_pool* pool,
                         struct host_output* output);

int
host_output_group_retain(struct host_output_group* group);

void
host_output_group_complete(struct host_output_group* group);

void
host_output_group_seal(struct host_output_group* group);
