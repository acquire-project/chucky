// Filesystem-backed shard writer pool.
// Uses io_queue for async pwrite with sequence-number fencing.
#pragma once

#include "zarr/io_backend.h"
#include "zarr/shard_pool.h"

#include <stdint.h>

struct io_queue;

// Create a filesystem shard pool with nslots writer slots.
// root: filesystem root path (keys are relative to this).
// unbuffered: use O_DIRECT for shard writes.
// Returns NULL on error.
struct shard_pool*
shard_pool_fs_create(const char* root, uint64_t nslots, int unbuffered);

// How a test puts a backend of its own in front of the filesystem one, to make
// requests fail or block. Every field may be left null.
struct shard_pool_fs_wrapper
{
  void* ctx;
  // Given the backend the pool built, return the one the queue should call.
  struct io_backend (*wrap)(void* ctx, struct io_backend inner);
  // Set to the queue the pool built, so a test can post requests of its own.
  struct io_queue** queue;
};

// Create a filesystem shard pool with a wrapper between its queue and its
// filesystem backend. shard_pool_fs_create passes an empty wrapper.
struct shard_pool*
shard_pool_fs_create_wrapped(const char* root,
                             uint64_t nslots,
                             int unbuffered,
                             struct shard_pool_fs_wrapper wrapper);

// Test helper: one-shot, fail the next truncate so a flush errors with IO
// still queued. Exercises the destroy-time drain that guards those buffers.
int
shard_pool_fs_inject_failing_truncate(struct shard_pool* pool);

// Test helper: mark the pool errored so later deliveries fail and stay queued.
void
shard_pool_fs_set_error(struct shard_pool* pool);
