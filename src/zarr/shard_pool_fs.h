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

// A test's own backend can be called in place of the filesystem one, to make
// requests fail or block. Every field may be left null.
struct shard_pool_fs_wrapper
{
  void* ctx;
  // The pool's backend is the argument, and the queue calls the result.
  struct io_backend (*wrap)(void* ctx, struct io_backend inner);
  struct io_queue** queue; // receives the queue the pool built
};

// Create a filesystem shard pool with the wrapper between its queue and its
// filesystem backend.
struct shard_pool*
shard_pool_fs_create_wrapped(const char* root,
                             uint64_t nslots,
                             int unbuffered,
                             struct shard_pool_fs_wrapper wrapper);

// Test helper: mark the pool errored so later deliveries fail and stay queued.
void
shard_pool_fs_set_error(struct shard_pool* pool);
