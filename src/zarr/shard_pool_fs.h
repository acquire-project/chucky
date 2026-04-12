// Filesystem-backed shard writer pool.
// Uses io_queue for async pwrite with sequence-number fencing.
#pragma once

#include "zarr/shard_pool.h"

#include <stdint.h>

// Create a filesystem shard pool with nslots writer slots.
// root: filesystem root path (keys are relative to this).
// unbuffered: use O_DIRECT for shard writes.
// Returns NULL on error.
struct shard_pool*
shard_pool_fs_create(const char* root, uint64_t nslots, int unbuffered);

// Test helper: enqueue a job that unconditionally marks the pool as errored
// when it runs. Lets tests exercise the flush/has_error propagation path
// without depending on filesystem behavior. Returns 0 on successful enqueue.
int
shard_pool_fs_inject_failing_job(struct shard_pool* pool);
