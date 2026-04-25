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

// Test helper: enqueue a job that spins until *gate becomes non-zero.
// Caller owns gate. Returns 0 on successful enqueue.
int
shard_pool_fs_inject_blocking_job(struct shard_pool* pool, _Atomic int* gate);

// Read counters of how many writes took the copy vs zero-copy path.
// Each *_bytes counter sums payload sizes seen on that path. Used by tests
// to assert that direct writes are taken when they're expected.
void
shard_pool_fs_path_counts(const struct shard_pool* pool,
                          uint64_t* out_copy_calls,
                          uint64_t* out_direct_calls,
                          uint64_t* out_copy_bytes,
                          uint64_t* out_direct_bytes);
