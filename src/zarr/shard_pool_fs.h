// Filesystem-backed shard writer pool.
// Uses io_queue for async pwrite with sequence-number fencing.
#pragma once

#include "zarr/shard_pool.h"

#include <stdint.h>

// Create a filesystem shard pool with nslots writer slots.
// root: filesystem root path (keys are relative to this).
// unbuffered: use O_DIRECT for shard writes.
// io: how much of the write backlog runs at once; NULL takes the defaults.
// Returns NULL on error.
struct shard_pool*
shard_pool_fs_create(const char* root,
                     uint64_t nslots,
                     int unbuffered,
                     const struct io_scheduling* io);

// Fill zero fields with the write-scheduling defaults, so a caller that has
// to record what it used has the numbers rather than the zeros.
void
shard_pool_fs_scheduling_defaults(struct io_scheduling* io);

// Test helper: enqueue a job that unconditionally marks the pool as errored
// when it runs. Lets tests exercise the flush/has_error propagation path
// without depending on filesystem behavior. Returns 0 on successful enqueue.
int
shard_pool_fs_inject_failing_job(struct shard_pool* pool);

// Test helper: enqueue a job that spins until *gate becomes non-zero.
// Caller owns gate. Returns 0 on successful enqueue.
int
shard_pool_fs_inject_blocking_job(struct shard_pool* pool, _Atomic int* gate);

// Test helper: one-shot, fail the next truncate so a flush errors with IO
// still queued. Exercises the destroy-time drain that guards those buffers.
int
shard_pool_fs_inject_failing_truncate(struct shard_pool* pool);

// Test helper: mark the pool errored so later deliveries fail and stay queued.
void
shard_pool_fs_set_error(struct shard_pool* pool);
