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

// Test helper: one-shot, fail the next truncate so a flush errors with IO
// still queued. Exercises the destroy-time drain that guards those buffers.
int
shard_pool_fs_inject_failing_truncate(struct shard_pool* pool);

// Test helper: mark the pool errored so later deliveries fail and stay queued.
void
shard_pool_fs_set_error(struct shard_pool* pool);

// Test helper: park writers partway through queueing a write, between counting
// the bytes and handing the job to the worker. Lets a test read pending_bytes
// at the one point where the count and the queued work disagree.
void
shard_pool_fs_pause_mid_write(struct shard_pool* pool, int paused);
