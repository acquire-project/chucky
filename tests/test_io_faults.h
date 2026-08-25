// This backend is for tests only. Every request goes to the filesystem backend
// behind it. One armed request can be held at a gate first, or made to fail.
#pragma once

#include "zarr/io_backend.h"
#include "zarr/types.io.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_queue;
struct shard_pool;
struct store;

enum io_fault
{
  IO_FAULT_NONE = 0,
  IO_FAULT_FAIL,  // report the request failed and mark the pool errored
  IO_FAULT_BLOCK, // hold the request until the gate opens
};

struct io_faults
{
  struct io_backend inner;
  struct io_queue* queue;
  struct shard_pool* pool;

  // One fault is armed at a time. The op is in the high byte and the fault in
  // the low, so that a request reads both as one value.
  _Atomic uint16_t armed;
  _Atomic int* block_gate;

  struct io_scheduling io; // zero fields take the pool's own defaults
};

// Create a filesystem shard pool whose io can be made to fail or block.
// io: how much of the write backlog runs at once; NULL takes the defaults.
struct shard_pool*
io_faults_pool_create(struct io_faults* f,
                      const char* root,
                      uint64_t nslots,
                      int unbuffered,
                      const struct io_scheduling* io);

// Set the write scheduling a later pool is built with. A store builds its pool
// after it is created, so this is how that pool's scheduling is chosen.
void
io_faults_set_io_scheduling(struct io_faults* f, struct io_scheduling io);

// Create a filesystem store whose pool can be made to fail or block. Only one
// pool can be built from it, even after that pool is destroyed.
struct store*
io_faults_store_create(struct io_faults* f, const char* root, int unbuffered);

// Queue a job that marks the pool errored when it runs. Returns 0 on a
// successful enqueue.
int
io_faults_inject_failing_job(struct io_faults* f);

// Queue a job that spins until *gate is non-zero. The caller owns the gate and
// must open it before tearing the pool down. Returns 0 on a successful
// enqueue.
int
io_faults_inject_blocking_job(struct io_faults* f, _Atomic int* gate);

// Fail the next truncate. Nothing is queued here: the failure lands when the
// worker reaches a truncate the caller's own code posted.
void
io_faults_fail_next_truncate(struct io_faults* f);
