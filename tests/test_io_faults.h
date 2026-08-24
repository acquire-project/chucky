// This backend is for tests only: every request goes to the filesystem backend
// behind it, except IO_OP_NOOP, which can be made to fail or to block.
#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_queue;
struct shard_pool;
struct store;

struct io_faults
{
  struct io_backend inner;
  struct io_queue* queue;
  struct shard_pool* pool;

  // Both flags are one-shot and apply to the next IO_OP_NOOP.
  _Atomic int fail_next_noop;
  _Atomic int block_next_noop;
  _Atomic int* block_gate;
};

// Create a filesystem shard pool whose io can be made to fail or block.
struct shard_pool*
io_faults_pool_create(struct io_faults* f,
                      const char* root,
                      uint64_t nslots,
                      int unbuffered);

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
