// Abstract shard writer pool.
// Manages a fixed number of reusable writer slots for streaming shard data.
// Handles async I/O lifecycle, fencing, and backpressure.
#pragma once

#include "writer.h"

#include <stddef.h>
#include <stdint.h>

struct shard_pool
{
  // Open writer slot for shard data at the given key.
  // If the slot has a pending finalize, waits for it first.
  struct shard_writer* (*open)(struct shard_pool* self,
                               uint64_t slot,
                               const char* key);

  // Record a fence capturing the current I/O sequence point.
  struct io_event (*record_fence)(struct shard_pool* self);

  // Block until all I/O up to ev has completed.
  void (*wait_fence)(struct shard_pool* self, struct io_event ev);

  // Wait for all pending I/O to complete. Returns non-zero on error.
  int (*flush)(struct shard_pool* self);

  // Returns non-zero if any I/O has failed.
  int (*has_error)(const struct shard_pool* self);

  // Returns number of bytes queued but not yet written.
  size_t (*pending_bytes)(const struct shard_pool* self);

  void (*destroy)(struct shard_pool* self);
};
