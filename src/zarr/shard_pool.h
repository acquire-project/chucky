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

  // Optional metadata publication. Copies key/data before returning and
  // atomically replaces the key after all previously accepted IO succeeds.
  // Queue insertion after backpressure defines the order. Errors are sticky
  // and flush waits for metadata as well as shard writes. NULL = synchronous.
  int (*queue_metadata)(struct shard_pool* self,
                        const char* key,
                        const void* data,
                        size_t len);

  // Wait for all pending I/O to complete. Returns non-zero on error.
  int (*flush)(struct shard_pool* self);

  // Returns non-zero if any I/O has failed.
  int (*has_error)(const struct shard_pool* self);

  // Bytes accepted but not yet written. A write counts from the moment it is
  // accepted until it lands, so the figure can read high but never low — a
  // caller deciding to slow down tolerates too high, never too low.
  uint64_t (*pending_bytes)(const struct shard_pool* self);

  // Required write alignment in bytes (e.g. page size for O_DIRECT).
  // NULL = no alignment constraint.
  size_t (*required_shard_alignment)(const struct shard_pool* self);

  void (*destroy)(struct shard_pool* self);
};

uint64_t
shard_pool_pending_bytes(const struct shard_pool* p);

size_t
shard_pool_required_shard_alignment(const struct shard_pool* p);

// NULL-safe destroy wrapper. Dispatches to p->destroy(p) when p is non-NULL.
void
shard_pool_destroy(struct shard_pool* p);
