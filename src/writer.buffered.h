#pragma once

#include "writer.h"

struct buffered_writer;

// Byte limits and append sizes must respect downstream input granularity.
// Requires max_drain_bytes <= capacity_bytes.
struct buffered_writer_config
{
  size_t capacity_bytes;  // maximum pending bytes; nonzero
  size_t max_drain_bytes; // maximum batch bytes; 0 = capacity_bytes
};

struct buffered_writer_stats
{
  uint64_t accepted_bytes;
  uint64_t forwarded_bytes; // consumed by downstream, not necessarily persisted
  uint64_t abandoned_bytes; // accepted but not consumed after a terminal result
  size_t pending_bytes;     // queued + currently forwarding
  size_t peak_pending_bytes;
  uint64_t completed_batches;
  size_t max_batch_bytes;
  uint64_t backpressure_ns; // caller time waiting for free capacity
  uint64_t downstream_ns;   // sum per batch, including partial retries
  uint64_t max_downstream_ns;
  int downstream_finished;
  int failed; // sticky, including abandoned input and flush/close failures
};

// Borrows downstream and its dependencies until destroy; do not call downstream
// separately. It must provide append/flush, allow serialized calls from another
// thread, and retain no pointers to consumed input after append returns.
// Callbacks must not reenter the adapter. Serialize append, flush, close and
// destroy. Returns NULL on invalid arguments or resource failure.
struct buffered_writer*
buffered_writer_create(struct writer* downstream,
                       const struct buffered_writer_config* config);

// Returns an interface valid until destroy.
// append copies an input prefix and returns the unaccepted suffix in rest.
// Accepted input is immediately reusable. Append blocks when full, without
// a timeout. Input order is preserved; append boundaries may change.
//
// Downstream failure or completion stops acceptance. Remaining accepted bytes
// are abandoned and cause failure. Always check flush or destroy for errors.
//
// flush stops acceptance, drains unless downstream has terminated, and flushes
// downstream even after failure. close calls downstream close after flush and
// is a no-op before flush. Both wait for completion, are idempotent, and report
// sticky failures.
struct writer*
buffered_writer_as_writer(struct buffered_writer* buffered);

// Thread-safe snapshot; stop concurrent readers before destroy.
struct buffered_writer_stats
buffered_writer_get_stats(struct buffered_writer* buffered);

// Flushes, closes and frees the adapter, leaving downstream alive.
// Returns nonzero on failure. NULL succeeds.
int
buffered_writer_destroy(struct buffered_writer* buffered);
