#pragma once

#include "writer.h"

struct buffered_writer;

struct buffered_writer_config
{
  size_t slot_bytes; // maximum bytes accepted by one append; nonzero
  size_t slot_count; // includes the slot being forwarded; nonzero
};

struct buffered_writer_stats
{
  uint64_t accepted_bytes;
  uint64_t forwarded_bytes; // consumed by downstream, not necessarily persisted
  uint64_t abandoned_bytes; // accepted but not consumed after a terminal result
  size_t pending_bytes;     // queued + currently forwarding
  size_t peak_pending_bytes;
  size_t occupied_slots; // includes the slot being forwarded
  size_t peak_occupied_slots;
  uint64_t completed_slots;
  uint64_t backpressure_ns; // caller time waiting for a free slot
  uint64_t queue_ns;        // sum: acceptance to start of downstream append
  uint64_t max_queue_ns;
  uint64_t downstream_ns; // sum: downstream append, including partial retries
  uint64_t max_downstream_ns;
  uint64_t max_completion_ns; // acceptance to downstream append return
  int downstream_finished;
  int failed; // sticky, including abandoned input and flush/close failures
};

// A bounded, copying adapter. Borrows downstream, which (along with its sink,
// configuration and GPU context) must outlive the adapter. Only the worker
// calls downstream; do not access it separately until destroying the adapter.
// The downstream writer must permit serialized calls from a different thread
// and release its reference to consumed input before append returns. CPU/GPU
// tile-stream writers satisfy this contract.
//
// Externally serialize append/flush/close/destroy calls. get_stats may run
// concurrently. Downstream callbacks must not reenter the adapter.
//
// Payload allocation is exactly slot_count * slot_bytes, plus fixed bookkeeping
// per slot and one worker thread. No per-append allocation. slot_bytes and
// input sizes must respect the downstream input granularity (e.g. whole
// elements). Returns NULL on invalid config, allocation failure or thread-start
// failure.
struct buffered_writer*
buffered_writer_create(struct writer* downstream,
                       const struct buffered_writer_config* config);

// append waits for one free slot, copies at most slot_bytes, and returns the
// unaccepted suffix in rest. The accepted prefix can immediately be reused.
// An empty rest can be {NULL,NULL}. Use writer_append_wait for larger inputs.
// Acceptance is not downstream completion; inspect flush's result even when
// every append succeeded. Backpressure blocks without a timeout or drops.
//
// A downstream failure stops forwarding and abandons the unconsumed accepted
// suffix, counted in stats. Early downstream `finished` does the same: if any
// accepted bytes remain, the adapter reports fail, otherwise finished. Future
// appends accept nothing. No borrowed/copy-buffer pointer is returned in rest.
// A downstream writer that makes no progress fails after writer_append_wait's
// bounded retries.
//
// flush stops acceptance, drains the queue (unless downstream terminates), then
// flushes downstream even after failure. close publishes downstream metadata
// after flush; before flush it is a no-op, as with the tile-stream writers.
// Both wait for completion, are idempotent, and report sticky failures.
struct writer*
buffered_writer_as_writer(struct buffered_writer* buffered);

struct buffered_writer_stats
buffered_writer_get_stats(struct buffered_writer* buffered);

// Flush, close, join and release the adapter, never downstream. Returns nonzero
// on any failure, including abandoned input. NULL is allowed and succeeds.
int
buffered_writer_destroy(struct buffered_writer* buffered);
