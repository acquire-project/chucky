#pragma once

#include "writer.h"
#include "zarr/io_backend.h"
#include "zarr/io_request.h"
#include "zarr/types.io.h"

#include <stdint.h>

struct io_queue;

// Ceilings on what may sit in the queue at once.
struct io_queue_limits
{
  uint64_t max_requests; // 0 selects the default, 1024
  uint64_t max_bytes;    // 0 means no ceiling
};

// A zeroed backend runs closures only.
struct io_queue*
io_queue_create(struct io_backend backend, struct io_queue_limits limits);
void
io_queue_destroy(struct io_queue* q);

// Zero on success; on failure nothing is posted and ctx is still yours.
int
io_queue_post(struct io_queue* q, struct io_request req);

// Claim room for a request. Waits until there is room. Zero on success;
// non-zero if the queue is shutting down or the file is already closing, and
// then nothing is claimed. Fill in op, file and nbytes; the payload can come
// later, at commit.
int
io_queue_reserve(struct io_queue* q, struct io_request req);

// Post a request whose room io_queue_reserve already claimed. The claim
// guarantees a slot, so this cannot fail. req.nbytes must equal the nbytes
// passed to the matching reserve.
void
io_queue_commit(struct io_queue* q, struct io_request req);

// Give back a claim that will not be committed.
void
io_queue_release(struct io_queue* q, uint64_t nbytes);

// Report the outcome of a request the backend answered with IO_SUBMITTED.
// Safe to call from any thread.
void
io_queue_complete(struct io_queue* q, struct io_completion c);

// Bytes of unfinished posted work; an upper bound, never low.
uint64_t
io_queue_pending_bytes(const struct io_queue* q);

// Copy out what has been measured so far.
void
io_queue_get_stats(const struct io_queue* q, struct io_queue_stats* out);

// Record an event capturing the current sequence number.
struct io_event
io_queue_record(struct io_queue* q);

// Block until all jobs up to and including ev.seq have completed.
void
io_event_wait(const struct io_queue* q, struct io_event ev);

// Returns non-zero once io_queue_destroy has been called on the queue.
// Long-running jobs can poll this to bail out early on shutdown.
int
io_queue_is_shutdown(const struct io_queue* q);
