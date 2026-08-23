#pragma once

#include "writer.h"
#include "zarr/io_backend.h"
#include "zarr/io_request.h"
#include "zarr/types.io.h"

#include <stdint.h>

struct io_queue;

// These are ceilings on what sits in the queue at once, and on how much of it
// runs together.
struct io_queue_limits
{
  uint64_t max_requests; // 0 selects the default, 1024
  uint64_t max_bytes;    // 0 means no ceiling
  uint64_t workers;      // 0 selects one

  // Requests handed to the backend and not yet finished. A blocking backend
  // cannot exceed its worker count whatever this says.
  uint64_t writes_in_flight;          // 0 means no ceiling
  uint64_t writes_in_flight_per_file; // 0 selects one
};

// The backend must carry an execute; without one there is nothing to run.
struct io_queue*
io_queue_create(struct io_backend backend, struct io_queue_limits limits);

// Every other call must have returned, or be parked inside the queue already,
// before this runs. A thread still on its way in cannot be counted, and the
// lock it is about to take is one of the things freed here.
void
io_queue_destroy(struct io_queue* q);

// Threads parked in a blocking queue call are counted, the workers aside.
uint64_t
io_queue_parked_threads(const struct io_queue* q);

// Zero is returned on success; on failure nothing is posted and the payload
// is still yours.
int
io_queue_post(struct io_queue* q, struct io_request req);

// Claim room for a request, waiting until there is room. Zero is returned on
// success, non-zero if the queue is shutting down or the file is already
// closing, and then nothing is claimed. Fill in op, file and nbytes; the
// payload can come later, at commit.
int
io_queue_reserve(struct io_queue* q, struct io_request req);

// Post a request whose room was already claimed by io_queue_reserve. A slot
// and a place in the open file table are guaranteed by the claim, so this
// cannot fail. req.nbytes and req.file must be the ones passed to the
// matching reserve.
void
io_queue_commit(struct io_queue* q, struct io_request req);

// Give back a claim that will not be committed.
void
io_queue_release(struct io_queue* q, uint64_t nbytes);

// Report the outcome of a request that was answered with IO_SUBMITTED. Any
// thread may call this.
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
