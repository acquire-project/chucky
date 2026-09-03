#pragma once

#include "writer.h"
#include "zarr/io_backend.h"
#include "zarr/io_request.h"

#include <stdint.h>

struct io_scheduler;

struct io_scheduler_limits
{
  uint64_t max_requests;
  uint64_t max_bytes;
  uint64_t workers;
  uint64_t max_in_flight_per_file;
};

// The backend must carry an execute; without one there is nothing to run.
struct io_scheduler*
io_scheduler_create(struct io_backend backend,
                    struct io_scheduler_limits limits);

// Every other call must have returned, or be parked inside the queue already,
// before this runs. A thread still on its way in cannot be counted, and the
// lock it is about to take is one of the things freed here.
void
io_scheduler_destroy(struct io_scheduler* q);

// Threads parked in a blocking queue call are counted, the workers aside.
uint64_t
io_scheduler_parked_threads(const struct io_scheduler* q);

// Zero is returned on success; on failure nothing is posted and the payload
// is still yours.
int
io_scheduler_post(struct io_scheduler* q, struct io_request req);

// Bytes of unfinished posted work; an upper bound, never low.
uint64_t
io_scheduler_pending_bytes(const struct io_scheduler* q);

// Record an event capturing the current sequence number.
struct io_event
io_scheduler_record(struct io_scheduler* q);

// Block until all jobs up to and including ev.seq have completed.
void
io_event_wait(const struct io_scheduler* q, struct io_event ev);
