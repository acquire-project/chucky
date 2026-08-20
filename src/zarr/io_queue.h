#pragma once

#include "types.io.h"
#include "writer.h"

#include <stdint.h>

struct io_queue;

// One piece of work for the queue to run.
struct io_work
{
  void (*fn)(void*);
  void* ctx;
  void (*ctx_free)(void*); // if non-NULL, called with ctx after fn returns
  uint64_t nbytes;         // payload size; 0 for truncate, close, and tests
  uint64_t file;           // which open file this belongs to; 0 for none
  int borrowed;            // payload is memory the job does not own
};

struct io_queue*
io_queue_create(void);
void
io_queue_destroy(struct io_queue* q);

// Post work to the queue. Returns 0 on success, non-zero on failure.
// On failure the work is NOT posted and the caller still owns ctx.
int
io_queue_post(struct io_queue* q, struct io_work work);

// Bytes carried by posted work that has not finished. An upper bound: a
// write counts from the moment the queue takes it, so this can read high
// but never low.
uint64_t
io_queue_pending_bytes(const struct io_queue* q);

// Copy out what the queue has measured so far.
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
