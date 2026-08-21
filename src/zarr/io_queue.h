#pragma once

#include "zarr/types.io.h"
#include "writer.h"

#include <stdint.h>

struct io_queue;

struct io_work
{
  void (*fn)(void*);
  void* ctx;
  void (*ctx_free)(void*); // if non-NULL, called with ctx after fn returns
  uint64_t nbytes;         // payload size; 0 for truncate, close, and tests
  uint64_t file;           // which open file; 0 for none
  int borrowed;            // payload memory owned elsewhere
};

struct io_queue*
io_queue_create(void);
void
io_queue_destroy(struct io_queue* q);

// Zero on success; on failure nothing is posted and ctx is still yours.
int
io_queue_post(struct io_queue* q, struct io_work work);

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
