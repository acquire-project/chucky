#pragma once

#include "writer.h"

#include <stdint.h>

struct io_queue;

struct io_queue*
io_queue_create(void);
void
io_queue_destroy(struct io_queue* q);

// Post a job to the queue. Returns 0 on success, non-zero on failure.
// On failure, the job is NOT posted; caller must free ctx.
// fn: the job function, called with ctx.
// ctx_free: if non-NULL, called with ctx after fn completes.
int
io_queue_post(struct io_queue* q,
              void (*fn)(void*),
              void* ctx,
              void (*ctx_free)(void*));

// Record an event capturing the current sequence number.
struct io_event
io_queue_record(struct io_queue* q);

// Block until all jobs up to and including ev.seq have completed.
void
io_event_wait(const struct io_queue* q, struct io_event ev);
