#pragma once

#include "zarr/io_backend.h"

#include <stdint.h>

struct io_queue;

struct io_queue_limits
{
  uint64_t max_requests;
  uint64_t workers;
};

struct io_queue_observer
{
  void* ctx;
  void (*started)(void* ctx, uint64_t seq);
  void (*busy)(void* ctx, uint64_t seq);
  void (*finished)(void* ctx, struct io_completion completion);
};

struct io_queue*
io_queue_create(struct io_backend backend,
                struct io_queue_limits limits,
                struct io_queue_observer observer);

void
io_queue_destroy(struct io_queue* queue);

int
io_queue_post(struct io_queue* queue,
              const struct io_request* request,
              uint64_t seq);

void
io_queue_complete(struct io_queue* queue, struct io_completion completion);
