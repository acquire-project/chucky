#pragma once

#include "zarr/io_request.h"

#include <stdint.h>

// Each request handed to a backend is answered with one of these.
enum io_dispatch
{
  IO_DONE = 0,  // finished; *out is filled in
  IO_SUBMITTED, // finished later, through io_queue_complete
};

// Descriptors and syscalls live behind this; the queue owns admission,
// ordering and retirement. The creator owns the backend and outlives the queue.
struct io_backend
{
  void* ctx;
  int (*execute)(void* ctx,
                 const struct io_request* req,
                 uint64_t seq,
                 struct io_completion* out);
};
