#pragma once

#include "zarr/io_request.h"

#include <stdint.h>

// Each request handed to a backend is answered with one of these.
enum io_dispatch
{
  IO_DONE = 0,  // finished; *out is filled in
  IO_SUBMITTED, // finished later, through io_scheduler_complete
  IO_BUSY,      // not taken; handed over again
};

// Descriptors and syscalls live behind this; the scheduler owns admission,
// ordering and retirement. The creator owns the backend and outlives it.
struct io_backend
{
  void* ctx;

  // A request is carried out here. Every request has to be taken eventually,
  // and one taken but not finished stays good until its outcome is reported.
  int (*execute)(void* ctx,
                 const struct io_request* req,
                 uint64_t seq,
                 struct io_completion* out); // good only for this call

  // A backend's own thread is stopped here, after every request has finished.
  // This may be null.
  void (*stop)(void* ctx);
};
