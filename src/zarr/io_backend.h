#pragma once

#include "zarr/io_request.h"

#include <stdint.h>

// Each request handed to a backend is answered with one of these.
enum io_dispatch
{
  IO_DONE = 0,  // finished; *out is filled in
  IO_SUBMITTED, // finished later, through io_queue_complete
  IO_BUSY,      // not taken; handed over again
};

// Descriptors and syscalls live behind this; the queue owns admission,
// ordering and retirement. The creator owns the backend and outlives the queue.
struct io_backend
{
  void* ctx;

  // A request taken but not finished is good until its outcome is reported.
  // The completion is good only for the length of the call. Every request has
  // to be taken eventually. A write that moved fewer bytes than asked is the
  // backend's to finish, not the queue's. Neither the count nor the status is
  // read, so a backend that gives up has to raise the error flag itself.
  int (*execute)(void* ctx,
                 const struct io_request* req,
                 uint64_t seq,
                 struct io_completion* out);

  // A backend's own thread is stopped here, after every request has finished.
  // This may be null.
  void (*stop)(void* ctx);
};
