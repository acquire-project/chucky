#pragma once

#include "zarr/io_request.h"

#include <stdint.h>

// Each request handed to a backend is answered with one of these.
enum io_dispatch
{
  IO_DONE = 0,  // finished; *out is filled in
  IO_SUBMITTED, // finished later, through io_queue_complete
  IO_BUSY,      // not taken; the queue hands it over again
};

// Descriptors and syscalls live behind this; the queue owns admission,
// ordering and retirement. The creator owns the backend and outlives the queue.
struct io_backend
{
  void* ctx;

  // The request outlives the call, so a backend answering IO_SUBMITTED can
  // keep reading it until it calls io_queue_complete; after that the queue
  // may reuse the slot. The completion does not outlive the call, and is good
  // only for as long as it runs. A backend with no room for the request
  // answers IO_BUSY, which keeps the request where it was in line, and one
  // that never takes a request keeps the queue from draining, so
  // io_queue_destroy never returns.
  //
  // A write that comes back short is finished here rather than by the queue,
  // by retrying what io_write_remaining hands back. Neither the count nor the
  // status is read by the queue, so a backend that gives up raises the error
  // flag it was given.
  int (*execute)(void* ctx,
                 const struct io_request* req,
                 uint64_t seq,
                 struct io_completion* out);

  // Called once by io_queue_destroy, after the last request has finished and
  // every worker has stopped, so work already handed over has to finish
  // without being asked to. A backend running a thread of its own stops it
  // here. This is optional, and a backend wrapping another passes it along.
  void (*stop)(void* ctx);
};
