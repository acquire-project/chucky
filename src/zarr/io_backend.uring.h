// Writes are handed to a kernel ring here, so no worker thread is held while
// one runs. Linux only, and only where liburing was found at build time;
// everywhere else no ring can be had and the filesystem backend writes.
#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;
struct io_backend_uring;
struct io_queue;

// A ring bigger than this is turned down by the kernel.
#define IO_BACKEND_URING_MAX_DEPTH 32768u

// Non-zero when a ring can be had on this machine. A refusal is reported once
// and then remembered.
int
io_backend_uring_supported(void);

// Create a ring holding depth writes at once, capped at the ceiling above, or
// NULL when no ring can be had. The file table and everything that is not a
// write are the filesystem backend's.
struct io_backend_uring*
io_backend_uring_create(struct io_backend_fs* files,
                        _Atomic int* io_error, // raised on any failure
                        uint64_t depth);

// Report finished writes to this queue from here on. Non-zero is returned when
// the thread that reports them could not be started; the queue is built from
// the backend, so it cannot be named any earlier.
int
io_backend_uring_start(struct io_backend_uring* b, struct io_queue* queue);

void
io_backend_uring_destroy(struct io_backend_uring* b);

struct io_backend
io_backend_uring_as_backend(struct io_backend_uring* b);
