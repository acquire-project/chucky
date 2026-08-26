// A write backend that hands each write to a kernel ring, so no worker thread
// is held while the write runs. Linux only, and only where liburing was found
// at build time; everywhere else the calls below report that no ring can be
// had and the filesystem backend is used instead.
#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;
struct io_backend_uring;
struct io_queue;

// Non-zero when a ring can be had on this machine. A refusal is reported once
// and then remembered.
int
io_backend_uring_supported(void);

// Create a ring that holds depth writes at once, or NULL when it cannot be
// had. The file table is the filesystem backend's, and so is every request
// that is not a write. io_error is the pool's flag, raised on any failure.
struct io_backend_uring*
io_backend_uring_create(struct io_backend_fs* files,
                        _Atomic int* io_error,
                        uint64_t depth);

// Report finished writes to this queue from here on. Non-zero is returned
// when the thread that reports them could not be started. The queue is built
// from the backend, so it cannot be named before this point.
int
io_backend_uring_start(struct io_backend_uring* b, struct io_queue* queue);

void
io_backend_uring_destroy(struct io_backend_uring* b);

struct io_backend
io_backend_uring_as_backend(struct io_backend_uring* b);
