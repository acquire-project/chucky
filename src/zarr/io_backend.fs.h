#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;

// io_error is the pool's flag; it is raised on any failure.
struct io_backend_fs*
io_backend_fs_create(_Atomic int* io_error, int open_flags);

void
io_backend_fs_destroy(struct io_backend_fs* b);

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b);

// Execution must follow the scheduler's per-file ordering.
struct io_file_token
io_backend_fs_reserve_file(struct io_backend_fs* b);

// Only a reservation whose open was never posted may be cancelled.
void
io_backend_fs_cancel_file(struct io_backend_fs* b, struct io_file_token file);

// Shard handle counts include open calls that have not returned yet.
// Metadata replacement uses a separate short-lived buffered handle.
uint32_t
io_backend_fs_handle_count(const struct io_backend_fs* b);

uint32_t
io_backend_fs_peak_handle_count(const struct io_backend_fs* b);
