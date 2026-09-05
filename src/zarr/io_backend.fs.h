#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;

// io_error is the pool's flag; it is raised on any failure.
struct io_backend_fs*
io_backend_fs_create(_Atomic int* io_error);

void
io_backend_fs_destroy(struct io_backend_fs* b);

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b);

// The path is copied; execution must follow the scheduler's per-file ordering.
struct io_file_token
io_backend_fs_reserve_file(struct io_backend_fs* b,
                           const char* path,
                           int open_flags);

// Only a reservation whose open was never posted may be cancelled.
void
io_backend_fs_cancel_file(struct io_backend_fs* b, struct io_file_token file);

// Counts include open calls that have not returned yet.
uint32_t
io_backend_fs_handle_count(const struct io_backend_fs* b);

uint32_t
io_backend_fs_peak_handle_count(const struct io_backend_fs* b);
