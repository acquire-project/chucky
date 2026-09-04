#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;

#define IO_BACKEND_FS_MAX_OPEN_FILES 64u

// io_error is the pool's flag; it is raised on any failure.
struct io_backend_fs*
io_backend_fs_create(_Atomic int* io_error);

void
io_backend_fs_destroy(struct io_backend_fs* b);

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b);

struct io_file_token
io_backend_fs_reserve_file(struct io_backend_fs* b);

void
io_backend_fs_cancel_file(struct io_backend_fs* b, struct io_file_token file);

uint32_t
io_backend_fs_handle_count(const struct io_backend_fs* b);

uint32_t
io_backend_fs_peak_handle_count(const struct io_backend_fs* b);
