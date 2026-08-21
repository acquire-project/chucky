#pragma once

#include "platform/platform_io.h"
#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;

// io_error is the pool's flag; the backend raises it on any failure.
struct io_backend_fs*
io_backend_fs_create(_Atomic int* io_error);

void
io_backend_fs_destroy(struct io_backend_fs* b);

struct io_backend
io_backend_fs_as_backend(struct io_backend_fs* b);

// Take ownership of an open descriptor and name it. A zero generation means
// the registry could not take it and the caller still owns fd.
struct io_file_token
io_backend_fs_add_file(struct io_backend_fs* b, platform_fd fd);

uint64_t
io_backend_fs_files_opened(const struct io_backend_fs* b);

uint64_t
io_backend_fs_files_open_peak(const struct io_backend_fs* b);
