#pragma once

#include "platform/platform_io.h"
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

// Take ownership of an open descriptor and name it. A zero generation means
// the descriptor was not taken and the caller still owns fd.
struct io_file_token
io_backend_fs_add_file(struct io_backend_fs* b, platform_fd fd);

uint64_t
io_backend_fs_files_opened(const struct io_backend_fs* b);

uint64_t
io_backend_fs_files_open_peak(const struct io_backend_fs* b);

// These test hooks are one-shot, each applying to the next op of its kind:
// IO_OP_NOOP for failure and block, IO_OP_TRUNCATE for the failing truncate.
// All three fail on the worker, where every real failure happens, so only a
// caller that waits can see them.
void
io_backend_fs_inject_failure(struct io_backend_fs* b);
void
io_backend_fs_inject_block(struct io_backend_fs* b, _Atomic int* gate);
void
io_backend_fs_inject_failing_truncate(struct io_backend_fs* b);

// Release a blocked request, for a queue torn down with no flush in front of
// it.
void
io_backend_fs_stop(struct io_backend_fs* b);
