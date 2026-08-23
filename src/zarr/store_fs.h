// Filesystem-backed store implementation.
#pragma once

#include "zarr/store.h"
#include "zarr/types.io.h"

// Create a filesystem store rooted at the given directory.
// unbuffered: use O_DIRECT / FILE_FLAG_NO_BUFFERING for shard pool writers.
// Returns NULL on error.
struct store*
store_fs_create(const char* root, int unbuffered);

// How much of the write backlog the shard pools created from here on run at
// once. A zero field takes the default.
void
store_fs_set_io_scheduling(struct store* s, struct io_scheduling io);
