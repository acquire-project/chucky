// Filesystem-backed store implementation.
#pragma once

#include "zarr/store.h"

// Create a filesystem store rooted at the given directory.
// unbuffered: use O_DIRECT / FILE_FLAG_NO_BUFFERING for shard pool writers.
// Returns NULL on error.
struct store*
store_fs_create(const char* root, int unbuffered);
