// Filesystem-backed store implementation.
#pragma once

#include "zarr/store.h"

// Create a filesystem store rooted at the given directory.
// unbuffered: use O_DIRECT / FILE_FLAG_NO_BUFFERING for shard pool writers.
// Metadata writes create missing parent directories and atomically replace
// files for readers. They remain buffered and do not add fsync.
// Returns NULL on error.
struct store*
store_fs_create(const char* root, int unbuffered);
