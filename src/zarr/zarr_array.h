// Private zarr array interface.
// Adds pool-borrowing variant for internal use by ngff/hcs layers.
#pragma once

#include "zarr.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"

// Private: create a zarr array that borrows an existing pool.
// The caller owns the pool lifetime — zarr_array_destroy will NOT destroy it.
struct zarr_array*
zarr_array_create_with_pool(struct store* store,
                            struct shard_pool* pool,
                            const char* prefix,
                            const struct zarr_array_config* cfg);
