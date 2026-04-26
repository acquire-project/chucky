// Private zarr array interface.
// Adds pool-borrowing variant for internal use by ngff/hcs layers.
#pragma once

#include "zarr.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"

// Private: create a zarr array that borrows an existing pool.
// The caller owns the pool lifetime — zarr_array_destroy will NOT destroy it.
// slot_base: first pool slot used by this array (avoids collisions when
//            multiple arrays share a pool).
struct zarr_array*
zarr_array_create_with_pool(struct store* store,
                            struct shard_pool* pool,
                            uint64_t slot_base,
                            const char* prefix,
                            const struct zarr_array_config* cfg);
