// Private ngff_multiscale interface.
// Adds pool-borrowing variant for internal use by hcs layer.
#pragma once

#include "ngff.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"

// Private: create a multiscale sink that borrows an existing pool.
// The caller owns the pool lifetime — ngff_multiscale_destroy will NOT
// destroy it.
struct ngff_multiscale*
ngff_multiscale_create_with_pool(struct store* store,
                                 struct shard_pool* pool,
                                 const char* prefix,
                                 const struct ngff_multiscale_config* cfg);
