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

// Submit an append-shape snapshot without waiting for queued IO. Used by NGFF
// to submit the child and group before one shared zarr_metadata_wait. Immediate
// rejection restores dimensions; accepted snapshots remain the current state
// even if completion later fails. Non-queue pools complete synchronously.
int
zarr_array_submit_append(struct zarr_array* a,
                         uint8_t n_append,
                         const uint64_t* append_sizes);

// Private: force the array's zarr.json to be rewritten now with current shape
// and buffered attributes. Production callers go through writer_close, which
// drives the sink-level flush hook; this remains exposed for low-level tests
// that exercise zarr_array without a stream.
int
zarr_array_flush_metadata(struct zarr_array* a);
