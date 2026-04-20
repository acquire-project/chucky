#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

// Upload level layouts to GPU (always, including L0). When multiscale is
// enabled, also uploads LOD plan shapes and builds scatter/reduce LUTs.
// Plan and level layouts must already be populated in lod->plan and
// lod->layouts (from compute_stream_layouts). Sets levels->nlod.
// Returns 0 on success.
int
lod_state_init(struct lod_state* lod,
               struct level_geometry* levels,
               const struct tile_stream_configuration* config);

// Allocate the engine-owned shared LOD resources (d_linear, d_morton, timing).
// linear_bytes / morton_bytes are the buffer sizes (multiarray passes the max
// across arrays; single-array passes the array's own sizes).
//
// The timing events are created AND seeded on `seed_stream` so the first
// downstream wait completes immediately.  `seed_stream` must be the same
// compute stream that later runs lod_run_epoch — seeding on an unrelated
// stream leaves the initial waits unsatisfied.
//
// Returns 0 on success; on failure, the struct is left safe to pass to
// lod_shared_state_destroy.
int
lod_shared_state_init(struct lod_shared_state* sh,
                      size_t linear_bytes,
                      size_t morton_bytes,
                      CUstream seed_stream);

// Free the engine-owned shared LOD resources.
void
lod_shared_state_destroy(struct lod_shared_state* sh);

// Allocate append-dim accumulators, level-ID buffer, and counts.
// Must be called AFTER lod_state_init.
// Returns 0 on success.
int
lod_state_init_accumulators(struct lod_state* lod,
                            const struct tile_stream_configuration* config);

// Free all LOD device allocations and plan.
void
lod_state_destroy(struct lod_state* lod);

// Run LOD pipeline for one epoch: gather -> reduce -> append fold ->
// morton-to-chunks. pool_epoch: pointer to this epoch's chunk pool region (all
// levels). *out_active_mask: set to bitmask of active LOD levels for this
// epoch. Returns 0 on success, non-zero on error.
int
lod_run_epoch(struct lod_state* lod,
              struct lod_shared_state* sh,
              int fc,
              const struct level_geometry* levels,
              void* pool_epoch,
              enum dtype dtype,
              enum lod_reduce_method reduce_method,
              enum lod_reduce_method append_reduce_method,
              const struct dim_info* dims,
              CUstream compute,
              uint32_t* out_active_mask);
