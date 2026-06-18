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
// `seed_stream` must be the same compute stream the epoch later runs on:
// the events are seeded on it so the first wait completes immediately, and
// seeding on an unrelated stream leaves those initial waits unsatisfied.
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

struct computed_stream_layouts;

// Sizing mirror of lod_state_init + lod_state_init_accumulators, for the
// memory estimate.
size_t
lod_state_device_bytes(const struct computed_stream_layouts* cl,
                       const struct tile_stream_configuration* config);

// Run LOD pipeline for one epoch: gather -> reduce -> append fold ->
// morton-to-chunks. pool_epoch: acquired view of this epoch's chunk pool
// region (all levels). *out_active_mask: set to bitmask of active LOD
// levels for this epoch. Returns 0 on success, non-zero on error.
int
lod_run_epoch(struct lod_state* lod,
              struct lod_shared_state* sh,
              struct gpu_ordering* ord,
              int fc,
              int timing_slot,
              const struct level_geometry* levels,
              struct gpu_pool_view pool_epoch,
              enum dtype dtype,
              enum lod_reduce_method reduce_method,
              enum lod_reduce_method append_reduce_method,
              const struct dim_info* dims,
              CUstream compute,
              uint32_t* out_active_mask);

// Bitmask of levels holding accumulated append-dim data.
uint32_t
lod_partial_append_mask(const struct lod_state* lod);

// Emit accumulated append-dim LOD data into the pool. pool0 is the base of
// the caller-acquired fill generation.
int
lod_emit_partial_append(struct lod_state* lod,
                        struct lod_shared_state* sh,
                        const struct level_geometry* levels,
                        const struct tile_stream_layout* layout,
                        enum dtype dtype,
                        enum lod_reduce_method append_reduce_method,
                        uint32_t active_levels_mask,
                        struct gpu_pool_view pool0,
                        CUstream compute);
