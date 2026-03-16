#pragma once

#include "stream_internal.h"

// Upload pre-computed LOD plan/layouts to GPU and build scatter LUTs.
// Plan and level layouts must already be populated in lod->plan and
// lod->layouts (from compute_stream_layouts). Sets levels->nlod.
// Returns 0 on success.
int
lod_state_init(struct lod_state* lod,
               struct level_geometry* levels,
               const struct stream_layout* l0,
               const struct tile_stream_configuration* config);

// Allocate d_linear, d_morton, LOD timing events.
// Must be called AFTER lod_state_init.
// Returns 0 on success.
int
lod_state_init_buffers(struct lod_state* lod,
                       const struct stream_layout* l0,
                       enum lod_dtype dtype);

// Allocate dim0 accumulators, level-ID buffer, and counts.
// Must be called AFTER lod_state_init.
// Returns 0 on success.
int
lod_state_init_accumulators(struct lod_state* lod,
                            const struct tile_stream_configuration* config);

// Free all LOD device allocations and plan.
void
lod_state_destroy(struct lod_state* lod);

// Run LOD pipeline for one epoch: gather -> reduce -> dim0 fold ->
// morton-to-tiles. pool_epoch: pointer to this epoch's tile pool region (all
// levels). *out_active_mask: set to bitmask of active LOD levels for this
// epoch. Returns 0 on success, non-zero on error.
int
lod_run_epoch(struct lod_state* lod,
              const struct level_geometry* levels,
              const struct stream_layout* layout,
              void* pool_epoch,
              enum lod_dtype dtype,
              enum lod_reduce_method reduce_method,
              enum lod_reduce_method dim0_reduce_method,
              CUstream compute,
              uint32_t* out_active_mask);
