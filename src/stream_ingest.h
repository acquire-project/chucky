#pragma once

#include "stream_internal.h"

// Allocate double-buffered staging buffers and events. Returns 0 on success.
int
ingest_init(struct staging_state* stage,
            size_t buffer_capacity_bytes,
            CUstream compute);

// Free staging buffers and events.
void
ingest_destroy(struct staging_state* stage);

// H2D transfer + scatter into chunk pool.
// pool_epoch: pointer to the target epoch's chunk region in the pool.
// pool_ready: event to record after scatter completes.
// cursor: in/out, incremented by elements transferred.
// Returns 0 on success, non-zero on error.
int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct tile_stream_layout* layout,
                        const struct tile_stream_layout_gpu* layout_gpu,
                        void* pool_epoch,
                        CUevent pool_ready,
                        uint64_t* cursor,
                        size_t bpe,
                        CUstream h2d,
                        CUstream compute);

// H2D transfer + copy to linear epoch buffer for LOD.
// L0 tiling is deferred to run_lod.
// d_linear: device pointer to the linear epoch buffer.
// epoch_elements: elements per epoch (layout.epoch_elements).
// cursor: in/out, incremented by elements transferred.
// Returns 0 on success, non-zero on error.
int
ingest_dispatch_multiscale(struct staging_state* stage,
                           CUdeviceptr d_linear,
                           uint64_t epoch_elements,
                           uint64_t* cursor,
                           size_t bpe,
                           CUstream h2d,
                           CUstream compute);
