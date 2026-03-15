#pragma once

// Included via stream_internal.h — all internal types available above.

// H2D transfer + scatter into tile pool.
// pool_epoch: pointer to the target epoch's tile region in the pool.
// pool_ready: event to record after scatter completes.
// cursor: in/out, incremented by elements transferred.
// Returns 0 on success, non-zero on error.
int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct stream_layout* layout,
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
