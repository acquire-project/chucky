#pragma once

#include "gpu/stream.internal.h"

struct threadpool;
struct stream_metric;

// memcpy into pinned staging, split across the pool when the payload is
// large enough to pay for the dispatch. The destination lies within one
// acquired h_in generation owned by the caller, so no ordering is queued.
void
ingest_copy(struct threadpool* pool, void* dst, const void* src, size_t n);

// Allocate double-buffered staging buffers and timing events; ordering
// events live in ord. Returns 0 on success.
int
ingest_init(struct staging_state* stage,
            size_t buffer_capacity_bytes,
            struct gpu_ordering* ord,
            CUstream compute);

// Free staging buffers and events. The caller synchronizes the streams first:
// a copy still reading a staging buffer faults once it is freed. Safe to call
// twice, and on a state whose setup failed.
void
ingest_destroy(struct staging_state* stage);

// Fold every finished scatter measurement into m, leaving the unfinished ones
// for a later call. Never blocks. Call from the append loop and again at flush
// so the last dispatches are not left unread.
void
ingest_collect_scatter_timing(struct staging_state* stage,
                              struct stream_metric* m);

// Same for the per-slot H2D intervals. Cheap at slot reacquire, where the
// acquire has already waited on the interval's end; the flush call picks up the
// final dispatches, which no reacquire ever revisits.
void
ingest_collect_h2d_timing(struct staging_state* stage, struct stream_metric* m);

// One dispatch's data is written to this region of the chunk pool.
struct scatter_destination
{
  struct gpu_pool_view first_epoch; // region for the epoch holding the first
                                    // element, within the produce generation
                                    // the caller acquired
  size_t epoch_bytes;               // one epoch's region in the pool
  uint32_t epochs; // regions the caller has, counting from first_epoch
};

// H2D transfer + scatter into chunk pool.
// first_element: append-cursor position of the buffer's first element.
// Returns 0 on success, non-zero on error.
int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct tile_stream_layout* layout,
                        struct scatter_destination dst,
                        uint64_t first_element,
                        size_t bpe,
                        CUstream h2d,
                        CUstream compute);

// H2D transfer + copy to linear epoch buffer for LOD.
// L0 tiling is deferred to run_lod.
// d_linear: device pointer to the linear epoch buffer, holding one epoch.
// epoch_elements: elements per epoch (layout.epoch_elements).
// first_element: append-cursor position of the buffer's first element. The
// staged bytes may not cross an epoch boundary.
// Returns 0 on success, non-zero on error.
int
ingest_dispatch_multiscale(struct staging_state* stage,
                           CUdeviceptr d_linear,
                           uint64_t epoch_elements,
                           uint64_t first_element,
                           size_t bpe,
                           CUstream h2d,
                           CUstream compute);
