#pragma once

#include "gpu/aggregate.h"
#include "gpu/compress.h"
#include "gpu/flush.handoff.h"
#include "gpu/reduce_csr_gpu.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "zarr/shard_delivery.h"
#include <stddef.h>

// --- Sub-struct definitions (shared between engine and internal headers) ---

struct pool_state
{
  CUdeviceptr buf[2];
  CUevent ready[2];
  int current; // 0 or 1
};

struct staging_slot
{
  void* h_in;              // pinned host WC, size = buffer_capacity_bytes
  CUdeviceptr d_in;        // device, size = buffer_capacity_bytes
  CUevent t_h2d_end;       // recorded after H2D memcpy completes
  CUevent t_h2d_start;     // recorded before H2D memcpy
  CUevent t_scatter_start; // recorded before scatter kernel
  CUevent t_scatter_end;   // recorded after scatter kernel
  size_t dispatched_bytes; // bytes transferred in last dispatch
};

struct staging_state
{
  struct staging_slot slot[2];
  int current;          // 0 or 1: which buffer the host is filling
  size_t bytes_written; // bytes written to current slot's h_in so far
};

// Per flush-slot: mutable batch state (masks + epoch count).
// batch_active_masks is heap-allocated to epochs_per_batch entries.
struct flush_slot_gpu
{
  uint32_t active_levels_mask; // union of per-epoch active masks
  uint32_t* batch_active_masks; // [K] per-epoch active level masks
  int batch_epoch_count; // number of epochs accumulated in this batch
};

struct level_flush_state
{
  struct aggregate_layout agg_layout;
  struct aggregate_slot agg[2]; // double-buffered, indexed by flush_current
  struct shard_state shard;
  CUdeviceptr
    d_batch_gather; // [K_l * M_l] uint32: batch-chunk -> compressed idx
  CUdeviceptr
    d_batch_perm; // [K_l * M_l] uint32: batch-chunk -> shard-ordered pos
  uint32_t batch_active_count; // K_l = K / 2^l for this level
};

// Per-frame-counter timing events (double-buffered).
struct lod_timing
{
  CUevent t_start;
  CUevent t_scatter_end;
  CUevent t_reduce_end;
  CUevent t_append_end;
  CUevent t_end;
};

// Engine-owned LOD resources shared across all arrays in a multiarray stream
// (sized to the max requirement).  For a single-array stream these are still
// owned by the engine and sized for that one array.
struct lod_shared_state
{
  CUdeviceptr d_linear;        // linear epoch buffer (device)
  CUdeviceptr d_morton;        // morton-ordered LOD output (all levels packed)
  struct lod_timing timing[2]; // double-buffered pipeline timing
};

// Per-array LOD state.  In multiarray, one instance per array; the active
// array's state is copied into the engine on bind.
struct lod_state
{
  struct lod_plan plan;

  CUdeviceptr d_full_shape;         // device copy of shapes[0]
  CUdeviceptr d_lod_shape;          // device copy of LOD-projected shapes[0]
  CUdeviceptr d_gather_lut;         // u32, lod_nelem[0] entries
  CUdeviceptr d_fixed_dims_offsets; // u32, fixed_dims_count entries

  // CSR reduce LUTs (precomputed, one per level transition).
  struct reduce_csr_gpu csrs[LOD_MAX_LEVELS];

  // Per-level chunk layouts [0..nlod-1]
  struct tile_stream_layout layouts[LOD_MAX_LEVELS];
  struct tile_stream_layout_gpu layout_gpu[LOD_MAX_LEVELS];

  // Morton-to-chunk scatter LUTs (precomputed)
  CUdeviceptr d_morton_chunk_lut[LOD_MAX_LEVELS];
  CUdeviceptr d_morton_fixed_dims_chunk_offsets[LOD_MAX_LEVELS];

  // Append-dim LOD accumulation state.
  struct
  {
    CUdeviceptr d_accum;
    CUdeviceptr d_level_ids;
    CUdeviceptr d_counts;
    uint32_t counts[LOD_MAX_LEVELS];
    uint64_t element_capacity;
    uint64_t morton_offset;
  } append_accum;
};

// CUDA stream handles (all immutable after create)
struct gpu_streams
{
  CUstream h2d, compute, compress, d2h;
};

// Batch accumulation: config + mutable counter + single pool-ready event.
// All K scatter ops run on the compute stream in order, so a single event
// recorded after the K-th scatter subsumes all per-epoch ready signals.
struct batch_state
{
  uint32_t epochs_per_batch; // K (immutable after create)
  uint32_t accumulated;      // mutable: 0..K-1
  CUevent pool_ready;        // recorded on compute after last accumulated epoch
};

// --- Stage types ---

struct compress_agg_input
{
  int fc;
  uint32_t n_epochs;
  uint32_t active_levels_mask;
  const uint32_t* batch_active_masks; // borrowed from flush_slot_gpu [K]
  CUdeviceptr pool_buf;
  CUevent pool_ready;    // batch-level (from batch_state.pool_ready)
  CUevent lod_done;
  CUevent prev_d2h_done; // prior D2H on same fc; compress waits on this so
                         // aggregate doesn't overwrite agg[fc].d_aggregated
                         // before the prior D2H has read it. Initialized
                         // signaled.
  uint32_t epochs_per_batch;
};

struct compress_agg_stage
{
  struct codec codec;
  CUdeviceptr d_compressed[2];
  CUevent t_compress_start[2];
  CUevent t_compress_end[2];
  CUevent t_aggregate_end[2];

  uint32_t* pool_epochs_scratch; // [K] scratch for kick-time mask scans

  struct level_flush_state levels[LOD_MAX_LEVELS];
};

struct d2h_deliver_stage
{
  CUevent t_d2h_start[2];
  CUevent
    offsets_ready[2]; // phase 1 (offset D2H) completion; drain syncs on this
  CUevent ready[2];   // phase 2 (bulk D2H) completion

  struct level_flush_state* levels; // borrowed
  int nlod;
  size_t shard_alignment;         // from sink; 0 = no alignment
  struct stream_metrics* metrics; // borrowed, for stall-time accumulation
};

// Lazy-delivery pipeline: each fc slot can independently hold a kicked-but-
// not-yet-delivered batch. Delivery of fc=A happens at the start of the
// drain_kick_and_swap round that is about to reuse fc=A (two rounds after
// that batch was kicked, in N=2 alternation). Both fcs may hold a pending
// batch simultaneously; drain order at final flush follows pending_seq.
struct flush_pipeline
{
  struct flush_slot_gpu slot[2];
  int current;                                 // fc currently being filled
  int pending[2];                              // per-fc: batch awaiting delivery
  uint64_t pending_seq[2];                     // monotonic kick seq per pending
  uint64_t next_seq;                           // next seq to assign on kick
  struct flush_handoff pending_handoff[2];     // per-fc
};

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// --- Engine / Context ---

// Per-array identity — lightweight, scales with number of arrays.
// Holds immutable configuration plus the append cursor.
// Mutable batch/flush/shard state lives in the engine's sub-structs and is
// swapped via bind/unbind when switching arrays (multiarray only).
struct stream_context
{
  struct tile_stream_configuration config;
  struct shard_sink* sink;
  struct tile_stream_layout layout;
  struct tile_stream_layout_gpu layout_gpu;
  struct level_geometry levels;
  struct dim_info dims;
  uint64_t cursor_elements;
  uint64_t total_element_limit; // configured stream length; 0 = unbounded
  size_t shard_alignment;       // from sink; 0 = no alignment
};

// Shared GPU resources — constant memory, allocated once.
// Contains all GPU allocations (pools, staging, codec, CUDA streams) and
// mutable pipeline state (batch accumulation, flush slots, shard tracking).
struct stream_engine
{
  int sync_flush; // 1 = synchronous batch flush (multiarray); 0 = pipelined
  struct gpu_streams streams;
  struct pool_state pools;
  size_t pool_bytes;
  struct staging_state stage;
  struct batch_state batch;
  struct flush_pipeline flush;
  struct compress_agg_stage compress_agg;
  struct d2h_deliver_stage d2h_deliver;
  struct lod_shared_state lod_shared; // engine-owned shared LOD resources
  struct lod_state lod;               // per-array; overwritten on array switch
  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};

// --- Engine operations ---

// Pointer to the given epoch's chunk region in the current pool.
void*
stream_engine_pool_epoch(struct stream_engine* e,
                         struct stream_context* ctx,
                         uint32_t epoch_in_batch);

// Build the initial set of metric labels.  Shared by single-array and
// multiarray GPU constructors; enable_multiscale only affects the label of
// the scatter/copy stage ("Copy" when multiscale, "Scatter" otherwise).
struct stream_metrics
stream_engine_init_metrics(int enable_multiscale);

// Append data to the stream. Handles staging, dispatch, epoch boundaries,
// batch flush, and backpressure. Used by both single-array and multiarray.
struct writer_result
stream_append_body(struct stream_engine* e,
                   struct stream_context* ctx,
                   struct slice input);

// Flush the stream: partial epoch, accumulated batch, partial append
// accumulators, finalize shards, update metadata.
struct writer_result
stream_flush_body(struct stream_engine* e, struct stream_context* ctx);
