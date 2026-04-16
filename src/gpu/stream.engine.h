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
struct flush_slot_gpu
{
  uint32_t active_levels_mask; // union of per-epoch active masks
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS]; // per-epoch active level masks
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

struct lod_state
{
  struct lod_plan plan;

  CUdeviceptr d_linear; // linear epoch buffer (device)
  CUdeviceptr d_morton; // morton-ordered LOD output (all levels packed)

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

  // Per-frame-counter timing events (double-buffered).
  struct lod_timing
  {
    CUevent t_start;
    CUevent t_scatter_end;
    CUevent t_reduce_end;
    CUevent t_append_end;
    CUevent t_end;
  } timing[2];

  // Append-dim LOD accumulation state.
  struct
  {
    CUdeviceptr d_accum;
    CUdeviceptr d_level_ids;
    CUdeviceptr d_counts;
    uint32_t counts[LOD_MAX_LEVELS];
    uint64_t total_elements;
    uint64_t morton_offset;
  } append_accum;
};

// CUDA stream handles (all immutable after create)
struct gpu_streams
{
  CUstream h2d, compute, compress, d2h;
};

// Batch accumulation: config + mutable counter + per-epoch events
struct batch_state
{
  uint32_t epochs_per_batch;             // K (immutable after create)
  uint32_t accumulated;                  // mutable: 0..K-1
  CUevent pool_events[MAX_BATCH_EPOCHS]; // per-epoch pool-ready signals
};

// --- Stage types ---

struct compress_agg_input
{
  int fc;
  uint32_t n_epochs;
  uint32_t active_levels_mask;
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS];
  CUdeviceptr pool_buf;
  CUevent epoch_events[MAX_BATCH_EPOCHS];
  CUevent lod_done;
  uint32_t epochs_per_batch;
};

struct compress_agg_stage
{
  struct codec codec;
  CUdeviceptr d_compressed[2];
  CUevent t_compress_start[2];
  CUevent t_compress_end[2];
  CUevent t_aggregate_end[2];

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
  CUstream d2h_stream; // set by kick, consumed by drain (always paired)
};

struct flush_pipeline
{
  struct flush_slot_gpu slot[2];
  int current;
  int pending;
  struct flush_handoff pending_handoff;
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
  uint64_t max_cursor_elements; // 0 = unbounded
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
  struct lod_state lod;
  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};

// --- Engine operations ---

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
