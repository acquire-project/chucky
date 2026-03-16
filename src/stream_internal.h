#pragma once

#include "aggregate.h"
#include "compress.h"
#include "flush_handoff.h"
#include "lod_plan.h"
#include "platform.h"
#include "shard_delivery.h"
#include "stream.h" // public types
#include <stddef.h>

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
// Compress/D2H events and d_compressed now live in the stage structs.
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

  CUdeviceptr d_full_shape;    // device copy of shapes[0]
  CUdeviceptr d_lod_shape;     // device copy of lod_shapes[0]
  CUdeviceptr d_gather_lut;    // u32, lod_counts[0] entries
  CUdeviceptr d_batch_offsets; // u32, batch_count entries

  CUdeviceptr d_child_shapes[LOD_MAX_LEVELS];
  CUdeviceptr d_parent_shapes[LOD_MAX_LEVELS];
  CUdeviceptr d_level_ends[LOD_MAX_LEVELS];

  // Per-level chunk layouts [1..nlod-1], index 0 unused
  struct tile_stream_layout layouts[LOD_MAX_LEVELS];
  struct tile_stream_layout_gpu layout_gpu[LOD_MAX_LEVELS];

  // Morton-to-chunk scatter LUTs (precomputed)
  CUdeviceptr d_morton_chunk_lut[LOD_MAX_LEVELS]; // u32, lod_counts[lv]
  CUdeviceptr d_morton_batch_chunk_offsets[LOD_MAX_LEVELS]; // u32, batch_count

  CUevent t_start;
  CUevent t_scatter_end;
  CUevent t_reduce_end;
  CUevent t_dim0_end;
  CUevent t_end;

  // Dim0 LOD accumulation state.
  // Single buffer covering all LOD levels 1+, same packed layout as d_morton.
  struct
  {
    CUdeviceptr d_accum;             // GPU: all levels 1+ packed, accum_bpe
    CUdeviceptr d_level_ids;         // GPU: u8 per element, maps to level
    CUdeviceptr d_counts;            // GPU: nlod uint32_t, per-level count
    uint32_t counts[LOD_MAX_LEVELS]; // CPU mirror of d_counts
    uint64_t total_elements;         // sum(batch_count * lod_counts[k]) k=1..
    uint64_t morton_offset; // levels.ends[0] (start of level 1 in d_morton)
  } dim0;
};

// CUDA stream handles (all immutable after create)
struct gpu_streams
{
  CUstream h2d, compute, compress, d2h;
};

// Per-level chunk geometry (all immutable after create)
struct level_geometry
{
  int nlod;                              // number of LOD levels
  int enable_multiscale;                 // has inner LOD
  int dim0_downsample;                   // has dim0 LOD
  uint64_t total_chunks;                 // sum across all levels per epoch
  uint64_t chunk_offset[LOD_MAX_LEVELS]; // first chunk index per level
  uint64_t chunk_count[LOD_MAX_LEVELS];  // chunks_per_epoch per level
};

// Batch accumulation: config + mutable counter + per-epoch events
struct batch_state
{
  uint32_t epochs_per_batch;             // K (immutable after create)
  uint32_t accumulated;                  // mutable: 0..K-1
  CUevent pool_events[MAX_BATCH_EPOCHS]; // per-epoch pool-ready signals
};

// --- Stage types (depend on types above, needed by tile_stream_gpu) ---

// Input to the compress+aggregate stage.
struct compress_agg_input
{
  int fc;                                        // flush slot index (0 or 1)
  uint32_t n_epochs;                             // epochs in this batch
  uint32_t active_levels_mask;                   // union of per-epoch masks
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS]; // per-epoch active level masks
  CUdeviceptr pool_buf;                          // chunk pool for this slot
  CUevent epoch_events[MAX_BATCH_EPOCHS];        // per-epoch pool-ready signals
  CUevent lod_done;                              // NULL if no multiscale
  uint32_t epochs_per_batch;                     // K, for LUT path decisions
};

// Compress+aggregate stage. Owns codec, d_compressed buffers, compress/agg
// events, per-level aggregate_layout + batch LUTs.
struct compress_agg_stage
{
  struct codec codec;
  CUdeviceptr d_compressed[2]; // per flush slot
  CUevent t_compress_start[2];
  CUevent t_compress_end[2];
  CUevent t_aggregate_end[2];

  struct level_flush_state levels[LOD_MAX_LEVELS]; // per-level agg+shard+LUTs
};

// D2H + deliver stage. Owns D2H events and references to shard states
// (which live in compress_agg_stage.levels[]).
struct d2h_deliver_stage
{
  CUevent t_d2h_start[2]; // per flush slot
  CUevent ready[2];       // per flush slot: D2H completion

  // Borrowed references (not owned)
  struct level_flush_state* levels; // points to compress_agg_stage.levels
  int nlod;
};

// Flush pipeline: double-buffered compress->D2H->deliver
struct flush_pipeline
{
  struct flush_slot_gpu slot[2];        // [0]=A pool, [1]=B pool
  int current;                          // mutable: which slot is active
  int pending;                          // mutable: has un-drained work
  struct flush_handoff pending_handoff; // saved handoff for drain
};

struct tile_stream_gpu
{
  struct writer writer;
  struct tile_stream_configuration config;
  struct tile_stream_layout layout;      // L0 chunk layout
  struct tile_stream_layout_gpu layout_gpu; // L0 device arrays
  struct level_geometry levels; // per-level accounting
  struct gpu_streams streams;   // CUDA stream handles
  struct batch_state batch;     // epoch accumulation
  struct flush_pipeline flush;  // orchestration state (slots + pending)
  struct compress_agg_stage compress_agg; // compress+aggregate stage
  struct d2h_deliver_stage d2h_deliver;   // D2H+deliver stage
  struct lod_state lod;                   // LOD buffers + plan

  struct pool_state pools;       // chunk pools
  struct staging_state stage;    // H2D staging buffers
  uint64_t cursor;               // current element position
  struct stream_metrics metrics; // telemetry
  struct platform_clock metadata_update_clock;
};

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// Set writer vtable (append/flush).
// Called from tile_stream_gpu_create after zeroing *out.
void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s);

// Per-level pre-computed layout information (CPU only, no GPU pointers).
struct level_layout_info
{
  struct aggregate_layout agg_layout; // host fields only, d_* = NULL
  uint32_t batch_active_count;
  uint64_t chunks_per_shard_0;
  uint64_t chunks_per_shard_inner;
  uint64_t chunks_per_shard_total;
  uint64_t shard_inner_count;
};

// All pre-computed layout data from CPU-only math.
// Produced by compute_stream_layouts, consumed by the create path
// and the memory estimate path.
struct computed_stream_layouts
{
  struct tile_stream_layout l0; // host fields only
  struct lod_plan plan;         // owned if enable_multiscale
  struct tile_stream_layout
    lod_layouts[LOD_MAX_LEVELS]; // host only; [0] unused
  struct level_geometry levels;
  uint32_t epochs_per_batch;
  size_t max_output_size; // codec-derived compressed chunk bound
  struct level_layout_info per_level[LOD_MAX_LEVELS];
};

// Validate config and compute all CPU-only layout math. Returns 0 on success.
int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       struct computed_stream_layouts* out);

// Free resources owned by computed_stream_layouts.
void
computed_stream_layouts_free(struct computed_stream_layouts* cl);

// Context for flush pipeline operations.
// Stack-allocated by the orchestrator; contains only pointers/copies.
struct flush_context
{
  struct flush_pipeline* flush;
  struct compress_agg_stage* compress_agg;
  struct d2h_deliver_stage* d2h_deliver;
  const struct level_geometry* levels;
  struct batch_state* batch;
  struct pool_state* pools;
  struct lod_state* lod;
  struct stream_metrics* metrics;
  const struct tile_stream_configuration* config;
  const struct tile_stream_layout* layout;
  struct gpu_streams streams;
  struct platform_clock* metadata_update_clock;
};
