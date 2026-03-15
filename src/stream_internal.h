#pragma once

#include "shard_delivery.h"
#include "stream.h" // public types

#include "aggregate.h"
#include "lod_plan.h"
#include "platform.h"

#define MAX_BATCH_EPOCHS 128

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

// Per flush-slot: holds compressed output + pre-built pointer arrays.
// flush[0] is used for A-pool epochs, flush[1] for B-pool epochs.
struct flush_slot_gpu
{
  CUdeviceptr d_compressed; // device: K * total_tiles * max_output_size
  CUevent t_compress_end;   // signals compress finished
  CUevent t_compress_start;
  CUevent t_aggregate_end;
  CUevent t_d2h_start;
  CUevent ready;               // signals all D2H for this slot is done
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
    d_batch_gather; // [K_l * M_l] uint32: batch-tile -> compressed idx
  CUdeviceptr
    d_batch_perm; // [K_l * M_l] uint32: batch-tile -> shard-ordered pos
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

  // Per-level tile layouts [1..nlod-1], index 0 unused
  struct stream_layout layouts[LOD_MAX_LEVELS];

  // Morton-to-tile scatter LUTs (precomputed)
  CUdeviceptr d_morton_tile_lut[LOD_MAX_LEVELS]; // u32, lod_counts[lv]
  CUdeviceptr d_morton_batch_tile_offsets[LOD_MAX_LEVELS]; // u32, batch_count

  CUevent t_start;
  CUevent t_scatter_end;
  CUevent t_reduce_end;
  CUevent t_dim0_end;
  CUevent t_end;

  // Dim0 (temporal) LOD accumulation state.
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

// Per-level tile geometry (all immutable after create)
struct level_geometry
{
  int nlod;                             // number of LOD levels
  int enable_multiscale;                // has spatial LOD
  int dim0_downsample;                  // has temporal LOD
  uint64_t total_tiles;                 // sum across all levels per epoch
  uint64_t tile_offset[LOD_MAX_LEVELS]; // first tile index per level
  uint64_t tile_count[LOD_MAX_LEVELS];  // tiles_per_epoch per level
};

// Batch accumulation: config + mutable counter + per-epoch events
struct batch_state
{
  uint32_t epochs_per_batch;             // K (immutable after create)
  uint32_t accumulated;                  // mutable: 0..K-1
  CUevent pool_events[MAX_BATCH_EPOCHS]; // per-epoch pool-ready signals
};

// Flush pipeline: double-buffered compress->D2H->deliver
struct flush_pipeline
{
  struct flush_slot_gpu slot[2];                   // [0]=A pool, [1]=B pool
  struct level_flush_state levels[LOD_MAX_LEVELS]; // per-level agg+shard
  int current; // mutable: which slot is active
  int pending; // mutable: has unkicked work
};

struct tile_stream_gpu
{
  struct writer writer;
  struct tile_stream_configuration config;
  struct stream_layout layout;  // L0 tile layout
  struct level_geometry levels; // per-level accounting
  struct gpu_streams streams;   // CUDA stream handles
  struct batch_state batch;     // epoch accumulation
  struct flush_pipeline flush;  // compress->deliver pipeline
  struct codec codec;           // compression state
  struct lod_state lod;         // LOD buffers + plan

  struct pool_state pools;       // tile pools
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
  uint64_t tiles_per_shard_0;
  uint64_t tiles_per_shard_inner;
  uint64_t tiles_per_shard_total;
  uint64_t shard_inner_count;
};

// All pre-computed layout data from CPU-only math.
// Produced by compute_stream_layouts, consumed by the create path
// and the memory estimate path.
struct computed_stream_layouts
{
  struct stream_layout l0; // host fields; d_* = NULL
  struct lod_plan plan;    // owned if enable_multiscale
  struct stream_layout
    lod_layouts[LOD_MAX_LEVELS]; // host; d_* = NULL; [0] unused
  struct level_geometry levels;
  uint32_t epochs_per_batch;
  size_t max_output_size; // codec-derived compressed tile bound
  struct level_layout_info per_level[LOD_MAX_LEVELS];
};

// Validate config and compute all CPU-only layout math. Returns 0 on success.
int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       struct computed_stream_layouts* out);

// Free resources owned by computed_stream_layouts.
void
computed_stream_layouts_free(struct computed_stream_layouts* cl);

#include "stream_flush.h"
#include "stream_ingest.h"
#include "stream_lod.h"
