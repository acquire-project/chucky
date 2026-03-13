#pragma once

#include "aggregate.h"
#include "compress.h"
#include "lod.h"
#include "lod_plan.h"
#include "metric.h"
#include "platform.h"
#include "shard_delivery.h"
#include "transpose.h"
#include "writer.h"
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_BATCH_EPOCHS 256

struct stream_metrics
{
  struct stream_metric memcpy;
  struct stream_metric h2d;
  struct stream_metric lod_gather;
  struct stream_metric lod_reduce;
  struct stream_metric lod_dim0_fold;
  struct stream_metric lod_morton_tile;
  struct stream_metric scatter;
  struct stream_metric compress;
  struct stream_metric aggregate;
  struct stream_metric d2h;
  struct stream_metric sink;
};

enum domain
{
  host,
  device
};

struct buffer
{
  void* data;
  CUevent ready;
  enum domain domain;
};

struct double_buffer
{
  struct buffer buf[2];
  int current; // 0 or 1
};

struct dimension
{
  uint64_t size; // 0 means unbounded (dim 0 only: stream indefinitely)
  uint64_t tile_size;
  uint64_t tiles_per_shard; // 0 means all tiles along this dimension
                            // (must be > 0 when size == 0)
  const char* name;         // optional label (e.g. "x"), may be NULL
  int downsample;           // include in LOD pyramid
  uint8_t storage_position; // position in storage layout (0=outermost).
                            // dims[0].storage_position must be 0.
                            // Must be a valid permutation of 0..rank-1.
};

struct tile_stream_configuration
{
  // FIXME: which buffer is this refering to?
  size_t buffer_capacity_bytes;
  size_t bytes_per_element;
  uint8_t rank;
  const struct dimension* dimensions;
  struct shard_sink* shard_sink; // downstream shard writer factory, not owned
  enum compression_codec codec;  // compression codec for tiles
  enum lod_reduce_method reduce_method;      // epoch LOD reduction method
  enum lod_reduce_method dim0_reduce_method; // dim0 LOD reduction
  // FIXME: this doesn't need 4-bytes - could be a u8
  uint32_t epochs_per_batch; // K: 0 = auto (target_min_tiles), must be pow2
  // FIXME: rename target_min_tiles to make it clear it's per batch
  uint32_t target_min_tiles; // minimum tiles per compress batch (default 1024)
  float metadata_update_interval_s; // seconds between metadata updates
  size_t
    shard_alignment; // 0 = no padding; platform_page_size() for unbuffered IO
};

struct staging_slot
{
  struct buffer h_in;      // pinned host WC, size = buffer_capacity_bytes
  struct buffer d_in;      // device, size = buffer_capacity_bytes
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

struct stream_layout
{
  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];

  uint64_t* d_lifted_shape;  // device copy (allocated once)
  int64_t* d_lifted_strides; // device copy (allocated once)

  uint64_t tile_elements;   // elements per tile
  uint64_t tile_stride;     // elements between tile starts (>= tile_elements)
  uint64_t tiles_per_epoch; // M = prod of tile_count[i] for i > 0
  uint64_t epoch_elements;  // elements per epoch = M * tile_elements
  size_t tile_pool_bytes;   // tiles_per_epoch * tile_stride * bpe
};

// Per flush-slot: holds compressed output + pre-built pointer arrays.
// flush[0] is used for A-pool epochs, flush[1] for B-pool epochs.
struct flush_slot_gpu
{
  struct buffer d_compressed; // device: K * total_tiles * max_output_size
  CUevent t_compress_start;
  CUevent t_aggregate_end;
  CUevent t_d2h_start;
  CUevent ready;               // signals all D2H for this slot is done
  uint32_t active_levels_mask; // union of per-epoch active masks
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS]; // per-epoch active level masks
  int batch_epoch_count; // number of epochs accumulated in this batch
};

struct lod_level_state
{
  struct aggregate_layout agg_layout;
  struct aggregate_slot agg[2]; // double-buffered, indexed by flush_current
  struct shard_state shard;
  CUdeviceptr d_batch_gather; // [K_l * M_l] uint32: batch-tile → compressed idx
  CUdeviceptr
    d_batch_perm; // [K_l * M_l] uint32: batch-tile → shard-ordered pos
  uint32_t batch_active_count; // K_l = K / 2^l for this level
};

// Dim0 (temporal) LOD accumulation state.
// Single buffer covering all LOD levels 1+, same packed layout as d_morton.
struct dim0_state
{
  struct buffer d_accum;           // GPU: all levels 1+ packed, accum_bpe
  CUdeviceptr d_level_ids;         // GPU: u8 per element, maps to level
  CUdeviceptr d_counts;            // GPU: nlod uint32_t, per-level count
  uint32_t counts[LOD_MAX_LEVELS]; // CPU mirror of d_counts
  uint64_t total_elements;         // sum(batch_count * lod_counts[k]) k=1..
  uint64_t morton_offset; // levels.ends[0] (start of level 1 in d_morton)
};

struct lod_state
{
  struct lod_plan plan;

  struct buffer d_linear; // linear epoch buffer (device)
  struct buffer d_morton; // morton-ordered LOD output (all levels packed)

  CUdeviceptr d_full_shape; // device copy of shapes[0]
  CUdeviceptr d_lod_shape;  // device copy of lod_shapes[0]
  CUdeviceptr d_ends;       // device copy of ends

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
};

struct tile_stream_gpu;

struct tile_stream_memory_info
{
  size_t device_bytes;      // total GPU memory
  size_t host_pinned_bytes; // total pinned host memory

  // Breakdown (device)
  size_t staging_bytes;         // 2 x buffer_capacity_bytes
  size_t tile_pool_bytes;       // 2 x total_tiles x tile_stride x bpe
  size_t compressed_pool_bytes; // 2 x total_tiles x max_output_size
  size_t aggregate_bytes;       // sum over levels: device aggregate buffers
  size_t lod_bytes;             // d_linear + d_morton + shape arrays
  size_t codec_bytes;           // nvcomp workspace + pointer arrays

  // Key parameters used in the estimate
  uint64_t tiles_per_epoch;  // L0
  uint64_t total_tiles;      // sum across all LOD levels
  size_t max_output_size;    // compressed tile bound
  int nlod;                  // number of LOD levels
  uint32_t epochs_per_batch; // K
};

// Estimate GPU memory requirements without allocating.
// Returns 0 on success, non-zero on invalid config.
int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_memory_info* info);

// Dispatch function: H2D + scatter (+ optional d_linear copy).
typedef int (*dispatch_scatter_fn)(struct tile_stream_gpu*);

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
  struct flush_slot_gpu slot[2]; // [0]=A pool, [1]=B pool
  int current;                   // mutable: which slot is active
  int pending;                   // mutable: has unkicked work
};

struct tile_stream_gpu
{
  struct writer writer;
  struct tile_stream_configuration config;
  dispatch_scatter_fn dispatch;
  struct stream_layout layout;  // L0 tile layout
  struct level_geometry levels; // per-level accounting
  struct gpu_streams streams;   // CUDA stream handles
  struct batch_state batch;     // epoch accumulation
  struct flush_pipeline flush;  // compress->deliver pipeline
  struct codec codec;           // compression state
  struct lod_state lod;         // LOD buffers + plan
  struct lod_level_state lod_levels[LOD_MAX_LEVELS]; // per-level agg+shard

  struct double_buffer pools;    // tile pools
  struct staging_state stage;    // H2D staging buffers
  struct dim0_state dim0;        // temporal accumulation
  uint64_t cursor;               // current element position
  struct stream_metrics metrics; // telemetry
  struct platform_clock metadata_update_clock;
};

// Initialize a tile_stream_gpu. Returns 0 on success, non-zero on error.
// On failure, *out is zeroed and safe to pass to tile_stream_gpu_destroy.
int
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct tile_stream_gpu* out);

void
tile_stream_gpu_destroy(struct tile_stream_gpu* stream);

// Return accumulated timing metrics.
struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s);
