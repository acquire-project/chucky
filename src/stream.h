#pragma once

#include "aggregate.h"
#include "compress.h"
#include "lod.h"
#include "lod_plan.h"
#include "metric.h"
#include "platform.h"
#include "transpose.h"
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_BATCH_EPOCHS 256

struct slice
{
  const void* beg;
  const void* end;
};

enum writer_error_code
{
  writer_error_ok = 0,
  writer_error_fail = 1,
  writer_error_finished = 2, // bounded dim0 capacity reached, data flushed
};

struct writer_result
{
  int error;         // writer_error_code; 0 = ok, 1 = fail, 2 = finished
  struct slice rest; // unconsumed input (empty on success for append)
};

struct writer
{
  struct writer_result (*append)(struct writer* self, struct slice data);
  struct writer_result (*flush)(struct writer* self);
};

struct shard_writer
{
  int (*write)(struct shard_writer* self,
               uint64_t offset, // byte offset within the shard
               const void* beg,
               const void* end);
  // Zero-copy write: caller guarantees buffer lifetime until io_event.
  // NULL = fall back to write (copy-based).
  int (*write_direct)(struct shard_writer* self,
                      uint64_t offset,
                      const void* beg,
                      const void* end);
  int (*finalize)(struct shard_writer* self); // shard complete, close/flush
};

struct shard_sink
{
  // Open/get a writer for the given flat shard index.
  struct shard_writer* (*open)(struct shard_sink* self,
                               uint8_t level,
                               uint64_t shard_index);

  // Optional: update dim0 extent in metadata (e.g. zarr.json shape[0]).
  // Called periodically during streaming and at final flush.
  // NULL means no-op (non-zarr sinks can ignore).
  void (*update_dim0)(struct shard_sink* self,
                      uint8_t level,
                      uint64_t dim0_size);

  // IO fence for backpressure. NULL = no async IO.
  struct io_event (*record_fence)(struct shard_sink* self, uint8_t level);
  void (*wait_fence)(struct shard_sink* self,
                     uint8_t level,
                     struct io_event ev);
};

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
};

struct tile_stream_configuration
{
  size_t buffer_capacity_bytes;
  size_t bytes_per_element;
  uint8_t rank;
  const struct dimension* dimensions;
  struct shard_sink* shard_sink; // downstream shard writer factory, not owned
  enum compression_codec codec;  // compression codec for tiles
  enum lod_reduce_method reduce_method;      // spatial LOD reduction method
  enum lod_reduce_method dim0_reduce_method; // dim0 (temporal) LOD reduction
  uint32_t epochs_per_batch; // K: 0 = auto (target_min_tiles), must be pow2
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

struct active_shard
{
  size_t data_cursor;
  uint64_t* index;             // 2 * tiles_per_shard_total entries
  struct shard_writer* writer; // from sink->open, NULL until first use
};

struct shard_state
{
  uint64_t epoch_in_shard;        // 0..tiles_per_shard[0]-1
  uint64_t shard_epoch;           // s_0 coordinate (0, 1, 2, ...)
  uint64_t shard_inner_count;     // S_inner = prod(shard_count[d] for d>0)
  uint64_t tiles_per_shard_inner; // prod(tps[d] for d>0)
  uint64_t tiles_per_shard_total; // prod(tps[d] for all d)
  uint64_t tiles_per_shard_0;     // tps[0]
  struct active_shard* shards;    // array[shard_inner_count]
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

struct tile_stream_gpu
{
  struct writer writer;
  dispatch_scatter_fn dispatch;
  CUstream h2d, compute, compress, d2h;
  struct staging_state stage;
  struct stream_layout layout; // L0 layout
  struct tile_stream_configuration config;
  struct stream_metrics metrics;
  uint64_t cursor;

  // Tile pools — unified across all levels
  struct double_buffer pools; // total_tiles * tile_stride * bpe each

  // Flush pipeline
  struct flush_slot_gpu flush[2]; // [0]=A epochs, [1]=B epochs
  int flush_current;              // 0 or 1
  int flush_pending;

  // Batch accumulation
  uint32_t epochs_per_batch;   // number of epochs per compress batch
  uint32_t epochs_accumulated; // counter of epochs in current pool
  CUevent batch_pool_events[MAX_BATCH_EPOCHS]; // per-epoch pool-ready signals

  // Unified compression state (sized for total_tiles)
  struct codec codec;
  uint64_t total_tiles;                       // sum of all level tile counts
  uint64_t level_tile_offset[LOD_MAX_LEVELS]; // first tile index per level
  uint64_t level_tile_count[LOD_MAX_LEVELS];  // tiles_per_epoch per level

  int nlod;              // 1 when multiscale off, lod.nlod when on
  int enable_multiscale; // computed from dimensions[].downsample
  uint32_t lod_mask;     // computed from dimensions[].downsample

  // LOD (multiscale) state
  struct lod_state lod;
  // Per-level aggregate + shard delivery
  // lod_levels[0] = L0 state (agg_layout, agg[2], shard)
  // lod_levels[1..nlod-1] = LOD levels
  struct lod_level_state lod_levels[LOD_MAX_LEVELS];

  // Dim0 (temporal) LOD accumulation
  int dim0_downsample;                       // 1 if dim 0 is downsampled
  enum lod_reduce_method dim0_reduce_method; // temporal reduction method
  struct dim0_state dim0;                    // single buffer for all levels 1+

  // Metadata update timer
  struct platform_clock metadata_update_clock;
};

// Initialize a tile_stream_gpu. Returns 0 on success, non-zero on error.
// On failure, *out is zeroed and safe to pass to tile_stream_gpu_destroy.
int
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct tile_stream_gpu* out);

void
tile_stream_gpu_destroy(struct tile_stream_gpu* stream);

// Dispatch to the writer's append method.
struct writer_result
writer_append(struct writer* w, struct slice data);

// Dispatch to the writer's flush method.
struct writer_result
writer_flush(struct writer* w);

// Append data to a writer, retrying with exponential back-off on stall.
struct writer_result
writer_append_wait(struct writer* w, struct slice data);

// Return accumulated timing metrics.
struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s);
