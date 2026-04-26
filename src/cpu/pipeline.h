#pragma once

#include "lod/reduce_csr.h"
#include "stream/layouts.h"
#include "types.codec.h"
#include "types.stream.h"
#include "zarr/shard_delivery.h"

struct threadpool;

// Per-slot aggregate workspace shared across all LODs in a batch. The data
// buffer is page-aligned so deliver-time write_direct guards still hold
// against unbuffered (O_DIRECT) shard sinks. Scratch arrays are sized for
// the worst-case unified batch (every LOD active across every epoch).
struct cpu_agg_slot
{
  void* data; // aggregated compressed chunks in shard order
  size_t data_capacity_bytes;

  // Unified scratch sized to max(total_batch_chunks, 1).
  uint32_t* perm;
  uint32_t* gather;

  // Unified scratch sized to max(total_batch_covering, 1).
  size_t* permuted_sizes;
  size_t* offsets;     // [max_total_covering + 1] exclusive prefix sum
  size_t* chunk_sizes; // [max_total_covering] pre-padding sizes
};

// ---- flush_batch ----

struct flush_level_view
{
  uint32_t batch_active_count;
  uint64_t chunk_offset;
  struct shard_state* shard;
};

struct flush_batch_params
{
  struct codec_config codec;
  size_t bytes_per_element;
  const void* chunk_pool;
  size_t chunk_stride_bytes;
  size_t chunk_bytes;
  void* compressed;
  size_t max_output_size_bytes;
  size_t* comp_sizes;
  uint64_t total_chunks;
  int nlod;
  const struct computed_stream_layouts* cl;
  const struct level_geometry* levels_geo;
  // Per-LOD aggregate_layouts for the unified batch LUT builder. Points
  // at a contiguous LOD_MAX_LEVELS-element array (typically owned by the
  // stream).
  const struct aggregate_layout* per_lod_agg_layouts;
  struct flush_level_view levels[LOD_MAX_LEVELS];
  struct shard_sink* sink;
  size_t shard_alignment_bytes;
  struct threadpool* pool;        // owned by stream
  uint32_t* pool_epochs_scratch;  // [K] scratch for LUT recompute
  struct cpu_agg_slot* agg_slots; // [2] shared per-batch workspace
  struct io_event* io_done;       // [2] per-slot pending-IO fence
  uint8_t* agg_current;           // single alternator byte (0 or 1)
  struct stream_metrics* metrics; // NULL to skip timing
};

int
cpu_pipeline_flush_batch(const struct flush_batch_params* p,
                         uint32_t n_epochs,
                         const uint32_t* active_masks);

// ---- scatter_epoch ----

struct scatter_epoch_params
{
  enum dtype dtype;
  enum lod_reduce_method reduce_method;
  enum lod_reduce_method append_reduce_method;
  const struct computed_stream_layouts* cl;
  const struct reduce_csr* csrs; // [nlod-1] CSR reduce LUTs
  void* chunk_pool;
  void* linear;
  void* lod_values;
  uint32_t* scatter_lut;
  uint64_t* scatter_fixed_dims_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_fixed_dims_offsets[LOD_MAX_LEVELS];
  void* append_accum;
  uint32_t* append_counts; // mutable
  struct threadpool* pool;
  struct stream_metrics* metrics; // NULL to skip timing
};

int
cpu_pipeline_scatter_epoch(const struct scatter_epoch_params* p,
                           uint32_t epoch_in_batch,
                           uint32_t* out_mask);

// ---- LUT computation ----

struct lut_targets
{
  uint32_t* scatter_lut;
  uint64_t* scatter_fixed_dims_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_fixed_dims_offsets[LOD_MAX_LEVELS];
};

void
cpu_pipeline_compute_luts(const struct computed_stream_layouts* cl,
                          const struct level_geometry* levels,
                          struct threadpool* pool,
                          struct lut_targets* out);

// ---- append drain ----

struct append_drain_params
{
  const struct computed_stream_layouts* cl;
  enum dtype dtype;
  enum lod_reduce_method append_reduce_method;
  void* lod_values;
  void* append_accum;
  uint32_t* append_counts;
  void* chunk_pool;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_fixed_dims_offsets[LOD_MAX_LEVELS];
  struct threadpool* pool;
  struct stream_metrics* metrics; // NULL to skip timing
};

int
cpu_pipeline_append_drain(const struct append_drain_params* p,
                          uint32_t* out_drain_mask);
