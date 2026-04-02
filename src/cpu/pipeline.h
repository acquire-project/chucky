#pragma once

#include "stream/layouts.h"
#include "types.codec.h"
#include "types.stream.h"
#include "zarr/shard_delivery.h"

// Aggregate output slot (one per level).
struct cpu_agg_slot
{
  void* data; // aggregated compressed chunks in shard order
  size_t data_capacity_bytes;
  size_t* offsets;     // [C_lv + 1] exclusive prefix sum
  size_t* chunk_sizes; // [C_lv] pre-padding sizes for shard index
};

// ---- flush_batch ----

struct flush_level_view
{
  struct aggregate_layout* agg_layout;
  uint32_t batch_active_count;
  uint64_t chunk_offset;
  uint32_t* batch_chunk_to_shard_map; // [K_l * M_lv] perm LUT (mutable)
  uint32_t* batch_gather;             // [K_l * M_lv] gather LUT (mutable)
  struct cpu_agg_slot* agg_slot;
  struct shard_state* shard;
  struct io_event*
    io_done; // tracks pending async IO for this level's agg buffer
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
  struct flush_level_view levels[LOD_MAX_LEVELS];
  size_t* shard_order_sizes_bytes;
  struct shard_sink* sink;
  size_t shard_alignment_bytes;
  int max_threads; // 0 = OpenMP default
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
  void* chunk_pool;
  void* linear;
  void* lod_values;
  uint32_t* scatter_lut;
  uint64_t* scatter_batch_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_batch_offsets[LOD_MAX_LEVELS];
  void* append_accum;
  uint32_t* append_counts;        // mutable
  int max_threads; // 0 = OpenMP default
  struct stream_metrics* metrics; // NULL to skip timing
};

int
cpu_pipeline_scatter_epoch(const struct scatter_epoch_params* p,
                           uint32_t epoch_in_batch,
                           uint32_t* out_mask);

// ---- LUT computation ----

struct lut_targets
{
  uint32_t* batch_gather[LOD_MAX_LEVELS];
  uint32_t* batch_chunk_to_shard_map[LOD_MAX_LEVELS];
  uint32_t* scatter_lut;
  uint64_t* scatter_batch_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_batch_offsets[LOD_MAX_LEVELS];
};

void
cpu_pipeline_compute_luts(
  const struct computed_stream_layouts* cl,
  const struct level_geometry* levels,
  const uint32_t batch_active_count[LOD_MAX_LEVELS],
  const struct aggregate_layout agg_layout[LOD_MAX_LEVELS],
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
  uint64_t* lod_batch_offsets[LOD_MAX_LEVELS];
  int max_threads; // 0 = OpenMP default
  struct stream_metrics* metrics; // NULL to skip timing
};

int
cpu_pipeline_append_drain(const struct append_drain_params* p,
                          uint32_t* out_drain_mask);
