#pragma once

#include "defs.limits.h"
#include "dimension.h"
#include "lod_plan.h"
#include "metric.h"
#include "types.aggregate.h"
#include "types.codec.h"
#include "types.lod.h"
#include "writer.h"

#include <stddef.h>
#include <stdint.h>

struct stream_metrics
{
  struct stream_metric memcpy;
  struct stream_metric h2d;
  struct stream_metric lod_gather;
  struct stream_metric lod_reduce;
  struct stream_metric lod_dim0_fold;
  struct stream_metric lod_morton_chunk;
  struct stream_metric scatter;
  struct stream_metric compress;
  struct stream_metric aggregate;
  struct stream_metric d2h;
  struct stream_metric sink;
};

struct tile_stream_configuration
{
  size_t buffer_capacity_bytes;
  enum lod_dtype dtype;
  uint8_t rank;
  struct dimension* dimensions;
  struct shard_sink* shard_sink; // downstream shard writer factory, not owned
  enum compression_codec codec;
  enum lod_reduce_method reduce_method;
  enum lod_reduce_method dim0_reduce_method;
  uint8_t epochs_per_batch; // K: 0 = auto (target_batch_chunks), must be pow2
  uint32_t target_batch_chunks; // minimum chunks per compress batch (default 1024)
  float metadata_update_interval_s;
  size_t shard_alignment; // 0 = no padding; platform_page_size() for unbuffered IO
};

struct tile_stream_layout
{
  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];

  uint64_t chunk_elements;
  uint64_t chunk_stride;
  uint64_t chunks_per_epoch;
  uint64_t epoch_elements;
  size_t chunk_pool_bytes;
};

// Per-level chunk geometry (immutable after create)
struct level_geometry
{
  int nlod;
  int enable_multiscale;
  int dim0_downsample;
  uint64_t total_chunks;
  uint64_t chunk_offset[LOD_MAX_LEVELS];
  uint64_t chunk_count[LOD_MAX_LEVELS];
};

struct tile_stream_status
{
  int nlod;
  int dim0_downsample;
  uint32_t epochs_per_batch;
  size_t max_compressed_size;
  enum lod_dtype dtype;
  enum compression_codec codec;
  size_t codec_batch_size;
  uint32_t batch_accumulated;
  int pool_current;
  int flush_pending;
};

// Per-level pre-computed layout information (CPU only, no GPU pointers).
struct level_layout_info
{
  struct aggregate_layout agg_layout;
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
  struct tile_stream_layout l0;
  struct lod_plan plan; // owned if enable_multiscale
  struct tile_stream_layout lod_layouts[LOD_MAX_LEVELS]; // [0] unused
  struct level_geometry levels;
  uint32_t epochs_per_batch;
  size_t max_output_size;
  struct level_layout_info per_level[LOD_MAX_LEVELS];
};
