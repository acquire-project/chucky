#pragma once

#include "lod/lod_plan.h"
#include "stream/dim_info.h"
#include "stream/types.aggregate.h"

#include <stddef.h>
#include <stdint.h>

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
  uint64_t total_chunks;
  uint64_t chunk_offset[LOD_MAX_LEVELS];
  uint64_t chunk_count[LOD_MAX_LEVELS];
};

// Per-level pre-computed layout information (CPU only, no GPU pointers).
struct level_layout_info
{
  struct aggregate_layout agg_layout;
  uint32_t batch_active_count;
  uint64_t chunks_per_shard_append;
  uint64_t chunks_per_shard_inner;
  uint64_t chunks_per_shard_total;
  uint64_t shard_inner_count;
};

// All pre-computed layout data from CPU-only math.
// Produced by compute_stream_layouts, consumed by the create path
// and the memory estimate path.
struct computed_stream_layouts
{
  struct dim_info dims; // resolved append/inner partition
  struct lod_plan plan; // owned if enable_multiscale
  struct tile_stream_layout layouts[LOD_MAX_LEVELS]; // [0] = L0
  struct level_geometry levels;
  uint32_t epochs_per_batch;
  size_t max_output_size;
  struct level_layout_info per_level[LOD_MAX_LEVELS];
};
