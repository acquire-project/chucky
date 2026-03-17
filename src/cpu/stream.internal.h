#pragma once

#include "platform.h"
#include "shard_delivery.h"
#include "stream.h"
#include "types.aggregate.h"

struct tile_stream_cpu
{
  struct writer writer;
  struct tile_stream_configuration config;
  struct computed_stream_layouts cl;

  // L0 layout (also in cl.l0, aliased here for convenience)
  struct tile_stream_layout layout;
  struct level_geometry levels;

  // Single-buffered chunk pool: total_chunks * chunk_stride * bpe bytes.
  void* chunk_pool;

  // Compressed output: total_chunks * max_output_size bytes.
  void* compressed;
  size_t* comp_sizes; // [total_chunks]

  // Per-level shard state
  struct shard_state shard[LOD_MAX_LEVELS];
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];

  // LOD (multiscale only)
  void* linear;     // linear epoch buffer (input accumulated here before scatter)
  void* lod_values; // morton-ordered LOD buffer (all levels packed)

  // Dim0 downsample accumulation state
  void* dim0_accum;                    // accumulator for levels 1+
  uint32_t dim0_counts[LOD_MAX_LEVELS]; // per-level fold count

  uint64_t cursor;
  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};
