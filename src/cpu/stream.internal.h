#pragma once

#include "cpu/pipeline.h"
#include "platform/platform.h"
#include "stream.cpu.h"
#include "stream/layouts.h"
#include "zarr/shard_delivery.h"

struct tile_stream_cpu
{
  struct writer writer;
  struct tile_stream_configuration config;
  struct shard_sink* shard_sink;
  struct computed_stream_layouts cl;

  // L0 layout (also in cl.layouts[0], aliased here for convenience)
  struct tile_stream_layout layout;
  struct level_geometry levels;

  // Chunk pool: total_chunks * chunk_stride * bpe bytes.
  void* chunk_pool;

  // Compressed output: total_chunks * max_output_size bytes.
  void* compressed;
  size_t* comp_sizes; // [total_chunks]

  // Per-level shard state
  struct shard_state shard[LOD_MAX_LEVELS];
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];

  // Per-level aggregate output (batch-scaled when K > 1).
  uint32_t* chunk_to_shard_map[LOD_MAX_LEVELS];  // [M] chunk → shard position
  struct cpu_agg_slot agg_slots[LOD_MAX_LEVELS]; // [level] sized for batch
  size_t* shard_order_sizes;                     // [max_batch_C] shared scratch

  // Batch aggregate LUTs (K > 1 only, per level).
  uint32_t* batch_gather[LOD_MAX_LEVELS]; // [K_l * M_l]
  uint32_t*
    batch_chunk_to_shard_map[LOD_MAX_LEVELS];  // [K_l * M_l] interleaved map
  uint32_t batch_active_count[LOD_MAX_LEVELS]; // K_l per level

  // LOD (multiscale only)
  void* linear; // linear epoch buffer (input accumulated here before scatter)
  void* lod_values; // morton-ordered LOD buffer (all levels packed)

  // Precomputed LOD LUTs (multiscale only)
  uint32_t* scatter_lut;                       // [lod_nelem[0]] L0 scatter LUT
  uint64_t* scatter_batch_offsets;             // [batch_count] for L0 gather
  uint32_t* morton_lut[LOD_MAX_LEVELS];        // [lod_nelem[lv]] per level
  uint64_t* lod_batch_offsets[LOD_MAX_LEVELS]; // [batch_count] per level

  // Append downsample accumulation state
  void* append_accum;                     // accumulator for levels 1+
  uint32_t append_counts[LOD_MAX_LEVELS]; // per-level fold count

  uint64_t cursor_elements;
  uint64_t
    max_cursor_elements; // precomputed: total elements across all append chunks
  int pool_fully_covered; // 1 if scatter overwrites every pool position

  // Batch accumulation state (K = cl.epochs_per_batch).
  uint32_t batch_accumulated;                    // 0..K-1
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS]; // per-epoch active level mask

  // IO fence state: tracks pending async IO per level so we don't
  // overwrite aggregate buffers before write_direct completes.
  struct io_event io_done[LOD_MAX_LEVELS];

  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};
