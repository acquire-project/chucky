#pragma once

#include "cpu/pipeline.h"
#include "lod/reduce_csr.h"
#include "platform/platform.h"
#include "stream/layouts.h"
#include "writer.h"
#include "zarr/shard_delivery.h"

// Unified view of CPU stream state for the shared append/flush bodies.
// Built from tile_stream_cpu (single-array) or from
// multiarray_tile_stream_cpu + array_descriptor (multiarray).
struct cpu_stream_view
{
  // Configuration (per-array)
  const struct tile_stream_configuration* config;
  struct shard_sink* sink;
  const struct computed_stream_layouts* cl;
  const struct tile_stream_layout* layout;
  const struct level_geometry* levels;

  // Mutable per-array cursor + batch state
  uint64_t* cursor_elements;
  uint64_t max_cursor_elements;
  uint32_t* batch_accumulated;
  uint32_t* batch_active_masks; // [MAX_BATCH_EPOCHS]
  int pool_fully_covered;

  // Per-array shard/LOD state
  struct shard_state* shard;           // [LOD_MAX_LEVELS] array
  struct aggregate_layout* agg_layout; // [LOD_MAX_LEVELS] array
  uint32_t* batch_active_count;        // [LOD_MAX_LEVELS] array
  struct reduce_csr* csrs;             // [nlod-1] CSR LUTs
  void* append_accum;
  uint32_t* append_counts;  // [LOD_MAX_LEVELS]
  struct io_event* io_done; // [LOD_MAX_LEVELS]

  // Shared buffers
  void* chunk_pool;
  size_t chunk_pool_bytes;
  void* compressed;
  size_t* comp_sizes;
  struct cpu_agg_slot* agg_slots; // [LOD_MAX_LEVELS]
  size_t* shard_order_sizes;
  void* linear;
  void* lod_values;

  // Shared LUTs
  uint32_t* batch_gather[LOD_MAX_LEVELS];
  uint32_t* batch_chunk_to_shard_map[LOD_MAX_LEVELS];
  uint32_t* scatter_lut;
  uint64_t* scatter_fixed_dims_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_fixed_dims_offsets[LOD_MAX_LEVELS];

  // Runtime config
  int nthreads;
  size_t shard_alignment;
  struct stream_metrics* metrics;
  struct platform_clock* metadata_update_clock; // NULL to skip updates
};

// Shared append body: scatter + epoch boundary + batch flush.
// Used by both single-array and multiarray CPU streams.
struct writer_result
cpu_stream_append_body(struct cpu_stream_view* v, struct slice input);

// Shared flush body: partial epoch + batch + append drain + shard finalize +
// metadata. Used by both single-array and multiarray CPU streams.
struct writer_result
cpu_stream_flush_body(struct cpu_stream_view* v);

// Flush only the accumulated batch (no shard finalize or metadata).
// For multiarray array-switch: delivers batch data then resets accumulated.
int
cpu_stream_flush_batch(struct cpu_stream_view* v);
