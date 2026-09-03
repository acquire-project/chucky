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
  uint64_t total_element_limit;
  uint32_t* batch_accumulated;
  uint32_t* batch_active_masks;  // [K]
  uint32_t* pool_epochs_scratch; // [K] scratch for LUT recompute
  int pool_fully_covered;

  // Per-array shard/LOD state
  struct shard_state* shard;           // [LOD_MAX_LEVELS] array
  struct aggregate_layout* agg_layout; // [LOD_MAX_LEVELS] array
  struct reduce_csr* csrs;             // [nlod-1] CSR LUTs
  void* append_accum;
  uint32_t* append_counts; // [LOD_MAX_LEVELS]

  uint8_t* agg_current; // single byte (slot 0 or 1)

  // Shared buffers
  void* chunk_pool;
  void* compressed;
  size_t* comp_sizes;
  struct cpu_agg_slot* agg_slots; // [2] unified per-batch workspace
  struct host_output_pool* output_pool;
  void* linear;
  void* lod_values;

  // Shared LUTs
  uint32_t* scatter_lut;
  uint64_t* scatter_fixed_dims_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_fixed_dims_offsets[LOD_MAX_LEVELS];

  // Runtime config
  struct threadpool* pool;
  size_t shard_alignment;
  struct stream_metrics* metrics;
  struct platform_clock* metadata_update_clock; // NULL to skip updates
};

// Shared append body: scatter + epoch boundary + batch flush.
// Used by both single-array and multiarray CPU streams.
struct writer_result
cpu_stream_append_body(struct cpu_stream_view* v, struct slice input);

// Shared flush body: partial epoch + batch + append drain + shard finalize.
// Used by both single-array and multiarray CPU streams. Drains every write it
// queues before returning.
struct writer_result
cpu_stream_flush_body(struct cpu_stream_view* v);

// Waits for queued writes to retire, then publishes the append extent and lets
// the sink write its own metadata.
struct writer_result
cpu_stream_close_body(struct cpu_stream_view* v);

// Flush only the accumulated batch (no shard finalize or metadata).
// For multiarray array-switch: delivers batch data then resets accumulated.
int
cpu_stream_flush_batch(struct cpu_stream_view* v);
