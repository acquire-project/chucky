#pragma once

#include "cpu/pipeline.h"
#include "cpu/stream.body.h"
#include "lod/reduce_csr.h"
#include "platform/platform.h"
#include "stream.cpu.h"
#include "stream/host_output_pool.h"
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

  // Two aggregate scratch sets shared across all LODs. Host output buffers
  // have an independent bounded pool and remain leased until their borrowed
  // writes retire.
  struct cpu_agg_slot agg_slots[2];
  struct host_output_pool* output_pool;
  uint8_t agg_current; // next slot to use (0 or 1)

  // LOD (multiscale only)
  void* linear; // linear epoch buffer (input accumulated here before scatter)
  void* lod_values; // morton-ordered LOD buffer (all levels packed)

  // Precomputed LOD LUTs (multiscale only)
  uint32_t* scatter_lut;                // [lod_nelem[0]] L0 scatter LUT
  uint64_t* scatter_fixed_dims_offsets; // [fixed_dims_count] for L0 gather
  uint32_t* morton_lut[LOD_MAX_LEVELS]; // [lod_nelem[lv]] per level
  uint64_t*
    lod_fixed_dims_offsets[LOD_MAX_LEVELS]; // [fixed_dims_count] per level

  // CSR reduce LUTs (multiscale only): nlod-1 entries, owned.
  struct reduce_csr* csrs;

  // Append downsample accumulation state
  void* append_accum;                     // accumulator for levels 1+
  uint32_t append_counts[LOD_MAX_LEVELS]; // per-level fold count

  uint64_t cursor_elements;
  uint64_t total_element_limit; // configured stream length in elements (across
                                // all append chunks). 0 = unbounded.
  int pool_fully_covered;       // 1 if scatter overwrites every pool position
  int flushed;                  // 1 once finalized; no further input is taken
  int flush_failed; // outcome of that finalize, re-reported by later calls
  int closed;       // 1 once close drained and, on success, published
  int close_failed; // outcome of that close, re-reported by later calls

  // Batch accumulation state (K = cl.epochs_per_batch).
  uint32_t batch_accumulated;    // 0..K-1
  uint32_t* batch_active_masks;  // [K] per-epoch active level mask
  uint32_t* pool_epochs_scratch; // [K] scratch for kick-time mask scans

  struct threadpool* pool; // owned by stream
  size_t shard_alignment;  // from sink; 0 = no alignment

  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};
