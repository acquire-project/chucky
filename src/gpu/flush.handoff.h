#pragma once

#include "gpu/aggregate.h"
#include "gpu/pool.h"
#include "lod/lod_plan.h"
#include "stream/types.aggregate.h"

#include <cuda.h>
#include <stdint.h>

struct shard_state;
struct compress_agg_array;

// Handoff from compress+aggregate to D2H+deliver.
//
// Post-unification, the GPU pipeline runs one aggregate dispatch over all
// LODs producing a single unified slot. Per-LOD geometry is encoded in
// `layout` (per-LOD segment offsets within the unified buffer) and the
// borrowed `per_lod_agg_layouts` array. `per_lod_n_active[lv]` carries
// the per-LOD active epoch count for delivery sizing.
//
// batch_active_masks is borrowed from the schedule_slot that produced this
// batch and is safe to read only inside compress+aggregate. D2H delivery
// must read per-LOD active counts from `per_lod_n_active` instead.
struct flush_handoff
{
  int fc;                                    // flush slot index
  uint32_t n_epochs;                         // epochs in batch
  uint32_t active_levels_mask;               // which levels active
  const uint32_t* batch_active_masks;        // borrowed [K] per-epoch masks
  uint32_t per_lod_n_active[LOD_MAX_LEVELS]; // owned, for delivery sizing
  uint8_t nlod;

  CUevent t_aggregate_end;   // D2H waits on this
  CUevent t_compress_start;  // for metrics
  CUevent t_compress_end;    // for metrics
  CUevent t_aggregate_start; // for metrics; after the tail-gate wait

  // The unified slot for fc travels as pool handles, never a raw pointer;
  // delivery acquires the facet it consumes (stream.engine.h).
  struct gpu_pool* agg_pool;            // borrowed
  struct gpu_pool* agg_host;            // borrowed
  struct gpu_pool* agg_index;           // borrowed
  struct batch_aggregate_layout layout; // owned (by-value snapshot)
  const struct aggregate_layout* per_lod_agg_layouts; // borrowed [nlod]
  struct shard_state* shards_by_lod[LOD_MAX_LEVELS];  // borrowed
  struct gpu_pool* tail;  // tail-state pool (#142); the drain's produce-
                          // acquire yields the compress_agg_array whose
                          // d_tail_bytes/d_tail_carry the delivery uploads
  size_t max_output_size; // codec bound

  // Pass-through codec (CODEC_NONE): per-LOD bytes equal worst-case, so
  // delivery skips the exact-size sync and keeps the kick-time bulk D2H
  // path that overlaps with the next batch's compute.
  uint8_t passthrough;
};
