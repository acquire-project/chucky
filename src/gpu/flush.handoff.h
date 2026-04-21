#pragma once

#include "gpu/aggregate.h"
#include "lod/lod_plan.h"

#include <cuda.h>
#include <stdint.h>

// Handoff from compress+aggregate to D2H+deliver.
// batch_active_masks is borrowed from the flush_slot_gpu that produced this
// batch and is only safe to read inside compress+aggregate (before the slot's
// masks buffer can be reused by the next batch at the same fc).  D2H delivery
// runs later (potentially after a swap has already reset the slot), so it
// reads from active_counts[lv], which is computed here at kick time.
struct flush_handoff
{
  int fc;                                 // flush slot index
  uint32_t n_epochs;                      // epochs in batch
  uint32_t active_levels_mask;            // which levels active
  const uint32_t* batch_active_masks;     // borrowed [K] per-epoch masks
  uint32_t active_counts[LOD_MAX_LEVELS]; // owned: per-level active epoch count

  CUevent t_aggregate_end;  // D2H waits on this
  CUevent t_compress_start; // for metrics
  CUevent t_compress_end;   // for metrics

  struct aggregate_slot* agg[LOD_MAX_LEVELS]; // borrowed
  const struct aggregate_layout* agg_layout[LOD_MAX_LEVELS];
  size_t max_output_size; // codec bound
};
