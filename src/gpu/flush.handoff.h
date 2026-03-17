#pragma once

#include "aggregate.h"
#include "defs.limits.h"
#include "lod_plan.h"

#include <cuda.h>
#include <stdint.h>

// Handoff from compress+aggregate to D2H+deliver.
struct flush_handoff
{
  int fc;                                        // flush slot index
  uint32_t n_epochs;                             // epochs in batch
  uint32_t active_levels_mask;                   // which levels active
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS]; // per-epoch masks

  CUevent t_aggregate_end;  // D2H waits on this
  CUevent t_compress_start; // for metrics
  CUevent t_compress_end;   // for metrics

  struct aggregate_slot* agg[LOD_MAX_LEVELS]; // borrowed
  const struct aggregate_layout* agg_layout[LOD_MAX_LEVELS];
  size_t max_output_size; // codec bound
};
