#pragma once

#include "gpu/aggregate.h"
#include "lod/lod_plan.h"

#include <cuda.h>
#include <stdint.h>

// Handoff from compress+aggregate to D2H+deliver.
// batch_active_masks is borrowed from the flush_slot_gpu that produced this
// batch. The slot's masks aren't reset until after d2h_deliver_drain completes,
// so the borrow is safe for the lifetime of the handoff.
struct flush_handoff
{
  int fc;                             // flush slot index
  uint32_t n_epochs;                  // epochs in batch
  uint32_t active_levels_mask;        // which levels active
  const uint32_t* batch_active_masks; // borrowed [K] per-epoch masks

  CUevent t_aggregate_end;  // D2H waits on this
  CUevent t_compress_start; // for metrics
  CUevent t_compress_end;   // for metrics

  struct aggregate_slot* agg[LOD_MAX_LEVELS]; // borrowed
  const struct aggregate_layout* agg_layout[LOD_MAX_LEVELS];
  size_t max_output_size; // codec bound
};
