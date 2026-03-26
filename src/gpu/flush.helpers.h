#pragma once

#include "gpu/stream.internal.h"

// How many epochs fire for level `lv` within a batch of `n_epochs` epochs.
// L0 fires every epoch; append-downsampled level lv fires every 2^lv epochs.
// Returns 0 if the level doesn't fire at all in this batch.
static inline uint32_t
level_active_epochs(const struct level_flush_state* lvl,
                    const struct batch_state* batch,
                    const struct level_geometry* levels,
                    int lv,
                    uint32_t n_epochs)
{
  uint32_t full = lvl->batch_active_count;
  if (n_epochs >= batch->epochs_per_batch)
    return full;
  uint32_t period = (levels->append_downsample && lv > 0) ? (1u << lv) : 1;
  return (n_epochs >= period) ? n_epochs / period : 0;
}

// Count actual active epochs for a level from per-epoch masks.
// For infrequent append-downsampled levels (period > K, batch_active_count == 0),
// level_active_epochs returns 0 even when the level fired.  This function
// falls back to scanning the per-epoch masks in that case.
static inline uint32_t
level_actual_active_count(const struct level_flush_state* lvl,
                          const struct batch_state* batch,
                          const struct level_geometry* levels,
                          const uint32_t* batch_active_masks,
                          int lv,
                          uint32_t n_epochs)
{
  uint32_t n = level_active_epochs(lvl, batch, levels, lv, n_epochs);
  if (n > 0)
    return n;
  // Infrequent level: count from actual per-epoch masks
  for (uint32_t e = 0; e < n_epochs; ++e)
    if (batch_active_masks[e] & (1u << lv))
      n++;
  return n;
}
