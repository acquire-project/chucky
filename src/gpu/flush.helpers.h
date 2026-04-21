#pragma once

#include "gpu/stream.internal.h"
#include "stream/dim_info.h"

// How many epochs fire for level `lv` within a batch of `n_epochs` epochs.
// L0 fires every epoch; append-downsampled level lv fires every 2^lv epochs.
// Returns 0 if the level doesn't fire at all in this batch.
static inline uint32_t
level_active_epochs(const struct level_flush_state* lvl,
                    const struct batch_state* batch,
                    const struct dim_info* dims,
                    int lv,
                    uint32_t n_epochs)
{
  uint32_t full = lvl->batch_active_count;
  if (n_epochs >= batch->epochs_per_batch)
    return full;
  uint32_t period = (dims->append_downsample && lv > 0) ? (1u << lv) : 1;
  return (n_epochs >= period) ? n_epochs / period : 0;
}

// Count actual active epochs for a level from per-epoch masks.
// Always scans masks: the steady-state pattern shifts between batches when
// K doesn't divide the level period, so lvl->batch_active_count is only a
// safe upper bound (for buffer sizing), not the current batch's actual count.
// Callers must pass masks that outlive any later pool/slot reuse — see
// struct flush_handoff::active_counts for the pre-computed, lifetime-safe
// alternative used by d2h delivery.
static inline uint32_t
level_actual_active_count(const struct level_flush_state* lvl,
                          const struct batch_state* batch,
                          const struct dim_info* dims,
                          const uint32_t* batch_active_masks,
                          int lv,
                          uint32_t n_epochs)
{
  (void)lvl;
  (void)batch;
  (void)dims;
  uint32_t n = 0;
  for (uint32_t e = 0; e < n_epochs; ++e)
    if (batch_active_masks[e] & (1u << lv))
      n++;
  return n;
}
