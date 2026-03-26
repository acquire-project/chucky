#pragma once

#include "gpu/stream.internal.h"

// Drain pending flush from the previous batch.
struct writer_result
flush_drain_pending(struct tile_stream_gpu* s);

// Kick compress->aggregate->D2H for a batch of n_epochs epochs.
// fc: flush slot index (matches pool index before swap).
int
flush_kick_batch(struct tile_stream_gpu* s, int fc, uint32_t n_epochs);

// Synchronously flush accumulated epochs (partial or full batch).
struct writer_result
flush_accumulated_sync(struct tile_stream_gpu* s);

// Drain partial append-dim accumulators on final flush.
struct writer_result
flush_partial_append(struct tile_stream_gpu* s);

// Compute active mask for current epoch (runs LOD if multiscale).
// Updates flush slot active masks. Returns 0 on success.
int
flush_run_epoch_lod(struct tile_stream_gpu* s);

// Accumulate one epoch into the current batch:
// 1. Compute epoch mask (LOD or all-active)
// 2. Record pool event
// 3. Increment batch.accumulated
// 4. If batch full: drain pending, kick new flush, swap pools, reset
// Returns writer_result.
struct writer_result
flush_accumulate_epoch(struct tile_stream_gpu* s);
