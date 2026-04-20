#pragma once

#include "gpu/stream.engine.h"

// Drain pending flush from the previous batch.
struct writer_result
flush_drain_pending(struct stream_engine* e, struct stream_context* ctx);

// Kick compress->aggregate->D2H for a batch of n_epochs epochs.
// fc: flush slot index (matches pool index before swap).
int
flush_kick_batch(struct stream_engine* e,
                 struct stream_context* ctx,
                 int fc,
                 uint32_t n_epochs);

// Synchronously flush accumulated epochs (partial or full batch).
struct writer_result
flush_accumulated_sync(struct stream_engine* e, struct stream_context* ctx);

// Drain partial append-dim accumulators on final flush.
struct writer_result
flush_partial_append(struct stream_engine* e, struct stream_context* ctx);

// Compute active mask for current epoch (runs LOD if multiscale).
// Updates flush slot active masks. Returns 0 on success.
int
flush_run_epoch_lod(struct stream_engine* e, struct stream_context* ctx);

// Accumulate one epoch into the current batch:
// 1. Compute epoch mask (LOD or all-active)
// 2. Record pool event
// 3. Increment batch.accumulated
// 4. If batch full: drain pending, kick new flush, swap pools, reset
// Returns writer_result.
struct writer_result
flush_accumulate_epoch(struct stream_engine* e, struct stream_context* ctx);
