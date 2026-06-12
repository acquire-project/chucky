#pragma once

// One owner for GPU orchestration (docs/gpu-orchestration.md §4): stream
// creation, pipeline depth, which stages run for the active configuration,
// and degraded schedules. Stages stay payload; the schedule places the
// cross-stage acquires/releases and decides when kicks and drains happen.

#include "gpu/flush.handoff.h"
#include "writer.h"

#include <cuda.h>

struct gpu_ordering;
struct stream_engine;
struct stream_context;
struct compress_agg_array;

struct gpu_streams
{
  CUstream h2d, compute, compress, d2h;
  // Drain-time copies must not share the d2h stream: by drain time it can
  // already hold the next kick's GPU_EDGE_AGG_DONE wait, which the tail
  // gate keeps parked until that drain publishes — sharing would deadlock.
  CUstream drain;
};

int
gpu_streams_init(struct gpu_streams* s);

void
gpu_streams_destroy(struct gpu_streams* s);

// Lets debug builds check each edge's record/wait stream against its
// declaration.
void
gpu_streams_register(const struct gpu_streams* s, struct gpu_ordering* ord);

// Pipeline depth and drain placement, selected once per array.
enum schedule_depth
{
  // Depth 2: kick now; drain a slot only when about to refill it.
  SCHEDULE_PIPELINED = 0,
  // Depth 1: drain every kicked slot before kicking. Page-aligned tail
  // state without a working gate cannot be ordered device-side, so the
  // drain's tail upload must complete before the next kick is enqueued.
  SCHEDULE_DRAIN_BEFORE_KICK,
  // Depth 1: drain immediately after kicking; no pool swap. Multiarray —
  // double-buffered pipeline state doesn't compose across array switches,
  // and the immediate drain host-orders the tail uploads.
  SCHEDULE_DRAIN_AFTER_KICK,
};

// Per-slot bookkeeping: the masks composing the batch being filled, then —
// once kicked — the handoff awaiting drain.
struct schedule_slot
{
  uint32_t active_levels_mask;  // union of per-epoch active masks
  uint32_t* batch_active_masks; // [epochs_per_batch]; per-array allocation
  int kicked;
  uint64_t kick_seq;
  struct flush_handoff handoff;
};

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// Schedule selections plus batch/slot progress. All per-array: multiarray
// swaps the whole struct on array switch. The other stage-shape selection,
// passthrough vs compressed, rides each batch's handoff (set from the
// codec; consumed by the schedule's d2h kick/drain placement).
struct gpu_scheduler
{
  enum schedule_depth depth;
  int lod_active; // multiscale: LOD is a second producer into the chunk pool
  uint32_t epochs_per_batch;
  uint32_t accumulated; // epochs in the batch being filled
  int fill;             // slot being filled (chunk pool + masks)
  uint64_t next_seq;
  struct schedule_slot slot[2];
};

// Depth selection from the array's configuration. gate_ord NULL means
// drains host-order the tail uploads (multiarray); call after the array's
// tail gate has been armed so gate support is known.
void
schedule_select(struct gpu_scheduler* sched,
                const struct compress_agg_array* ar,
                const struct gpu_ordering* gate_ord);

// The LOD producer edge exists only when the engine allocated shared LOD
// resources; enable_multiscale alone is a per-array fact. Selected at bind.
int
schedule_lod_active(const struct gpu_ordering* ord, int enable_multiscale);

// Count one epoch into the batch being filled; kick (and, per depth, drain)
// at the batch boundary.
struct writer_result
schedule_accumulate_epoch(struct stream_engine* e, struct stream_context* ctx);

// Count a final partial epoch into the batch without kicking; the caller
// flushes immediately after.
int
schedule_add_partial_epoch(struct stream_engine* e, struct stream_context* ctx);

// Drain every kicked slot, oldest first.
struct writer_result
schedule_drain_kicked(struct stream_engine* e, struct stream_context* ctx);

// Kick and drain the accumulated (possibly partial) batch; no pool swap.
struct writer_result
schedule_flush_accumulated(struct stream_engine* e, struct stream_context* ctx);

// Emit and flush partial append-dim LOD accumulators on final flush.
struct writer_result
schedule_flush_partial_append(struct stream_engine* e,
                              struct stream_context* ctx);
