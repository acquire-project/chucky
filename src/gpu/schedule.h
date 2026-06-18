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
struct platform_thread;
struct platform_mutex;
struct platform_cond;
struct compress_agg_array;
struct compress_agg_stage;
struct compress_agg_input;
struct d2h_deliver_stage;
struct level_geometry;
struct shard_sink;
struct dim_info;
struct tile_stream_layout;
struct tile_stream_configuration;
struct lod_state;
struct lod_shared_state;
struct stream_metrics;
struct platform_clock;

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

// Delivery worker: the pipelined schedule queues each kicked batch's drain
// here at kick time and joins it before refilling the slot, so polls and
// sink delivery run off the producer thread. One job slot per fc, queued in
// kick order and run oldest-first (the tail gate's GEQ threshold and the
// GPU_EDGE_DELIVER_OLDEST_FIRST rule). Ownership: the producer writes a job
// between kick and enqueue and reads it after join; the worker owns it in
// between — the same single-writer-per-generation discipline as the pools.
// Engine/ctx pointers stay valid from enqueue to join (single-array only;
// depth-1 schedules and multiarray drain inline and never queue here).
struct delivery_job
{
  int queued;
  int done;
  uint64_t seq;
  struct stream_engine* e;
  struct stream_context* ctx;
  struct writer_result result;
};

struct gpu_delivery
{
  struct platform_thread* thread; // NULL = drains run inline on the producer
  struct platform_mutex* mu;
  struct platform_cond* cv;
  CUcontext cuda; // captured at init; made current on the worker
  int stop;
  int hold; // test-only: worker parks after each job until stop is set, so a
            // later job stays queued and stop_join must run it out
  struct delivery_job job[2]; // by fc
};

// Captures the calling thread's CUDA context and starts the worker.
// Failure leaves the worker absent (drains run inline) — callers may
// degrade rather than abort.
int
gpu_delivery_init(struct gpu_delivery* d);

void
gpu_delivery_enqueue(struct gpu_delivery* d,
                     struct stream_engine* e,
                     struct stream_context* ctx,
                     int fc,
                     uint64_t seq);

int
gpu_delivery_pending(struct gpu_delivery* d, int fc);

struct writer_result
gpu_delivery_join(struct gpu_delivery* d, int fc);

// Runs every queued job to completion (each publishes its tail generation,
// failure or not), then joins the thread. Idempotent; call before any
// forced gate release — a worker publish after release_all would regress
// the published count below parked thresholds.
void
gpu_delivery_stop_join(struct gpu_delivery* d);

// Test-only: when on, the worker parks after completing each job until stop is
// set. Lets a test guarantee a job is still queued when stop_join runs.
void
gpu_delivery_set_hold(struct gpu_delivery* d, int on);

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

// One batch through compress+aggregate: the stage's payload phases with the
// schedule-owned acquires, tail-gate arm, and releases placed between them.
// lod_active queues the second producer edge into the chunk pool.
int
schedule_compress_agg_kick(struct compress_agg_stage* stage,
                           const struct compress_agg_input* in,
                           const struct level_geometry* levels,
                           struct gpu_pool* chunk_pool,
                           int lod_active,
                           CUstream compress_stream,
                           struct flush_handoff* out);

// Slot-reuse fence wait plus the slot acquire around the D2H kick payload.
// Releases the chunk-index facet (compressed) or the slot itself
// (passthrough, whose drain has no chunk index to poll).
int
schedule_d2h_kick(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  struct shard_sink* sink,
                  CUstream d2h_stream);

// Host-acquire the drained slot per codec shape, deliver it to the sink,
// and publish the tail generation exactly once — on failure paths too.
struct writer_result
schedule_d2h_drain(struct d2h_deliver_stage* stage,
                   const struct flush_handoff* handoff,
                   const struct level_geometry* levels,
                   const struct dim_info* dims,
                   const struct tile_stream_layout* layout,
                   const struct tile_stream_configuration* config,
                   struct shard_sink* sink,
                   const struct lod_state* lod,
                   const struct lod_shared_state* lod_shared,
                   struct stream_metrics* metrics,
                   struct platform_clock* metadata_update_clock);

// Quiesce the output slots for the departing sink before another array
// binds in (multiarray): stale fences only retire on the sink that issued
// them.
void
schedule_quiesce_output(struct stream_engine* e, struct shard_sink* sink);

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
