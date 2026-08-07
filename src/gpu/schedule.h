#pragma once

// One owner for GPU orchestration (dev/gpu-orchestration.md): stream
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
  // Exact-size payload copies are dispatched only after their metadata is
  // host-complete, so they use a stream separate from the metadata D2H queue.
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
  // Depth 1: drain every kicked slot before kicking. Used when the delivery
  // worker is unavailable, so tail uploads are ordered on the producer.
  SCHEDULE_DRAIN_BEFORE_KICK,
  // Depth 1: drain immediately after kicking; no pool swap. Multiarray:
  // double-buffered pipeline state does not compose across array switches.
  SCHEDULE_DRAIN_AFTER_KICK,
};

// Per-slot bookkeeping: the masks composing the batch being filled, then,
// once kicked, the prepared aggregation inputs and handoff awaiting drain.
struct schedule_slot
{
  uint32_t active_levels_mask;  // union of per-epoch active masks
  uint32_t* batch_active_masks; // [epochs_per_batch]; per-array allocation
  int kicked;
  uint64_t generation;

  // Retained between PREPARED and SUBMITTED for host-coordinated,
  // page-aligned batches. The views remain owned by their pools; this slot
  // merely carries the acquired generation until aggregation is enqueued.
  struct compress_agg_plan plan;
  struct gpu_pool_view pool_buf;
  struct gpu_pool_view aggregate_slot;
  struct flush_handoff handoff;
};

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// Schedule selections plus batch/slot progress. All per-array: multiarray
// swaps the whole struct on array switch. Codec shape rides each handoff.
struct gpu_scheduler
{
  enum schedule_depth depth;
  int host_coordinated; // split prepare/submit for page-aligned single-array
  int lod_active; // multiscale: LOD is a second producer into the chunk pool
  uint32_t epochs_per_batch;
  uint32_t accumulated; // epochs in the batch being filled
  int fill;             // slot being filled (chunk pool + masks)
  uint64_t next_generation; // monotonic, starts at 1
  struct schedule_slot slot[2];
};

// Page-aligned single-array batches enter PREPARED after compression and
// become SUBMITTED only when the preceding generation's tail upload is
// host-complete. Contiguous batches enter SUBMITTED directly. All jobs drain
// oldest-first and are joined before their slot is refilled.
enum delivery_job_state
{
  DELIVERY_JOB_EMPTY = 0,
  DELIVERY_JOB_PREPARED,
  DELIVERY_JOB_SUBMITTED,
  DELIVERY_JOB_DONE,
};

struct delivery_job
{
  enum delivery_job_state state;
  uint64_t generation;
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
  int hold; // test-only: park after a drain for teardown coverage
  int hold_before_drain; // test-only: park a SUBMITTED job before its drain
  int sticky_error;
  uint64_t submitted_generation;
  uint64_t tail_ready_generation;
  struct delivery_job job[2]; // by fc
};

// Captures the calling thread's CUDA context and starts the worker.
// Failure leaves the worker absent; single-array scheduling degrades to
// drain-before-kick.
int
gpu_delivery_init(struct gpu_delivery* d);

int
gpu_delivery_enqueue_prepared(struct gpu_delivery* d,
                              struct stream_engine* e,
                              struct stream_context* ctx,
                              int fc,
                              uint64_t generation);

int
gpu_delivery_enqueue_submitted(struct gpu_delivery* d,
                               struct stream_engine* e,
                               struct stream_context* ctx,
                               int fc,
                               uint64_t generation);

// Shared-buffer barrier: wait until aggregation for `generation` has been
// enqueued, or return failure if the coordinator has failed.
int
gpu_delivery_wait_submitted(struct gpu_delivery* d, uint64_t generation);

int
gpu_delivery_pending(struct gpu_delivery* d, int fc);

struct writer_result
gpu_delivery_join(struct gpu_delivery* d, int fc);

// Runs every valid queued job to completion and cancels jobs made invalid by
// an earlier coordinator/sink failure, then joins the thread. Idempotent.
void
gpu_delivery_stop_join(struct gpu_delivery* d);

// Test-only: park after a drain so another queued job survives until teardown.
void
gpu_delivery_set_hold(struct gpu_delivery* d, int on);

// Test-only: park after submission and before drain/tail publication.
void
gpu_delivery_set_hold_before_drain(struct gpu_delivery* d, int on);

enum delivery_job_state
gpu_delivery_job_state(struct gpu_delivery* d,
                       int fc,
                       uint64_t* generation);

void
gpu_delivery_generations(struct gpu_delivery* d,
                         uint64_t* submitted,
                         uint64_t* tail_ready);

// Depth selection from the array and worker availability. delivery == NULL is
// the multi-array immediate-drain path. An absent worker selects depth one.
void
schedule_select(struct gpu_scheduler* sched,
                const struct compress_agg_array* ar,
                const struct gpu_delivery* delivery);

// The LOD producer edge exists only when the engine allocated shared LOD
// resources; enable_multiscale alone is a per-array fact. Selected at bind.
int
schedule_lod_active(const struct gpu_ordering* ord, int enable_multiscale);

// Prepare compression and retain the aggregation inputs in `slot` without
// enqueueing aggregation.
int
schedule_compress_agg_prepare(struct compress_agg_stage* stage,
                              const struct compress_agg_input* in,
                              const struct level_geometry* levels,
                              struct gpu_pool* chunk_pool,
                              int lod_active,
                              CUstream compress_stream,
                              struct schedule_slot* slot);

// Enqueue aggregation for a prepared slot and publish AGG_DONE/POOL_CONSUMED.
int
schedule_compress_agg_submit(struct compress_agg_stage* stage,
                             struct schedule_slot* slot,
                             CUstream compress_stream);

// Direct compatibility path used by contiguous pipelines and stage tests.
// Callers are responsible for ensuring any page-aligned tail state is ready.
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

// Host-acquire the drained slot per codec shape, deliver it to the sink, and
// synchronously upload page-aligned tail state.
struct writer_result
schedule_d2h_drain(struct d2h_deliver_stage* stage,
                   const struct flush_handoff* handoff,
                   const struct level_geometry* levels,
                   const struct dim_info* dims,
                   const struct tile_stream_layout* layout,
                   const struct tile_stream_configuration* config,
                   struct shard_sink* sink,
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
