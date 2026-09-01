#pragma once

// One owner for GPU orchestration (dev/gpu-orchestration.md): stream
// creation, pipeline depth, which stages run for the active configuration,
// and degraded schedules. Stages stay payload; the schedule places the
// cross-stage acquires/releases and decides when batches are kicked and
// delivered.

#include "gpu/flush.handoff.h"
#include "writer.h"

#include <cuda.h>

struct gpu_ordering;
struct stream_engine;
struct stream_context;
struct platform_thread;
struct platform_mutex;
struct platform_cond;
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
  CUstream payload_copy;
};

int
gpu_streams_init(struct gpu_streams* s);

void
gpu_streams_destroy(struct gpu_streams* s);

void
gpu_streams_sync(const struct gpu_streams* s);

// Lets debug builds check each edge's record/wait stream against its
// declaration.
void
gpu_streams_register(const struct gpu_streams* s, struct gpu_ordering* ord);

// Pipeline depth, delivery placement, and who enqueues aggregation, selected
// once per array. One value per legal combination, so no caller has to check
// two fields to know which schedule it is looking at.
enum schedule_mode
{
  // Depth 2: kick now; deliver a slot only when about to refill it.
  SCHEDULE_PIPELINED_DIRECT = 0,
  // Depth 1: deliver immediately after kicking; no pool swap. Multiarray:
  // double-buffered pipeline state does not compose across array switches.
  SCHEDULE_DELIVER_AFTER_KICK,
};

// Per-slot bookkeeping: the masks composing the batch being filled, then,
// once kicked, the prepared aggregation inputs and handoff awaiting delivery.
struct schedule_slot
{
  uint32_t active_levels_mask;  // union of per-epoch active masks
  uint32_t* batch_active_masks; // [epochs_per_batch]; per-array allocation
  int kicked;
  uint64_t generation;
  int64_t delivery_submitted_ns;

  // Retained through submission and oldest-first host copying. The views
  // remain owned by their pools; this slot carries their generation.
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
  enum schedule_mode mode;
  int lod_active; // multiscale: LOD is a second producer into the chunk pool
  uint32_t epochs_per_batch;
  uint32_t accumulated;     // epochs in the batch being filled
  int fill;                 // slot being filled (chunk pool + masks)
  uint64_t next_generation; // monotonic, starts at 1
  struct schedule_slot slot[2];
};

enum delivery_job_state
{
  DELIVERY_JOB_EMPTY = 0,
  DELIVERY_JOB_SUBMITTED,
  DELIVERY_JOB_DONE,
};

struct delivery_job
{
  enum delivery_job_state state;
  uint64_t generation;
  int64_t submitted_ns;
  struct stream_engine* e;
  struct stream_context* ctx;
  struct writer_result result;
};

// Test-only: where the worker parks so a test can observe a handoff that is
// otherwise over before it can be read.
enum delivery_hold_point
{
  DELIVERY_HOLD_NONE = 0,
  DELIVERY_HOLD_BEFORE_DELIVERY,
  DELIVERY_HOLD_AFTER_DELIVERY,
};

struct gpu_delivery
{
  struct platform_thread* thread; // NULL = delivery runs inline on the producer
  struct platform_mutex* mu;
  struct platform_cond* cv;
  CUcontext cuda; // the engine's, made current on the worker
  int stop;
  enum delivery_hold_point hold_at;
  int sticky_error;
  uint64_t submitted_generation;
  struct delivery_job job[2]; // by fc
};

// Captures the calling thread's CUDA context and starts the worker.
// Failure leaves the worker absent; callers may degrade rather than abort.
int
gpu_delivery_init(struct gpu_delivery* d, CUcontext cuda);

int
gpu_delivery_enqueue_submitted(struct gpu_delivery* d,
                               struct stream_engine* e,
                               struct stream_context* ctx,
                               int fc,
                               uint64_t generation,
                               int64_t submitted_ns);

int
gpu_delivery_pending(struct gpu_delivery* d, int fc);

struct writer_result
gpu_delivery_join(struct gpu_delivery* d, int fc);

// Runs every valid queued job to completion and cancels jobs made invalid by
// an earlier coordinator/sink failure, then joins the thread. Idempotent.
void
gpu_delivery_stop_join(struct gpu_delivery* d);

void
gpu_delivery_set_hold(struct gpu_delivery* d, enum delivery_hold_point at);

enum delivery_job_state
gpu_delivery_job_state(struct gpu_delivery* d, int fc, uint64_t* generation);

uint64_t
gpu_delivery_submitted_generation(struct gpu_delivery* d);

// delivery == NULL selects the multi-array immediate-delivery path. Every
// single-array configuration otherwise uses the same depth-two schedule.
void
schedule_select(struct gpu_scheduler* sched,
                const struct gpu_delivery* delivery);

// The LOD producer edge exists only when the engine allocated shared LOD
// resources; enable_multiscale alone is a per-array fact. Selected at bind.
int
schedule_lod_active(const struct gpu_ordering* ord, int enable_multiscale);

// Prepare compression and retain the aggregation inputs in `slot`.
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

// Direct compatibility path used by stage tests.
int
schedule_compress_agg_kick(struct compress_agg_stage* stage,
                           const struct compress_agg_input* in,
                           const struct level_geometry* levels,
                           struct gpu_pool* chunk_pool,
                           int lod_active,
                           CUstream compress_stream,
                           struct flush_handoff* out);

// Slot-reuse fence wait plus the host-copy stage begin lifecycle.  The selected
// host-copy stage owns device/index pool leases and every codec-specific copy
// decision behind this call.
int
schedule_d2h_kick(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  CUstream d2h_stream);

// Finish host copying and deliver its normalized host batch to the sink.
struct writer_result
schedule_deliver_batch(struct d2h_deliver_stage* stage,
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

// Count one epoch into the batch being filled; kick (and, per depth, deliver)
// at the batch boundary.
struct writer_result
schedule_accumulate_epoch(struct stream_engine* e, struct stream_context* ctx);

// Count a final partial epoch into the batch without kicking; the caller
// flushes immediately after.
int
schedule_add_partial_epoch(struct stream_engine* e, struct stream_context* ctx);

// Drain every kicked slot, oldest first.
struct writer_result
schedule_deliver_kicked(struct stream_engine* e, struct stream_context* ctx);

// Kick and deliver the accumulated (possibly partial) batch; no pool swap.
struct writer_result
schedule_flush_accumulated(struct stream_engine* e, struct stream_context* ctx);

// Emit and flush partial append-dim LOD accumulators on final flush.
struct writer_result
schedule_flush_partial_append(struct stream_engine* e,
                              struct stream_context* ctx);
