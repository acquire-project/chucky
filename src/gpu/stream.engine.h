#pragma once

#include "gpu/aggregate.h"
#include "gpu/compress.h"
#include "gpu/d2h.materializer.h"
#include "gpu/ordering.h"
#include "gpu/pool.h"
#include "gpu/reduce_csr_gpu.h"
#include "gpu/schedule.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "zarr/shard_delivery.h"
#include <stddef.h>

struct threadpool;
struct stream_context;

// --- Sub-struct definitions (shared between engine and internal headers) ---

struct pool_state
{
  struct gpu_pool p;  // buf generations: ready=POOL_FILLED,
                      //                  consumed=POOL_CONSUMED (#140)
  CUdeviceptr buf[2]; // payloads; non-init code goes through p
};

// Ordering events (h2d-done, scatter-done) live in gpu_ordering, instanced
// by staging slot; only timing-interval starts stay here. h_in/d_in are the
// pool payloads — non-init code reaches them through d_pool/h_pool only.
struct staging_slot
{
  void* h_in;              // pinned host, size = buffer_capacity_bytes
  CUdeviceptr d_in;        // device, size = buffer_capacity_bytes
  CUevent t_h2d_start;     // recorded before H2D memcpy (timing)
  size_t dispatched_bytes; // bytes transferred in last dispatch
  int h2d_pending;         // dispatched, interval not yet folded into metrics
};

// One scatter measurement, owning both ends of its own interval so it can
// outlive the staging slot that produced it.
struct scatter_timing
{
  CUevent t_start;
  CUevent t_end;
  size_t bytes;
  int pending;
};

struct staging_state
{
  struct staging_slot slot[2];
  struct gpu_pool d_pool; // d_in: ready=STAGING_H2D_DONE,
                          //       consumed=STAGING_SCATTER_DONE
  struct gpu_pool h_pool; // h_in: consumed=STAGING_FREE; ready is host call
                          //       order (fill precedes dispatch)
  int current;            // 0 or 1: which buffer the host is filling
  size_t bytes_written;   // bytes written to current slot's h_in so far

  // One buffer serves every array of a multiarray stream, so its bytes belong
  // to one array at a time. NULL while it holds none.
  struct stream_context* owner;
  uint64_t first_element; // append position of the first staged byte

  struct scatter_timing timing[SCATTER_TIMING_SLOTS];
  int next_timing;
  uint64_t scatter_samples_lost; // ring wrapped while still outstanding
};

// One epoch's LOD phase boundaries.
struct lod_timing
{
  CUevent t_start;
  CUevent t_scatter_end;
  CUevent t_reduce_end;
  CUevent t_append_end;
  CUevent t_end;
  int pending;         // recorded, not yet folded into metrics
  int has_append_fold; // the append-fold phase ran this epoch

  // Captured when the epoch is recorded. The geometry they come from is
  // per-array and does not outlive a switch to another array.
  size_t epoch_bytes;
  size_t reduced_bytes;
  size_t morton_bytes;
  size_t pool_bytes;
  size_t folded_bytes;
  size_t emitted_bytes;
};

// Engine-owned LOD resources shared across all arrays in a multiarray stream
// (sized to the max requirement).  For a single-array stream these are still
// owned by the engine and sized for that one array.
struct lod_shared_state
{
  CUdeviceptr d_linear; // linear epoch buffer (device)
  CUdeviceptr d_morton; // morton-ordered LOD output (all levels packed)

  // Recorded every epoch and never rotated, so a bound handle stays valid.
  CUevent lod_done[2];

  // Rotated per epoch and read by the producer, which is also the only writer
  // — so the cross-thread collision #154 worked around cannot arise.
  struct lod_timing timing[LOD_TIMING_SLOTS];
  int next_timing_slot;
  uint64_t timing_samples_lost; // ring wrapped while still unread
};

// Per-array LOD state.  In multiarray, one instance per array; the active
// array's state is copied into the engine on bind.
struct lod_state
{
  struct lod_plan plan;

  CUdeviceptr d_full_shape;         // device copy of shapes[0]
  CUdeviceptr d_lod_shape;          // device copy of LOD-projected shapes[0]
  CUdeviceptr d_gather_lut;         // u32, lod_nelem[0] entries
  CUdeviceptr d_fixed_dims_offsets; // u32, fixed_dims_count entries

  // CSR reduce LUTs (precomputed, one per level transition).
  struct reduce_csr_gpu csrs[LOD_MAX_LEVELS];

  // Per-level chunk layouts [0..nlod-1]
  struct tile_stream_layout layouts[LOD_MAX_LEVELS];

  // Morton-to-chunk scatter LUTs (precomputed)
  CUdeviceptr d_morton_chunk_lut[LOD_MAX_LEVELS];
  CUdeviceptr d_morton_fixed_dims_chunk_offsets[LOD_MAX_LEVELS];

  // Append-dim LOD accumulation state.
  struct
  {
    CUdeviceptr d_accum;
    CUdeviceptr d_level_ids;
    CUdeviceptr d_counts;
    uint32_t counts[LOD_MAX_LEVELS];
    uint64_t element_capacity;
    uint64_t morton_offset;
  } append_accum;
};

// --- Stage types ---

struct compress_agg_input
{
  int fc;
  uint32_t n_epochs;
  uint32_t active_levels_mask;
  const uint32_t* batch_active_masks; // borrowed from schedule_slot [K]
  uint32_t epochs_per_batch;
};

// Per-array slice of compress+aggregate state. Multiarray swaps this into
// the engine wholesale on array switch; any new per-array field added here
// rides along with that single assignment.
struct compress_agg_array
{
  // Immutable per array; read by the host LUT builder and at delivery.
  struct aggregate_layout per_lod_agg_layouts[LOD_MAX_LEVELS];
  uint8_t nlod;

  // Fixed-extent persistent tails live only inside these host shard states.
  // Indexed aligned delivery retains zero padding instead; aggregation stays
  // compact and never consumes tail length or content.
  struct shard_state shard[LOD_MAX_LEVELS];
  uint64_t total_shards; // immutable sum, useful for status/tests
};

struct compress_agg_stage
{
  struct gpu_ordering* ord; // borrowed
  struct codec codec;
  CUdeviceptr d_compressed[2];
  CUevent t_compress_start[2];  // timing
  CUevent t_compress_end[2];    // timing
  CUevent t_aggregate_start[2]; // aggregation timing start

  uint32_t* pool_epochs_scratch; // [LOD_MAX_LEVELS * K] scratch for mask scans

  // Unified aggregate slot (per-fc), sized to max_batch_layout maxima. Holds:
  //   d_aggregated:     max compact payload capacity
  //   h_aggregated:     larger host-run/alignment capacity
  //   d_offsets/sizes:  total_batch_covering + LOD_MAX_LEVELS (+ 1 for offsets)
  //   plus pinned host shadows of matching size.
  // Non-init code reaches a slot through the pools below; each guards a
  // different facet of the same payload for a different consumer.
  struct aggregate_slot agg[2];
  struct gpu_pool agg_pool;  // device facet: ready=AGG_DONE,
                             //               consumed=SLOT_DRAINED
  struct gpu_pool agg_host;  // h_aggregated facet: ready=D2H_DONE (alias);
                             // consumed is the drain-before-rekick host rule
  struct gpu_pool agg_index; // h_offsets/h_permuted_sizes facet:
                             // ready=CHUNK_INDEX_READY (compressed only)
  size_t max_total_batch_chunks;
  size_t max_total_batch_covering;
  size_t max_device_data_bytes;
  size_t max_host_data_bytes;

  // Unified LUTs. Sized to max_total_batch_chunks. Uploaded per kick when the
  // firing pattern shifts; cached in steady state by comparing both the
  // per-LOD active counts AND each LOD's pool_epoch values, since the gather
  // LUT encodes the actual pool_epoch values (mid-stream phase shifts can
  // leave the counts unchanged while the epoch positions move).
  CUdeviceptr d_batch_gather;
  CUdeviceptr d_batch_perm;
  uint32_t* h_lut_gather_scratch; // for building unified LUT host-side
  uint32_t* h_lut_perm_scratch;
  uint32_t cached_per_lod_n_active[LOD_MAX_LEVELS]; // last uploaded counts
  uint32_t* cached_pool_epochs; // [LOD_MAX_LEVELS * pool_epochs_stride]
  uint32_t pool_epochs_stride;  // max K used by scratch + cache
  int lut_cache_valid;

  // Per-array state; swapped on multiarray bind.
  struct compress_agg_array ar;
};

struct d2h_deliver_stage
{
  struct d2h_materializer materializer;

  size_t shard_alignment; // from sink; 0 = no alignment
};

// All per-array mutable engine state, grouped so a multiarray array switch
// is whole-struct assignment in each direction — never a field checklist.
struct engine_array_state
{
  struct gpu_scheduler sched;
  struct lod_state lod;
  struct compress_agg_array agg;
};

// --- Engine / Context ---

// Per-array identity — lightweight, scales with number of arrays.
// Holds immutable configuration plus the append cursor.
// Mutable batch/flush/shard state lives in the engine's sub-structs and is
// swapped via bind/unbind when switching arrays (multiarray only).
struct stream_context
{
  struct tile_stream_configuration config;
  struct shard_sink* sink;
  struct tile_stream_layout layout;
  struct level_geometry levels;
  struct dim_info dims;
  uint64_t cursor_elements;
  uint64_t total_element_limit; // configured stream length; 0 = unbounded
  size_t shard_alignment;       // from sink; 0 = no alignment

  // A dispatch that fails partway leaves the epochs it transferred uncounted,
  // and nothing can un-enqueue them, so this array's cursor no longer says
  // where data belongs and it stops taking any. Per array: the other arrays of
  // a multiarray stream are unaffected and still close out normally.
  int append_failed;
};

// Shared GPU resources — constant memory, allocated once.
// Contains all GPU allocations (pools, staging, codec, CUDA streams) and
// the scheduler's mutable pipeline state.
struct stream_engine
{
  struct gpu_streams streams;
  struct gpu_ordering ord;
  struct gpu_scheduler sched;
  struct pool_state pools;
  size_t pool_bytes;
  struct staging_state stage;
  struct compress_agg_stage compress_agg;
  struct d2h_deliver_stage d2h_deliver;
  struct lod_shared_state lod_shared; // engine-owned shared LOD resources
  struct lod_state lod;               // per-array; overwritten on array switch
  struct threadpool* copy_pool;       // staging-copy helpers (append body)
  struct gpu_delivery delivery;       // drain worker (pipelined schedule)
  // The context the streams and device memory belong to. The drain worker
  // clears its own copy when it fails to start, and the engine runs on.
  CUcontext cuda;
  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};

// --- Engine init / teardown (shared by single-array and multiarray) ---

// Sizing for the engine's shared resources. Single-array fills this from its
// one layout; multiarray accumulates the max across arrays — shared buffers
// are sized to the maxima while each array still runs at its own geometry.
struct engine_limits
{
  size_t buffer_capacity; // page-aligned staging slot size
  size_t pool_bytes;      // one chunk-pool buffer
  size_t chunk_bytes;     // codec chunk stride in bytes
  uint64_t codec_batch;   // K * total_chunks
  uint32_t epochs_per_batch;
  int max_nlod;
  uint64_t max_total_batch_chunks;
  uint64_t max_total_batch_covering;
  size_t max_device_data_bytes;
  size_t max_host_data_bytes;
  size_t lod_linear_bytes;
  size_t lod_morton_bytes;
  int any_multiscale;
  int max_threads; // max over arrays; 0 = platform default
};

// Fold one array's requirements into *lim (max per field). Call once per
// array on a zeroed struct before stream_engine_init.
int
engine_limits_accumulate(struct engine_limits* lim,
                         const struct computed_stream_layouts* cl,
                         const struct tile_stream_configuration* config);

// scatter_is_copy selects the ingest metric label only. On failure the
// engine is left partially initialized; stream_engine_destroy handles it.
int
stream_engine_init(struct stream_engine* e,
                   const struct engine_limits* lim,
                   enum compression_codec codec_id,
                   int scatter_is_copy);

// Free shared engine resources. Per-array state (engine_array_state) is
// destroyed separately by its owner. Caller must have synchronized all
// engine streams first.
void
stream_engine_destroy(struct stream_engine* e);

// Initialize one array's engine state. ctx->config/sink/shard_alignment must
// be set; fills the remaining ctx fields and takes ownership of cl->plan.
// Pass the engine delivery worker for a single-array stream; pass NULL for
// multi-array, whose immediate drains host-order per-array state changes.
int
engine_array_state_init(struct engine_array_state* st,
                        struct stream_context* ctx,
                        struct computed_stream_layouts* cl,
                        struct gpu_delivery* delivery);

void
engine_array_state_destroy(struct engine_array_state* st);

// Make *st the engine's active array (whole-struct handoff plus LUT-cache
// invalidation). On failure, the engine's per-array views are unchanged.
int
stream_engine_bind_array(struct stream_engine* e,
                         const struct engine_array_state* st,
                         const struct stream_context* ctx);

// --- Engine operations ---

// View of the given epoch's chunk region in the current pool — within the
// produce generation acquired at the last swap.
struct gpu_pool_view
stream_engine_pool_epoch(struct stream_engine* e,
                         struct stream_context* ctx,
                         uint32_t epoch_in_batch);

// Build the initial set of metric labels.  Shared by single-array and
// multiarray GPU constructors; enable_multiscale only affects the label of
// the scatter/copy stage ("Copy" when multiscale, "Scatter" otherwise).
struct stream_metrics
stream_engine_init_metrics(int enable_multiscale);

// Point the ordering host-poll edges at this engine's edge_stall metrics.
// Call after assigning engine.metrics; the bench prints the rows.
void
stream_engine_attach_edge_stalls(struct stream_engine* e);

// Append data to the stream. Handles staging, dispatch, epoch boundaries,
// batch flush, and backpressure. Used by both single-array and multiarray.
struct writer_result
stream_append_body(struct stream_engine* e,
                   struct stream_context* ctx,
                   struct slice input);

// Hand whatever staging holds to the device and count into the batch every
// epoch that completes. Callers that read the append cursor as the position of
// delivered data must call this first. A bind swaps the per-array engine state
// the dispatch reads, so the owner must still be the bound array.
struct writer_result
stream_dispatch_staged(struct stream_engine* e);

// Flush the stream: partial epoch, accumulated batch, partial append
// accumulators, finalize shards.
//
// Queues the writes and returns; it does not wait for them to retire. Pair it
// with stream_close_body before the buffers those writes point into are freed.
struct writer_result
stream_flush_body(struct stream_engine* e, struct stream_context* ctx);

// Waits for queued writes to retire, then publishes the append extent and lets
// the sink write its own metadata.
struct writer_result
stream_close_body(struct compress_agg_array* ar, struct stream_context* ctx);
