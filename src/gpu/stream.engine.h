#pragma once

#include "gpu/aggregate.h"
#include "gpu/compress.h"
#include "gpu/flush.handoff.h"
#include "gpu/ordering.h"
#include "gpu/pool.h"
#include "gpu/reduce_csr_gpu.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "zarr/shard_delivery.h"
#include <stddef.h>

// --- Sub-struct definitions (shared between engine and internal headers) ---

struct pool_state
{
  struct gpu_pool p;  // buf generations: ready=POOL_FILLED,
                      //                  consumed=POOL_CONSUMED (#140)
  CUdeviceptr buf[2]; // payloads; non-init code goes through p
  int current;        // 0 or 1
};

// Ordering events (h2d-done, scatter-done) live in gpu_ordering, instanced
// by staging slot; only timing-interval starts stay here. h_in/d_in are the
// pool payloads — non-init code reaches them through d_pool/h_pool only.
struct staging_slot
{
  void* h_in;              // pinned host WC, size = buffer_capacity_bytes
  CUdeviceptr d_in;        // device, size = buffer_capacity_bytes
  CUevent t_h2d_start;     // recorded before H2D memcpy (timing)
  CUevent t_scatter_start; // recorded before scatter kernel (timing)
  size_t dispatched_bytes; // bytes transferred in last dispatch
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
};

// Per flush-slot: mutable batch state (masks + epoch count).
// batch_active_masks is heap-allocated to epochs_per_batch entries.
struct flush_slot_gpu
{
  uint32_t active_levels_mask;  // union of per-epoch active masks
  uint32_t* batch_active_masks; // [K] per-epoch active level masks
  int batch_epoch_count;        // number of epochs accumulated in this batch
};

// Per-frame-counter timing events (double-buffered).
struct lod_timing
{
  CUevent t_start;
  CUevent t_scatter_end;
  CUevent t_reduce_end;
  CUevent t_append_end;
  CUevent t_end;
};

// Engine-owned LOD resources shared across all arrays in a multiarray stream
// (sized to the max requirement).  For a single-array stream these are still
// owned by the engine and sized for that one array.
struct lod_shared_state
{
  CUdeviceptr d_linear;        // linear epoch buffer (device)
  CUdeviceptr d_morton;        // morton-ordered LOD output (all levels packed)
  struct lod_timing timing[2]; // double-buffered pipeline timing
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
  struct tile_stream_layout_gpu layout_gpu[LOD_MAX_LEVELS];

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

// CUDA stream handles (all immutable after create)
struct gpu_streams
{
  CUstream h2d, compute, compress, d2h;
};

// Batch accumulation. Pool readiness is GPU_EDGE_POOL_FILLED: all K scatter
// ops run on the compute stream in order, so one record after the K-th
// scatter subsumes all per-epoch ready signals.
struct batch_state
{
  uint32_t epochs_per_batch; // K (immutable after create)
  uint32_t accumulated;      // mutable: 0..K-1
};

// --- Stage types ---

struct compress_agg_input
{
  int fc;
  uint32_t n_epochs;
  uint32_t active_levels_mask;
  const uint32_t* batch_active_masks; // borrowed from flush_slot_gpu [K]
  struct gpu_pool* pool; // chunk pool; the kick acquires slot fc's batch
  int lod_active; // wait GPU_EDGE_LOD_DONE (multiscale with bound edge only)
  uint32_t epochs_per_batch;
};

// Per-shard layout tables shared across arrays, sized to the max
// total_shards. d_base_offsets / d_tps_group / d_offsets_base depend on
// per-batch active counts and are re-uploaded every kick; d_shard_capacity
// is constant per array and uploaded at bind.
struct shard_tables
{
  size_t* h_base_offsets;   // base byte offset in d_aggregated
  uint64_t* h_tps_group;    // chunks-per-shard within a batch
  uint64_t* h_offsets_base; // base index in d_offsets / d_permuted_sizes

  size_t* d_base_offsets;
  size_t* d_shard_capacity;
  uint64_t* d_tps_group;
  uint64_t* d_offsets_base;
};

// Per-array slice of compress+aggregate state. Multiarray swaps this into
// the engine wholesale on array switch; any new per-array field added here
// rides along with that single assignment.
struct compress_agg_array
{
  // Immutable per array; read by the host LUT builder and at delivery.
  struct aggregate_layout per_lod_agg_layouts[LOD_MAX_LEVELS];
  uint8_t nlod;

  // Stays per-LOD because deliver_to_shards_batch and finalize_shards
  // iterate it; the per-shard tail/carry bytes the GPU kernels consume
  // live below instead.
  struct shard_state shard[LOD_MAX_LEVELS];

  uint64_t total_shards;    // sum_lv num_shards[lv]
  size_t* h_shard_capacity; // per shard; uploaded to shards.d_shard_capacity

  // Persistent across-batch tail state, unified across all shards. Sized to
  // total_shards / total_shards * page_size. d_* uploaded by host after each
  // batch's delivery; replaces the per-LOD d_tail_bytes / d_tail_carry pair.
  size_t page_size; // uniform: sink-required alignment, or 0 for legacy
  size_t* h_tail_bytes;
  size_t* d_tail_bytes;
  CUdeviceptr d_tail_carry;
  size_t tail_carry_bytes; // == total_shards * page_size

  // Tail-generation gate (page-aligned lazy pipeline). Kick #k's tail reads
  // consume the d_tail_bytes/d_tail_carry upload made by kick #k-1's
  // delivery, which on the lazy path runs AFTER kick #k is enqueued;
  // nothing else orders that host upload against the queued kernels. Kick
  // #k waits the generation counter >= k, the delivery publishes after each
  // upload (flush.d2h_deliver.c), and drains are oldest-first, so the
  // published count tracks delivered kicks. Counter state lives in
  // gpu_ordering (GPU_EDGE_TAIL_PUBLISHED).

  // Per-LOD slice info, needed by delivery to view the unified slot buffers.
  uint32_t shards_begin[LOD_MAX_LEVELS]; // first global shard index for LOD lv
  uint32_t n_shards[LOD_MAX_LEVELS];     // num_shards_lv
};

struct compress_agg_stage
{
  struct gpu_ordering* ord; // borrowed
  struct codec codec;
  CUdeviceptr d_compressed[2];
  CUevent t_compress_start[2]; // timing
  CUevent t_compress_end[2];   // timing

  uint32_t* pool_epochs_scratch; // [LOD_MAX_LEVELS * K] scratch for mask scans

  // Unified aggregate slot (per-fc), sized to max_batch_layout maxima. Holds:
  //   d_aggregated:     max_batch_layout.total_data_bytes
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
  struct gpu_pool tail; // d_tail_bytes/d_tail_carry generations (#142):
                        // ready=TAIL_PUBLISHED (GEN_COUNTER); consumed is
                        // the deliver-oldest-first host rule. Payload is
                        // the bound compress_agg_array (ar below).
  size_t max_total_batch_chunks;
  size_t max_total_batch_covering;
  size_t max_total_data_bytes;

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
  uint64_t lut_steady_count;
  uint64_t lut_recompute_count;

  // Per-shard tables shared across arrays (sized to maxima).
  struct shard_tables shards;

  // Per-array state; swapped on multiarray bind.
  struct compress_agg_array ar;
};

struct d2h_deliver_stage
{
  struct gpu_ordering* ord; // borrowed
  CUevent t_d2h_start[2];   // timing

  // Drain-time copies must not share the d2h stream: by drain time it can
  // already hold the next kick's GPU_EDGE_AGG_DONE wait, which the tail
  // gate keeps parked until THIS drain publishes — sharing would deadlock.
  // The drain's host poll of GPU_EDGE_CHUNK_INDEX_READY already proves the
  // copy source is stable, so no device-side ordering is needed here.
  CUstream drain_stream;

  size_t shard_alignment;         // from sink; 0 = no alignment
  struct stream_metrics* metrics; // borrowed, for stall-time accumulation
};

// Lazy-delivery pipeline: each fc slot can independently hold a kicked-but-
// not-yet-delivered batch. Delivery of fc=A happens at the start of the
// drain_kick_and_swap round that is about to reuse fc=A (two rounds after
// that batch was kicked, in N=2 alternation). Both fcs may hold a pending
// batch simultaneously; drain order at final flush follows pending_seq.
struct flush_pipeline
{
  struct flush_slot_gpu slot[2];
  int current;                             // fc currently being filled
  int pending[2];                          // per-fc: batch awaiting delivery
  uint64_t pending_seq[2];                 // monotonic kick seq per pending
  uint64_t next_seq;                       // next seq to assign on kick
  struct flush_handoff pending_handoff[2]; // per-fc
};

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// All per-array mutable engine state, grouped so a multiarray array switch
// is whole-struct assignment in each direction — never a field checklist.
struct engine_array_state
{
  struct batch_state batch;
  int pool_current;
  struct flush_pipeline flush;
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
  struct tile_stream_layout_gpu layout_gpu;
  struct level_geometry levels;
  struct dim_info dims;
  uint64_t cursor_elements;
  uint64_t total_element_limit; // configured stream length; 0 = unbounded
  size_t shard_alignment;       // from sink; 0 = no alignment
};

// Shared GPU resources — constant memory, allocated once.
// Contains all GPU allocations (pools, staging, codec, CUDA streams) and
// mutable pipeline state (batch accumulation, flush slots, shard tracking).
struct stream_engine
{
  int sync_flush; // 1 = synchronous batch flush (multiarray); 0 = pipelined
  struct gpu_streams streams;
  struct gpu_ordering ord;
  struct pool_state pools;
  size_t pool_bytes;
  struct staging_state stage;
  struct batch_state batch;
  struct flush_pipeline flush;
  struct compress_agg_stage compress_agg;
  struct d2h_deliver_stage d2h_deliver;
  struct lod_shared_state lod_shared; // engine-owned shared LOD resources
  struct lod_state lod;               // per-array; overwritten on array switch
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
  size_t max_total_data_bytes;
  uint64_t max_total_shards;
  size_t lod_linear_bytes;
  size_t lod_morton_bytes;
  int any_multiscale;
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

// Initialize one array's engine state. ctx->config/sink/shard_alignment
// must be set; fills the remaining ctx fields and takes ownership of
// cl->plan. gate_ord arms the tail-generation gate (pipelined single-array
// path); pass NULL on the multiarray sync-flush path, whose immediate
// drains host-order the tail uploads instead.
int
engine_array_state_init(struct engine_array_state* st,
                        struct stream_context* ctx,
                        struct computed_stream_layouts* cl,
                        struct gpu_ordering* gate_ord,
                        CUstream gate_stream);

void
engine_array_state_destroy(struct engine_array_state* st);

// Make *st the engine's active array (whole-struct handoff + per-array
// shard-capacity upload + LUT-cache invalidation).
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

// Flush the stream: partial epoch, accumulated batch, partial append
// accumulators, finalize shards, update metadata.
struct writer_result
stream_flush_body(struct stream_engine* e, struct stream_context* ctx);
