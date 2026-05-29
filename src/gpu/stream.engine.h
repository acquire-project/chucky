#pragma once

#include "gpu/aggregate.h"
#include "gpu/compress.h"
#include "gpu/flush.handoff.h"
#include "gpu/reduce_csr_gpu.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "zarr/shard_delivery.h"
#include <stddef.h>

// --- Sub-struct definitions (shared between engine and internal headers) ---

struct pool_state
{
  CUdeviceptr buf[2];
  CUevent ready[2];
  int current; // 0 or 1
};

struct staging_slot
{
  void* h_in;              // pinned host WC, size = buffer_capacity_bytes
  CUdeviceptr d_in;        // device, size = buffer_capacity_bytes
  CUevent t_h2d_end;       // recorded after H2D memcpy completes
  CUevent t_h2d_start;     // recorded before H2D memcpy
  CUevent t_scatter_start; // recorded before scatter kernel
  CUevent t_scatter_end;   // recorded after scatter kernel
  size_t dispatched_bytes; // bytes transferred in last dispatch
};

struct staging_state
{
  struct staging_slot slot[2];
  int current;          // 0 or 1: which buffer the host is filling
  size_t bytes_written; // bytes written to current slot's h_in so far
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

// Batch accumulation: config + mutable counter + single pool-ready event.
// All K scatter ops run on the compute stream in order, so a single event
// recorded after the K-th scatter subsumes all per-epoch ready signals.
struct batch_state
{
  uint32_t epochs_per_batch; // K (immutable after create)
  uint32_t accumulated;      // mutable: 0..K-1
  CUevent pool_ready;        // recorded on compute after last accumulated epoch
};

// --- Stage types ---

struct compress_agg_input
{
  int fc;
  int output_idx; // output reservoir slot; cycles on close, not per batch
  uint32_t n_epochs;
  uint32_t active_levels_mask;
  const uint32_t* batch_active_masks; // borrowed from flush_slot_gpu [K]
  CUdeviceptr pool_buf;
  CUevent pool_ready;
  CUevent lod_done;
  // initialized signaled; guards d_aggregated overwrite. At cap>1,
  // fit_decision_k can route this kick to either slot, so we wait on BOTH
  // slots' d2h-done.
  CUevent prev_d2h_done[2];
  uint32_t epochs_per_batch;
};

// Per-shard layout tables, sized to total_shards = sum(num_shards_lv) across
// all LODs. These collapse what was N per-LOD shard-bias/leading-tail launches
// into a single launch over the flat shard list. d_base_offsets / d_tps_group
// / d_offsets_base are rebuilt and re-uploaded every kick (depends on
// per_lod_n_active); d_shard_capacity is constant per array and uploaded once
// at init (or on multiarray bind).
struct shard_tables
{
  uint64_t total_shards;
  // Per-shard parameters used by add_shard_bias_unified_k and
  // copy_leading_tail_unified_k. Host shadows are owned; device buffers are
  // allocated at init sized to the max across arrays.
  size_t* h_base_offsets;   // base byte offset in d_aggregated
  size_t* h_shard_capacity; // per shard
  uint64_t* h_tps_group;    // chunks-per-shard within a batch
  uint64_t* h_offsets_base; // base index in d_offsets / d_permuted_sizes

  size_t* d_base_offsets;
  size_t* d_shard_capacity;
  uint64_t* d_tps_group;
  uint64_t* d_offsets_base;

  // Persistent across-batch tail state, unified across all shards. Sized to
  // total_shards / total_shards * page_size. d_* uploaded by host after each
  // batch's delivery; replaces the per-LOD d_tail_bytes / d_tail_carry pair.
  size_t page_size; // uniform: sink-required alignment, or 0 for legacy
  size_t* h_tail_bytes;
  size_t* d_tail_bytes;
  CUdeviceptr d_tail_carry;
  size_t tail_carry_bytes; // == total_shards * page_size

  // Per-LOD slice info, needed by delivery to view the unified slot buffers.
  uint32_t shards_begin[LOD_MAX_LEVELS]; // first global shard index for LOD lv
  uint32_t n_shards[LOD_MAX_LEVELS];     // num_shards_lv
};

struct compress_agg_stage
{
  struct codec codec;
  CUdeviceptr d_compressed[2];
  CUevent t_compress_start[2];
  CUevent t_compress_end[2];
  CUevent t_aggregate_end[2];

  uint32_t* pool_epochs_scratch; // [LOD_MAX_LEVELS * K] scratch for mask scans

  // Output reservoir slots (Prep-0: indexed by output_idx, today output_idx
  // == fc; later prep decouples). Sized to max_batch_layout maxima. Holds:
  //   d_aggregated:     max_batch_layout.total_data_bytes
  //   d_offsets/sizes:  total_batch_covering + LOD_MAX_LEVELS (+ 1 for offsets)
  //   plus pinned host shadows of matching size.
  // NB: the type stays `aggregate_slot` — it already contains only output-side
  // state. The field rename signals the role split conceptually.
  struct aggregate_slot output[2];
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

  // Per-LOD aggregate layouts (immutable per array; switched on multiarray
  // bind). One per LOD; carries shard_capacity, num_shards, cps_inner,
  // page_size, etc. Read by host LUT builder and at delivery time.
  struct aggregate_layout per_lod_agg_layouts[LOD_MAX_LEVELS];
  uint8_t nlod;

  // Cached "max" batch layout — segment offsets / total sizes assuming the
  // worst-case active_count_max per LOD. Buffers are sized from this.
  struct batch_aggregate_layout max_batch_layout;

  // Per-shard tables (replaces per-LOD d_tail_*, d_batch_gather/perm).
  struct shard_tables shards;

  // Routing decision for the current batch, written by fit_decision_k.
  struct d_routing* d_routing;
  size_t* d_temp_offsets;
  size_t* d_temp_perm_sizes;
  volatile struct d_routing* h_routing;
  struct agg_routing_cb_args cb_args_ring[4];
  uint64_t cb_args_seq;

  // Per-LOD shard_state (one shard_state per LOD, mirrors CPU). Carries
  // per-shard writers, tail/footer pools, generation-boundary bookkeeping.
  // The shard_state itself stays per-LOD because deliver_to_shards_batch and
  // finalize_shards iterate it; the per-shard tail/carry bytes consumed by
  // the GPU kernels live in shard_tables instead.
  struct shard_state shard[LOD_MAX_LEVELS];
};

struct d2h_deliver_stage
{
  CUevent t_d2h_start[2];
  CUevent h_chunk_index_ready[2]; // h_offsets + h_permuted_sizes on host
  CUevent ready[2];               // full D2H done; gates slot reuse

  size_t shard_alignment;         // from sink; 0 = no alignment
  struct stream_metrics* metrics; // borrowed, for stall-time accumulation
};

enum output_slot_state
{
  OUTPUT_SLOT_EMPTY = 0,     // may be selected for aggregate writes
  OUTPUT_SLOT_OPEN = 1,      // accumulating one or more stacked batches
  OUTPUT_SLOT_CLOSED = 2,    // D2H was kicked; host delivery not complete
  OUTPUT_SLOT_DELIVERING = 3 // sink delivery owns/reads host buffers
};

// Output-slot lifecycle is host-owned. Aggregate may write only to EMPTY or
// the current OPEN slot; CLOSED/DELIVERING slots are immutable until drained
// and reset to EMPTY. The CUDA routing kernel can compute placement, but it
// must never be allowed to swap into a non-EMPTY alternate slot.
struct flush_pipeline
{
  struct flush_slot_gpu slot[2];
  int output_current; // output reservoir index; flips on slot close
  enum output_slot_state output_state[2];
  uint64_t pending_seq[2];
  uint64_t next_seq;
  struct flush_handoff pending_handoff[2];
};

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

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

// --- Engine operations ---

// Pointer to the given epoch's chunk region in the current pool.
void*
stream_engine_pool_epoch(struct stream_engine* e,
                         struct stream_context* ctx,
                         uint32_t epoch_in_batch);

// Build the initial set of metric labels.  Shared by single-array and
// multiarray GPU constructors; enable_multiscale only affects the label of
// the scatter/copy stage ("Copy" when multiscale, "Scatter" otherwise).
struct stream_metrics
stream_engine_init_metrics(int enable_multiscale);

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
