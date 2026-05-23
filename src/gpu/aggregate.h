#pragma once

#include "defs.limits.h"
#include "stream/types.aggregate.h"
#include "writer.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Query CUB scratch workspace size for ExclusiveSum over count elements.
  // Writes result to *out_bytes. Returns 0 on success.
  int aggregate_cub_temp_bytes(uint64_t count, size_t* out_bytes);

  // One batch's slice inside a slot. Phase 2 always has batches_per_slot=1
  // and base offsets stay at 0; Phase 3 will stack multiple entries with
  // monotonically advancing bases.
  //
  // Slot-relative byte offset for LOD lv =
  //   data_base + per_lod_lods[lv].data_segment_offset
  // Slot-relative descriptor index for LOD lv =
  //   desc_base + per_lod_lods[lv].batch_covering_offset + lv
  struct batch_slice_entry
  {
    size_t data_base_offset;
    uint64_t desc_base_offset;
    uint8_t nlod;
    struct lod_segment per_lod_lods[LOD_MAX_LEVELS];
  };

  struct aggregate_slot;

  // Each callback invocation needs to know its own batch_idx; reading
  // slot->batches_per_slot would race against the main thread.
  struct cb_context
  {
    struct aggregate_slot* slot;
    uint32_t batch_idx;
  };

  // Device-resident counterpart of slot_cursor/slot_desc_cursor/
  // batches_per_slot. Source of truth once GPU routing is active.
  struct slot_runtime_state
  {
    size_t cursor;
    uint64_t desc_cursor;
    uint32_t batches_per_slot;
    uint32_t _pad;
  };

  // Per-batch routing decision. Downstream aggregate kernels read this
  // instead of taking slot pointers/offsets as launch args.
  struct d_routing
  {
    void* target_d_aggregated;
    size_t* target_d_offsets;
    size_t* target_d_permuted_sizes;
    size_t* target_d_shard_base_offsets_dense;
    size_t data_base_offset;
    uint64_t desc_base_offset;
    uint32_t batch_idx_in_slot;
    int32_t target_slot_idx;
    int32_t close_prior_slot_idx; // -1 if no close
    uint32_t _pad;
  };

  struct slot_dev_ptrs
  {
    void* d_aggregated;
    size_t* d_offsets;
    size_t* d_permuted_sizes;
    size_t* d_shard_base_offsets_dense;
    struct slot_runtime_state* d_runtime;
  };

  int fit_decision_launch(struct d_routing* d_routing,
                          struct slot_dev_ptrs slot_0,
                          struct slot_dev_ptrs slot_1,
                          const size_t* d_actual_bytes_ptr,
                          size_t slot_capacity,
                          uint32_t batches_per_slot_cap,
                          uint64_t batch_desc_advance,
                          int active_slot_idx,
                          CUstream stream);

  struct aggregate_slot
  {
    size_t* d_permuted_sizes; // device: slot_chunk_cap size_t
    size_t* d_offsets;        // device: slot_chunk_cap size_t
    void* d_aggregated;       // device: comp_pool_bytes
    void* h_aggregated;       // host pinned: comp_pool_bytes
    size_t* h_offsets;        // host pinned: slot_chunk_cap size_t
    size_t* h_permuted_sizes; // host pinned: slot_chunk_cap size_t
    void* d_temp;             // CUB scratch
    size_t temp_bytes;

    // Macro-aggregation state (Phase 2: scaffolding only).
    size_t slot_cursor;      // bytes written in d_aggregated
    size_t slot_desc_cursor; // entries written in d_offsets / d_permuted_sizes
    size_t slot_capacity_bytes; // size of d_aggregated; fit-check budget
    uint32_t batches_per_slot;
    uint32_t batches_per_slot_cap;
    struct batch_slice_entry* slot_batches; // [batches_per_slot_cap]
    struct cb_context* cb_contexts;         // [batches_per_slot_cap]
    struct slot_runtime_state* d_runtime;   // device, 1 instance

    // Pinned per-slot counter: the just-aggregated batch's actual byte
    // total (prefix-sum end). Filled by a small D2H on compress_stream
    // after aggregate; cuLaunchHostFunc reads it to update slot bookkeeping
    // before the next batch kicks. Volatile because reader (orchestrator
    // main thread) and writer (driver thread running the host function)
    // are different threads.
    volatile size_t* h_batch_actual_bytes;
    // Per-shard compressed-bytes sum for the just-aggregated batch.
    // d_shard_sum_bytes is filled by compute_shard_sum_bytes_k; the D2H
    // staging lands the values in h_shard_sum_bytes for the host callback
    // to compute dense shard_base_offsets for the NEXT batch in the slot.
    // Sized to max_total_shards (passed at slot init).
    size_t* d_shard_sum_bytes;
    volatile size_t* h_shard_sum_bytes;
    // Dense per-shard base offsets for this batch, computed by the host
    // callback as cumulative sum of (h_tail_bytes_view[i]+h_shard_sum_bytes[i])
    // plus slot_cursor. Pinned so a subsequent cuMemcpyHtoDAsync can stage
    // it to d_shard_base_offsets_dense; 2f will flip the bias/tail kernels
    // to read from there instead of the uniform-packed table.
    volatile size_t* h_shard_base_offsets_dense;
    size_t* d_shard_base_offsets_dense;
    // Borrowed pointer to host-side tail-bytes-prev array; orchestrator
    // sets before each kick. Read by the host callback alongside
    // h_shard_sum_bytes. Lifetime owned by orchestrator.
    const size_t* h_tail_bytes_view;
    // Future-state per-shard tail bytes for the NEXT batch. Populated by
    // the host callback as (h_tail_bytes_view[s] + h_shard_sum_bytes[s])
    // % page_size. For B=1 this is purely additive — orchestrator's
    // post-delivery H2D from h_tail_bytes still overwrites GPU d_tail_bytes;
    // for B>1 it becomes the source-of-truth handoff to the next in-slot
    // batch. Sized shard_sum_capacity.
    volatile size_t* h_tail_bytes_next;
    // Uniform sink page size; orchestrator sets before each kick so the
    // host callback can compute h_tail_bytes_next without depending on
    // shard_tables. 0 = no carry-over path (callback skips tail compute).
    size_t page_size;
    // Number of shards in the just-kicked batch (callback uses this to
    // bound its loop). Orchestrator sets before each kick.
    uint64_t total_shards_in;
    uint64_t shard_sum_capacity; // number of slots in the buffers above
    // Event recorded on compress_stream after the cuLaunchHostFunc
    // completes. Orchestrator waits on this before reading slot bookkeeping
    // updated by the callback.
    CUevent host_func_done;

    CUevent ready; // D2H completion
    // TODO(phase3): single io_event per slot means all B batches in a
    // macro-group share one IO fence. Fine for delivery serialization but
    // batches can't be retired independently. Acceptable for the planned
    // flush-on-close model.
    struct io_event io_done; // tracks IO completion from this slot's data
  };

  // Upload pre-computed layout arrays to GPU. Must be called after
  // aggregate_layout_compute. Returns 0 on success.
  int aggregate_layout_upload(struct aggregate_layout* layout);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  // Close the slot: reset slot_cursor/desc_cursor/batches_per_slot to 0 so
  // the next kick treats the slot as fresh. At B=1 the orchestrator calls
  // this after every kick; at B>1 it's called only on slot close (fit fail
  // or end-of-stream).
  void output_slot_close_reset(struct aggregate_slot* slot);

  // Append a batch entry into slot_batches[batches_per_slot] capturing this
  // batch's data/desc bases (current cursors) and per-LOD segments. Returns
  // 0 on success, non-zero on cap overflow.
  struct batch_aggregate_layout;
  int output_slot_append_batch_entry(
    struct aggregate_slot* slot,
    const struct batch_aggregate_layout* layout,
    uint8_t nlod);

  // Returns 1 if the slot has room for next_batch_bytes additional data
  // (slot_cursor + batches_per_slot < cap AND slot_cursor + next_batch_bytes
  // <= slot_capacity_bytes), else 0. Pure host function.
  int slot_can_fit(const struct aggregate_slot* slot, size_t next_batch_bytes);

  // Returns 1 if no shard would finalize during the next in-slot batch
  // (epoch_in_shard[lv] + n_active_next[lv] < cps_append[lv] for all
  // lv < nlod), else 0. Mid-slot finalization corrupts the GPU's in-slot
  // tail rollforward, so this must be true for B>1 stacking. Pure host
  // function; arrays are borrowed read-only.
  int no_shard_finalizes(uint8_t nlod,
                         const uint64_t* epoch_in_shard,
                         const uint32_t* n_active_next,
                         const uint64_t* cps_append);

  // slot_chunk_cap is the descriptor-array length (entries). Sized once at
  // init to W / min_compressed_chunk_bytes so Phase 3 macro-agg can pack
  // many batches' descriptors into the same slot without resize.
  // batches_per_slot_cap is the size of the slot_batches[] table. Phase 2
  // uses 1; Phase 3 will raise it.
  // max_total_shards sizes d_shard_sum_bytes / h_shard_sum_bytes. Pass 0
  // for codecs/configurations without per-shard aggregation; the kernel
  // launch and D2H are skipped in that case.
  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t batch_chunk_count,
                                uint64_t slot_chunk_cap,
                                size_t comp_pool_bytes,
                                uint32_t batches_per_slot_cap,
                                uint64_t max_total_shards);

  // d_tail_bytes: persistent per-LOD device array of size_t[num_shards]; read
  //   by add_shard_bias_k for this batch's leading-tail accounting. The host
  //   uploads the post-delivery values after every batch.
  // d_tail_carry: persistent per-LOD device buffer of num_shards * page_size
  //   bytes; holds the actual ragged tail bytes between batches. Same
  //   uploaded-by-host invariant as d_tail_bytes.
  int aggregate_batch_by_shard_async(const void* d_compressed,
                                     size_t* d_comp_sizes,
                                     const uint32_t* d_batch_gather,
                                     const uint32_t* d_batch_perm,
                                     uint64_t batch_chunk_count,
                                     uint64_t batch_covering_count,
                                     size_t max_comp_chunk_bytes,
                                     const struct aggregate_layout* layout,
                                     struct aggregate_slot* slot,
                                     size_t* d_tail_bytes,
                                     CUdeviceptr d_tail_carry,
                                     CUstream stream);

  // Single dispatch across all LODs. The kernels read per-shard parameters
  // from device-side tables built host-side at kick time. Per-LOD info is
  // encoded in the tables; the kernels themselves are level-agnostic.
  //
  //   d_compressed         : the chunk pool (sized M * max_comp_chunk_bytes,
  //                          shared across LODs — single pool stride)
  //   d_comp_sizes         : per-chunk actual compressed sizes [M]
  //   d_batch_gather       : unified gather LUT, length total_batch_chunks.
  //                          Maps unified output position -> chunk-pool index.
  //   d_batch_perm         : unified perm LUT, length total_batch_chunks.
  //                          Maps unified output position -> permuted index in
  //                          d_offsets/d_permuted_sizes (already LOD-shifted
  //                          by +lv inside aggregate_batch_luts_unified).
  //   total_batch_chunks   : M_total = sum_lv n_active[lv] *
  //   chunks_per_epoch[lv] total_batch_covering : C_total = sum_lv n_active[lv]
  //   * covering_count[lv] nlod                 : number of LODs
  //   max_comp_chunk_bytes : uniform pool stride
  //   slot                 : unified aggregate_slot (d_aggregated sized to
  //                          max_total_data_bytes; d_offsets/d_permuted_sizes
  //                          sized to max_total_batch_covering +
  //                          LOD_MAX_LEVELS).
  //   d_shard_base_offsets : [total_shards] base byte offset in d_aggregated
  //                          for each shard. Replaces uniform s*shard_capacity.
  //   d_shard_capacity     : [total_shards] each shard's capacity in bytes
  //                          (unused by gather_batch_k but kept for
  //                          symmetry/asserts).
  //   d_shard_tps_group    : [total_shards] chunks-per-shard within this batch.
  //                          Replaces uniform tps_group.
  //   d_shard_offsets_base : [total_shards] base index in d_offsets for each
  //                          shard's run. Replaces uniform s*tps_group.
  //   d_tail_bytes         : [total_shards] persistent per-shard tail-bytes.
  //                          Read as leading-tail for this batch's bias and
  //                          updated in place by the post-gather rollforward
  //                          kernel with this batch's trailing tail. The host
  //                          H2D post-delivery still resyncs for B=1 (correct
  //                          across mid-batch shard finalization).
  //   d_tail_carry         : [total_shards * page_size] persistent tail bytes
  //                          carry buffer. Same lifecycle as d_tail_bytes.
  //   page_size            : uniform sink page size (one for all shards today).
  //                          0 = no carry-over path.
  //   total_shards         : sum_lv num_shards[lv]
  //   stream               : compress stream
  // slot_data_base / slot_desc_base bias kernels' writes into the slot's
  // d_aggregated / d_offsets / d_permuted_sizes so multiple batches in
  // Phase 3 can stack without aliasing. Phase 2 passes 0 for both.
  //
  // TODO(phase3): when batches_per_slot > 1, compress_agg_kick will need
  // to call this once per batch with advancing slot_*_base, OR this fn
  // will need to take a batch index and iterate. The single-dispatch
  // shape today aggregates one batch per call.
  int aggregate_batch_unified_async(const void* d_compressed,
                                    size_t* d_comp_sizes,
                                    const uint32_t* d_batch_gather,
                                    const uint32_t* d_batch_perm,
                                    uint64_t total_batch_chunks,
                                    uint64_t total_batch_covering,
                                    uint8_t nlod,
                                    size_t max_comp_chunk_bytes,
                                    struct aggregate_slot* slot,
                                    size_t slot_data_base,
                                    uint64_t slot_desc_base,
                                    const size_t* d_shard_base_offsets,
                                    const size_t* d_shard_capacity,
                                    const uint64_t* d_shard_tps_group,
                                    const uint64_t* d_shard_offsets_base,
                                    size_t* d_tail_bytes,
                                    CUdeviceptr d_tail_carry,
                                    size_t page_size,
                                    uint64_t total_shards,
                                    CUstream stream);

#ifdef __cplusplus
}
#endif
