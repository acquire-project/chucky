#pragma once

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

  struct aggregate_slot
  {
    size_t* d_permuted_sizes; // device: (C+1) size_t, zeroed each epoch
    size_t* d_offsets;        // device: (C+1) size_t
    void* d_aggregated;       // device: comp_pool_bytes
    void* h_aggregated;       // host pinned: comp_pool_bytes
    size_t* h_offsets;        // host pinned: (C+1) size_t
    size_t* h_permuted_sizes; // host pinned: C size_t (real compressed sizes)
    void* d_temp;             // CUB scratch
    size_t temp_bytes;
    struct io_event io_done; // tracks IO completion from this slot's data
  };

  // Upload pre-computed layout arrays to GPU. Must be called after
  // aggregate_layout_compute. Returns 0 on success.
  int aggregate_layout_upload(struct aggregate_layout* layout);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t batch_covering_count,
                                size_t comp_pool_bytes);

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
  //   total_batch_chunks   : M_total = sum_lv n_active[lv] * chunks_per_epoch[lv]
  //   total_batch_covering : C_total = sum_lv n_active[lv] * covering_count[lv]
  //   nlod                 : number of LODs
  //   max_comp_chunk_bytes : uniform pool stride
  //   slot                 : unified aggregate_slot (d_aggregated sized to
  //                          max_total_data_bytes; d_offsets/d_permuted_sizes
  //                          sized to max_total_batch_covering + LOD_MAX_LEVELS).
  //   d_shard_base_offsets : [total_shards] base byte offset in d_aggregated
  //                          for each shard. Replaces uniform s*shard_capacity.
  //   d_shard_capacity     : [total_shards] each shard's capacity in bytes
  //                          (unused by gather_batch_k but kept for symmetry/asserts).
  //   d_shard_tps_group    : [total_shards] chunks-per-shard within this batch.
  //                          Replaces uniform tps_group.
  //   d_shard_offsets_base : [total_shards] base index in d_offsets for each
  //                          shard's run. Replaces uniform s*tps_group.
  //   d_tail_bytes         : [total_shards] persistent per-shard tail-bytes
  //                          carried from prior batch. Uploaded by host
  //                          post-delivery.
  //   d_tail_carry         : [total_shards * page_size] persistent tail bytes
  //                          carry buffer. Uploaded post-delivery.
  //   page_size            : uniform sink page size (one for all shards today).
  //                          0 = no carry-over path.
  //   total_shards         : sum_lv num_shards[lv]
  //   stream               : compress stream
  int aggregate_batch_unified_async(
    const void* d_compressed,
    size_t* d_comp_sizes,
    const uint32_t* d_batch_gather,
    const uint32_t* d_batch_perm,
    uint64_t total_batch_chunks,
    uint64_t total_batch_covering,
    uint8_t nlod,
    size_t max_comp_chunk_bytes,
    struct aggregate_slot* slot,
    const size_t* d_shard_base_offsets,
    const size_t* d_shard_capacity,
    const uint64_t* d_shard_tps_group,
    const uint64_t* d_shard_offsets_base,
    const size_t* d_tail_bytes,
    CUdeviceptr d_tail_carry,
    size_t page_size,
    uint64_t total_shards,
    CUstream stream);

#ifdef __cplusplus
}
#endif
