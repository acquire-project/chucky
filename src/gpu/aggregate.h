#pragma once

#include "io_queue.h"
#include "transpose.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Compute the aggregated-buffer size including page-alignment padding slack.
  // covering_count and cps_inner are per-epoch values.
  static inline size_t agg_pool_bytes(uint64_t chunk_count,
                                      size_t max_comp_chunk_bytes,
                                      uint64_t covering_count,
                                      uint64_t cps_inner,
                                      size_t page_size)
  {
    size_t bytes = chunk_count * max_comp_chunk_bytes;
    if (page_size > 0 && cps_inner > 0) {
      uint64_t num_shards = covering_count / cps_inner;
      bytes += num_shards * page_size + page_size;
    }
    return bytes;
  }

  struct aggregate_layout
  {
    uint8_t lifted_rank; // 2 * (rank - 1)
    uint64_t lifted_shape[MAX_RANK];
    int64_t lifted_strides[MAX_RANK];
    uint64_t* d_lifted_shape;  // device copy
    int64_t* d_lifted_strides; // device copy
    uint64_t chunks_per_epoch; // M: actual chunk count
    uint64_t covering_count;   // C >= M: product of padded dims
    size_t max_comp_chunk_bytes;
    uint64_t cps_inner; // product of chunks_per_shard for inner dims
    size_t page_size;   // 0 = no padding
  };

  struct aggregate_slot
  {
    size_t* d_permuted_sizes; // device: (C+1) size_t, zeroed each epoch
    size_t* d_offsets;        // device: (C+1) size_t
    uint32_t* d_perm;         // device: M uint32_t
    void* d_aggregated;       // device: comp_pool_bytes
    void* h_aggregated;       // host pinned: comp_pool_bytes
    size_t* h_offsets;        // host pinned: (C+1) size_t
    size_t* h_permuted_sizes; // host pinned: C size_t (real, pre-padding)
    void* d_temp;             // CUB scratch
    size_t temp_bytes;
    CUevent ready;           // D2H completion
    struct io_event io_done; // tracks IO completion from this slot's data
  };

  // Compute host-side aggregate layout fields (pure CPU, no GPU allocation).
  //
  // rank:                full dimensionality (>= 1); dims 1..rank-1 are used.
  // chunk_count[d]:       number of chunks along dim d (d = 0..rank-1).
  // chunks_per_shard[d]:  chunks per shard along dim d (d = 1..rank-1, each >=
  // 1).
  //                       chunks_per_shard[0] is ignored.
  // chunks_per_epoch:     M = prod(chunk_count[d] for d > 0), the number of
  //                       chunks per epoch.
  // max_comp_chunk_bytes: upper bound on compressed size of one chunk.
  //
  // Returns 0 on success. On failure, *layout is zeroed and safe to destroy.
  int aggregate_layout_compute(struct aggregate_layout* layout,
                               uint8_t rank,
                               const uint64_t* chunk_count,
                               const uint64_t* chunks_per_shard,
                               uint64_t chunks_per_epoch,
                               size_t max_comp_chunk_bytes,
                               size_t page_size);

  // Upload pre-computed layout arrays to GPU. Must be called after
  // aggregate_layout_compute. Returns 0 on success.
  int aggregate_layout_upload(struct aggregate_layout* layout);

  // Compute + upload (convenience wrapper). Returns 0 on success.
  // On failure, *layout is zeroed and safe to destroy.
  int aggregate_layout_init(struct aggregate_layout* layout,
                            uint8_t rank,
                            const uint64_t* chunk_count,
                            const uint64_t* chunks_per_shard,
                            uint64_t chunks_per_epoch,
                            size_t max_comp_chunk_bytes,
                            size_t page_size);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  // Enqueue all 3 passes on stream. Returns 0 on success.
  int aggregate_by_shard_async(const struct aggregate_layout* layout,
                               void* d_compressed,
                               size_t* d_comp_sizes,
                               struct aggregate_slot* slot,
                               CUstream stream);

  // Initialize a slot sized for batch aggregation (K_l * M_l chunks).
  // batch_chunk_count: total chunks in the batch (K_l * M_l).
  // batch_covering_count: covering count for the batch (K_l * C_l).
  // comp_pool_bytes: batch_chunk_count * max_comp_chunk_bytes.
  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t batch_chunk_count,
                                uint64_t batch_covering_count,
                                size_t comp_pool_bytes);

  // Batch aggregate using precomputed LUTs.
  // d_batch_gather[i]: index into d_compressed (compressed buffer chunk index)
  // d_batch_perm[i]: index into shard-ordered output position
  // batch_chunk_count: total chunks in the batch (K_l * M_l)
  // batch_covering_count: covering count for shard ordering (K_l * C_l)
  // Returns 0 on success.
  int aggregate_batch_by_shard_async(void* d_compressed,
                                     size_t* d_comp_sizes,
                                     const uint32_t* d_batch_gather,
                                     const uint32_t* d_batch_perm,
                                     uint64_t batch_chunk_count,
                                     uint64_t batch_covering_count,
                                     size_t max_comp_chunk_bytes,
                                     const struct aggregate_layout* layout,
                                     struct aggregate_slot* slot,
                                     CUstream stream);

#ifdef __cplusplus
}
#endif
