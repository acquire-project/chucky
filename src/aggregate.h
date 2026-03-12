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

  struct aggregate_layout
  {
    uint8_t lifted_rank;                // 2 * (rank - 1)
    uint64_t lifted_shape[MAX_RANK];
    int64_t lifted_strides[MAX_RANK];
    uint64_t* d_lifted_shape;           // device copy
    int64_t* d_lifted_strides;          // device copy
    uint64_t tiles_per_epoch;            // M: actual tile count
    uint64_t covering_count;            // C >= M: product of padded dims
    size_t max_chunk_bytes;
    uint64_t tps_inner;                 // product of tiles_per_shard for inner dims
    size_t page_size;                   // 0 = no padding
  };

  struct aggregate_slot
  {
    size_t* d_permuted_sizes;   // device: (C+1) size_t, zeroed each epoch
    size_t* d_offsets;          // device: (C+1) size_t
    uint32_t* d_perm;           // device: M uint32_t
    void* d_aggregated;         // device: comp_pool_bytes
    void* h_aggregated;         // host pinned: comp_pool_bytes
    size_t* h_offsets;          // host pinned: (C+1) size_t
    void* d_temp;               // CUB scratch
    size_t temp_bytes;
    CUevent ready;              // D2H completion
    struct io_event io_done;    // tracks IO completion from this slot's data
  };

  // Build the shard-permutation layout for reordering compressed tiles.
  //
  // rank:                full dimensionality (>= 2); dims 1..rank-1 are used.
  // tile_count[d]:       number of tiles along dim d (d = 0..rank-1).
  // tiles_per_shard[d]:  tiles per shard along dim d (d = 1..rank-1, each >= 1).
  //                      tiles_per_shard[0] is ignored.
  // tiles_per_epoch:     M = prod(tile_count[d] for d > 0), the number of tiles
  //                      per epoch.
  // max_chunk_bytes:     upper bound on compressed size of one tile.
  //
  // Returns 0 on success. On failure, *layout is zeroed and safe to destroy.
  int aggregate_layout_init(struct aggregate_layout* layout,
                            uint8_t rank,
                            const uint64_t* tile_count,
                            const uint64_t* tiles_per_shard,
                            uint64_t tiles_per_epoch,
                            size_t max_chunk_bytes,
                            size_t page_size);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  // Enqueue all 3 passes on stream. Returns 0 on success.
  int aggregate_by_shard_async(const struct aggregate_layout* layout,
                               void* d_compressed,
                               size_t* d_comp_sizes,
                               struct aggregate_slot* slot,
                               CUstream stream);

  // Initialize a slot sized for batch aggregation (K_l * M_l tiles).
  // batch_tile_count: total tiles in the batch (K_l * M_l).
  // batch_covering_count: covering count for the batch (K_l * C_l).
  // comp_pool_bytes: batch_tile_count * max_chunk_bytes.
  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t batch_tile_count,
                                uint64_t batch_covering_count,
                                size_t comp_pool_bytes);

  // Batch aggregate using precomputed LUTs.
  // d_batch_gather[i]: index into d_compressed (compressed buffer tile index)
  // d_batch_perm[i]: index into shard-ordered output position
  // batch_tile_count: total tiles in the batch (K_l * M_l)
  // batch_covering_count: covering count for shard ordering (K_l * C_l)
  // Returns 0 on success.
  int aggregate_batch_by_shard_async(
    void* d_compressed,
    size_t* d_comp_sizes,
    const uint32_t* d_batch_gather,
    const uint32_t* d_batch_perm,
    uint64_t batch_tile_count,
    uint64_t batch_covering_count,
    size_t max_chunk_bytes,
    struct aggregate_slot* slot,
    CUstream stream);

#ifdef __cplusplus
}
#endif
