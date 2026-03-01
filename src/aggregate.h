#pragma once

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
                            size_t max_chunk_bytes);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  // Initialize a slot (allocate device/host buffers).
  // comp_pool_bytes = tiles_per_epoch * max_chunk_bytes.
  int aggregate_slot_init(struct aggregate_slot* slot,
                          const struct aggregate_layout* layout,
                          size_t comp_pool_bytes);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  // Enqueue all 3 passes on stream. Returns 0 on success.
  int aggregate_by_shard_async(const struct aggregate_layout* layout,
                               void* d_compressed,
                               size_t* d_comp_sizes,
                               struct aggregate_slot* slot,
                               CUstream stream);

#ifdef __cplusplus
}
#endif
