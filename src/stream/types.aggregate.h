#pragma once

#include "defs.limits.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Compute the aggregated-buffer size including page-alignment padding slack.
  // covering_count and cps_inner are per-epoch values.
  // Returns 0 on overflow (callers use this for allocation sizes).
  size_t agg_pool_bytes(uint64_t chunk_count,
                        size_t max_comp_chunk_bytes,
                        uint64_t covering_count,
                        uint64_t cps_inner,
                        size_t page_size);

  // Aggregate layout fields. Host fields are always valid.
  // d_* fields are GPU device pointers (NULL on CPU).
  struct aggregate_layout
  {
    uint8_t lifted_rank; // 2 * (rank - 1)
    uint64_t lifted_shape[MAX_RANK];
    int64_t lifted_strides[MAX_RANK];
    uint64_t* d_lifted_shape;  // device copy (NULL on CPU)
    int64_t* d_lifted_strides; // device copy (NULL on CPU)
    uint64_t chunks_per_epoch; // M: actual chunk count
    uint64_t covering_count;   // C >= M: product of padded dims
    size_t max_comp_chunk_bytes;
    uint64_t cps_inner; // product of chunks_per_shard for inner dims
    size_t page_size;   // 0 = no padding
  };

  // Compute host-side aggregate layout fields (pure CPU, no GPU allocation).
  int aggregate_layout_compute(struct aggregate_layout* layout,
                               uint8_t rank,
                               const uint64_t* chunk_count,
                               const uint64_t* chunks_per_shard,
                               uint64_t chunks_per_epoch,
                               size_t max_comp_chunk_bytes,
                               size_t page_size);

  // Aggregated output for shard delivery (CUDA-free).
  struct aggregate_result
  {
    void* data;          // aggregated compressed chunks in shard order
    size_t* offsets;     // [C+1] byte offsets (prefix sum)
    size_t* chunk_sizes; // [C] real pre-padding compressed sizes
  };

#ifdef __cplusplus
}
#endif
