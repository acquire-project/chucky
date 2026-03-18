#pragma once

#include "io_queue.h"
#include "types.aggregate.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

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

  // Upload pre-computed layout arrays to GPU. Must be called after
  // aggregate_layout_compute. Returns 0 on success.
  int aggregate_layout_upload(struct aggregate_layout* layout);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  int aggregate_by_shard_async(const struct aggregate_layout* layout,
                               void* d_compressed,
                               size_t* d_comp_sizes,
                               struct aggregate_slot* slot,
                               CUstream stream);

  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t batch_chunk_count,
                                uint64_t batch_covering_count,
                                size_t comp_pool_bytes);

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
