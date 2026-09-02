#pragma once

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Scratch bytes an exclusive sum over count elements needs. The memory
  // estimate asks without a device, so an unanswerable query leaves 0 rather
  // than failing a configuration that is fine.
  int aggregate_cub_temp_bytes(uint64_t count, size_t* out_bytes);

  struct aggregate_slot
  {
    size_t* d_permuted_sizes; // device: (C+1) size_t, zeroed each epoch
    size_t* d_offsets;        // device: (C+1) size_t
    void* d_aggregated;       // device: compact aggregate capacity
    size_t* h_offsets;        // host pinned: (C+1) size_t
    size_t* h_permuted_sizes; // host pinned: C size_t (real compressed sizes)
    void* d_temp;             // CUB scratch
    size_t temp_bytes;
  };

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t batch_covering_count,
                                size_t device_data_bytes);

  // Sizing mirror of aggregate_batch_slot_init, for the memory estimate.
  int aggregate_batch_slot_memory(uint64_t batch_covering_count,
                                  size_t device_data_bytes,
                                  size_t* device_bytes,
                                  size_t* host_bytes);

  // clang-format off
  // Single compact dispatch across all LODs. Per-LOD gather and permutation
  // LUTs are unified, so the kernels themselves are level-agnostic.
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
  //   nlod                 : number of LODs (one zero-sized sentinel per LOD)
  //   max_comp_chunk_bytes : uniform pool stride
  //   slot                 : unified aggregate_slot (d_aggregated sized for
  //                          compact payload; d_offsets/d_permuted_sizes sized
  //                          to max_total_batch_covering + LOD_MAX_LEVELS).
  //   stream               : compress stream
  // clang-format on
  int aggregate_batch_unified_async(const void* d_compressed,
                                    size_t* d_comp_sizes,
                                    const uint32_t* d_batch_gather,
                                    const uint32_t* d_batch_perm,
                                    uint64_t total_batch_chunks,
                                    uint64_t total_batch_covering,
                                    uint8_t nlod,
                                    size_t max_comp_chunk_bytes,
                                    struct aggregate_slot* slot,
                                    CUstream stream);

#ifdef __cplusplus
}
#endif
