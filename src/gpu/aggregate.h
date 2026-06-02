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

  int aggregate_cub_temp_bytes(uint64_t count, size_t* out_bytes);

  struct batch_slice_entry
  {
    size_t data_base_offset;
    uint64_t desc_base_offset;
    uint8_t nlod;
    struct lod_segment per_lod_lods[LOD_MAX_LEVELS];
  };

  struct aggregate_slot;

  struct slot_runtime_state
  {
    size_t cursor;
    uint64_t desc_cursor;
    uint32_t batches_per_slot;
    uint32_t _pad;
  };

  struct aggregate_append_measurement
  {
    size_t data_bytes;
    uint64_t desc_entries;
    int32_t closes_after_append;
    int32_t tail_rollforward_blocked;
  };

  struct aggregate_slot_reservation
  {
    int32_t slot;
    int32_t close_slot;
    size_t data_base;
    uint64_t desc_base;
    uint32_t batch_index;
  };

  struct d_routing
  {
    void* target_d_aggregated;
    size_t* target_d_offsets;
    size_t* target_d_permuted_sizes;
    size_t* target_d_shard_base_offsets_dense;
    size_t data_base_offset;
    uint64_t desc_base_offset;
    size_t actual_bytes;
    uint32_t batch_idx_in_slot;
    int32_t target_slot_idx;
    int32_t close_prior_slot_idx; // -1 if no close
    struct aggregate_append_measurement measurement;
  };

  struct agg_routing_cb_args
  {
    struct aggregate_slot* slots[2];
    volatile struct d_routing* h_routing;
    struct batch_aggregate_layout layout;
    uint8_t nlod;
    uint32_t per_lod_n_active[LOD_MAX_LEVELS];
  };

  struct slot_dev_ptrs
  {
    void* d_aggregated;
    size_t* d_offsets;
    size_t* d_permuted_sizes;
    size_t* d_shard_base_offsets_dense;
    struct slot_runtime_state* d_runtime;
  };

  int route_reservation_launch(
    struct d_routing* d_routing,
    struct slot_dev_ptrs target,
    const struct aggregate_append_measurement* d_measurement,
    const struct aggregate_slot_reservation* reservation,
    CUstream stream);

  int aggregate_measurement_launch(
    struct aggregate_append_measurement* d_measurement,
    const size_t* d_actual_bytes_ptr,
    const size_t* d_tail_sum_bytes_ptr,
    int include_tail_sum,
    uint64_t desc_entries,
    int closes_after_append,
    int tail_rollforward_blocked,
    CUstream stream);

  int dense_offsets_launch(struct d_routing* d_routing,
                           const size_t* d_shard_sum_bytes,
                           const size_t* d_tail_bytes,
                           uint64_t total_shards,
                           CUstream stream);

  int copy_to_slot_launch(const struct d_routing* d_routing,
                          const size_t* d_temp_offsets,
                          const size_t* d_temp_perm_sizes,
                          uint64_t count,
                          CUstream stream);

  struct aggregate_slot
  {
    size_t* d_permuted_sizes; // device: slot_chunk_cap size_t
    size_t* d_offsets;        // device: slot_chunk_cap size_t
    void* d_aggregated;       // device: comp_pool_bytes
    void* h_aggregated;       // host pinned: comp_pool_bytes
    size_t* h_offsets;        // host pinned: slot_chunk_cap size_t
    size_t* h_permuted_sizes; // host pinned: slot_chunk_cap size_t
    void* d_temp;
    size_t temp_bytes;

    size_t slot_cursor;
    size_t slot_desc_cursor;
    size_t slot_capacity_bytes;
    uint64_t slot_desc_capacity;
    uint32_t batches_per_slot;
    uint32_t batches_per_slot_cap;
    struct batch_slice_entry* slot_batches;
    struct slot_runtime_state* d_runtime;

    size_t* d_shard_sum_bytes;
    volatile size_t* h_shard_base_offsets_dense;
    size_t* d_shard_base_offsets_dense;
    CUevent host_func_done;

    // Reset on close and on swap (callback sees bi==0).
    uint64_t stacked_n_active[LOD_MAX_LEVELS];

    CUevent ready;
    // All cap-stacked batches share one io_event — fine for FIFO delivery,
    // but batches cannot be retired independently.
    struct io_event io_done;
  };

  int aggregate_layout_upload(struct aggregate_layout* layout);

  void aggregate_layout_destroy(struct aggregate_layout* layout);

  void aggregate_slot_destroy(struct aggregate_slot* slot);

  void output_slot_close_reset(struct aggregate_slot* slot);

  int slot_can_fit(const struct aggregate_slot* slot, size_t next_batch_bytes);

  // Returns 1 if no shard would finalize during the next in-slot batch.
  // Mid-slot finalization corrupts the GPU's in-slot tail rollforward, so
  // cap>1 stacking with page_size>0 requires this to be true.
  int no_shard_finalizes(uint8_t nlod,
                         const uint64_t* epoch_in_shard,
                         const uint32_t* n_active_next,
                         const uint64_t* cps_append);

  int aggregate_batch_slot_init(struct aggregate_slot* slot,
                                uint64_t slot_chunk_cap,
                                size_t comp_pool_bytes,
                                uint32_t batches_per_slot_cap,
                                uint64_t max_total_shards);

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

  int aggregate_batch_measure_unified_async(
    const void* d_compressed,
    size_t* d_comp_sizes,
    const uint32_t* d_batch_gather,
    const uint32_t* d_batch_perm,
    uint64_t total_batch_chunks,
    uint64_t total_batch_covering,
    uint8_t nlod,
    size_t max_comp_chunk_bytes,
    struct aggregate_slot* slot,
    const size_t* d_shard_capacity,
    const uint64_t* d_shard_tps_group,
    const uint64_t* d_shard_offsets_base,
    size_t* d_tail_bytes,
    CUdeviceptr d_tail_carry,
    size_t page_size,
    uint64_t total_shards,
    int would_finalize_stay,
    int would_finalize_alone,
    struct aggregate_append_measurement* d_measurement,
    volatile struct aggregate_append_measurement* h_measurement,
    CUevent measurement_ready,
    size_t* d_tail_sum_bytes,
    size_t* d_temp_offsets,
    size_t* d_temp_perm_sizes,
    CUstream stream);

  int aggregate_batch_write_reserved_unified_async(
    const void* d_compressed,
    size_t* d_comp_sizes,
    const uint32_t* d_batch_gather,
    const uint32_t* d_batch_perm,
    uint64_t total_batch_chunks,
    uint64_t total_batch_covering,
    uint8_t nlod,
    size_t max_comp_chunk_bytes,
    struct aggregate_slot* scratch_slot,
    struct aggregate_slot* target_slot,
    const struct aggregate_slot_reservation* reservation,
    const uint64_t* d_shard_tps_group,
    const uint64_t* d_shard_offsets_base,
    size_t* d_tail_bytes,
    CUdeviceptr d_tail_carry,
    size_t page_size,
    uint64_t total_shards,
    struct d_routing* d_routing,
    struct aggregate_append_measurement* d_measurement,
    size_t* d_temp_offsets,
    size_t* d_temp_perm_sizes,
    struct agg_routing_cb_args* cb_args,
    CUstream stream);

#ifdef __cplusplus
}
#endif
