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

  // Compute the aggregated-buffer size for the GPU streaming path that uses
  // tail carry-over (shard-capacity-sized regions per shard). Returns 0 on
  // overflow or when num_shards is 0.
  struct aggregate_layout;
  size_t agg_pool_bytes_layout(const struct aggregate_layout* layout);

  // Aggregate layout fields. Host fields are always valid.
  // d_* fields are GPU device pointers (NULL on CPU).
  struct aggregate_layout
  {
    uint8_t lifted_rank; // 2 * (rank - n_append)
    uint64_t lifted_shape[MAX_RANK];
    int64_t lifted_strides[MAX_RANK];
    uint64_t* d_lifted_shape;  // device copy (NULL on CPU)
    int64_t* d_lifted_strides; // device copy (NULL on CPU)
    uint64_t chunks_per_epoch; // M: actual chunk count
    uint64_t covering_count;   // C >= M: product of padded dims
    size_t max_comp_chunk_bytes;
    uint64_t cps_inner;        // product of chunks_per_shard for inner dims
    uint64_t num_shards;       // covering_count / cps_inner
    uint32_t active_count_max; // worst-case active epochs per batch
    size_t page_size;          // 0 = no padding
    // Per-shard reservation in d_aggregated. Sized so each shard has space for
    // a leading tail (< page_size) carried over from the prior batch plus the
    // worst-case batch_active_count * cps_inner * max_comp_chunk_bytes.
    // Page-aligned. Zero when page_size == 0.
    size_t shard_capacity;
  };

  // Compute host-side aggregate layout fields (pure CPU, no GPU allocation).
  // active_count_max is the maximum number of active epochs per batch for this
  // level (used to size shard_capacity).
  int aggregate_layout_compute(struct aggregate_layout* layout,
                               uint8_t rank,
                               uint8_t n_append,
                               const uint64_t* chunk_count,
                               const uint64_t* chunks_per_shard,
                               uint64_t chunks_per_epoch,
                               size_t max_comp_chunk_bytes,
                               uint32_t active_count_max,
                               size_t page_size);

  // Aggregated output for shard delivery (CUDA-free).
  struct aggregate_result
  {
    void* data;          // aggregated compressed chunks in shard order
    size_t* offsets;     // [C+1] byte offsets (prefix sum)
    size_t* chunk_sizes; // [C] real pre-padding compressed sizes
  };

  // Compute gather + perm LUTs for a batch of active_count epochs.
  // pool_epochs[a] gives the compressed-pool epoch index for active slot a.
  // Perm uses epoch-major shard order matching deliver_to_shards_batch():
  //   target = si * active_count * cps_inner + a * cps_inner + c
  struct level_geometry;
  void aggregate_batch_luts(const struct aggregate_layout* agg,
                            const struct level_geometry* levels,
                            int lv,
                            uint32_t active_count,
                            const uint32_t* pool_epochs,
                            uint32_t* out_gather,
                            uint32_t* out_perm);

  // Per-LOD segment within a unified per-batch aggregate. Tracks both the
  // per-LOD geometry needed by the aggregate inner loops and the offsets
  // identifying this LOD's slice of the unified gather / perm / sizes /
  // offsets / data arrays.
  struct lod_segment
  {
    uint64_t chunks_per_epoch;       // M_lv
    uint64_t covering_count;         // C_lv
    uint64_t chunks_per_shard_inner; // cps_inner_lv
    uint32_t n_active;               // active epochs for this LOD in batch

    uint64_t batch_chunk_offset;    // start in unified gather/perm
    uint64_t batch_covering_offset; // start in unified offsets/chunk_sizes/...
    size_t data_segment_offset;     // page-aligned byte offset in shared data
    size_t data_segment_bytes;      // capacity reserved for this segment
  };

  // Unified per-batch aggregate layout: collects per-LOD segment info plus
  // the totals needed to size unified scratch and gather buffers.
  struct batch_aggregate_layout
  {
    uint8_t nlod;
    size_t page_size;
    size_t max_comp_chunk_bytes; // shared across LODs (uniform pool stride)
    struct lod_segment lods[LOD_MAX_LEVELS];

    uint64_t total_batch_chunks;   // sum of n_active * chunks_per_epoch
    uint64_t total_batch_covering; // sum of n_active * covering_count
    size_t total_data_bytes;       // sum of segment caps, each page-aligned
  };

  // Populate a batch layout from the per-LOD aggregate_layouts and active
  // counts. Each LOD's data segment offset is page-aligned within the
  // unified buffer so deliver-time write_direct guards still hold.
  // Returns 0 on success, non-zero on overflow / inconsistency.
  int batch_aggregate_layout_init(struct batch_aggregate_layout* out,
                                  const struct aggregate_layout* per_lod,
                                  const uint32_t* per_lod_n_active,
                                  uint8_t nlod,
                                  size_t page_size);

  // Build unified gather + perm for a batch across all LODs.
  // per_lod and pool_epochs are ragged: pool_epochs[lv][a] for a in
  // 0..layout->lods[lv].n_active. out_gather/out_perm are sized
  // total_batch_chunks. perm values are unified — i.e. per-LOD perm
  // targets are already shifted by batch_covering_offset.
  void aggregate_batch_luts_unified(const struct batch_aggregate_layout* layout,
                                    const struct aggregate_layout* per_lod,
                                    const struct level_geometry* levels,
                                    const uint32_t* const* pool_epochs,
                                    uint32_t* out_gather,
                                    uint32_t* out_perm);

#ifdef __cplusplus
}
#endif
