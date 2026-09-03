#pragma once

#include "defs.limits.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct aggregate_layout
  {
    uint8_t lifted_rank; // 2 * (rank - n_append)
    uint64_t lifted_shape[MAX_RANK];
    int64_t lifted_strides[MAX_RANK];
    uint64_t chunks_per_epoch; // M: actual chunk count
    uint64_t covering_count;   // C >= M: product of padded dims
    size_t max_comp_chunk_bytes;
    uint64_t cps_inner;  // product of chunks_per_shard for inner dims
    uint64_t num_shards; // covering_count / cps_inner
    size_t page_size;    // 0 = no padding
    uint64_t chunks_per_shard_append; // shard length along append dims
  };

  // Per-shard footer size: one page for trailing sub-page data, index
  // entries (16 bytes per chunk), 4-byte CRC, padded to page boundary.
  // Used by init_shard_state to size shard_state.footer_buf_pool. Returns 0
  // when page_size is 0 (no alignment requirement) or on overflow.
  size_t footer_capacity_for(uint64_t chunks_per_shard_total, size_t page_size);

  // Compute host-side aggregate layout fields (pure CPU, no GPU allocation).
  int aggregate_layout_compute(struct aggregate_layout* layout,
                               uint8_t rank,
                               uint8_t n_append,
                               const uint64_t* chunk_count,
                               const uint64_t* chunks_per_shard,
                               uint64_t chunks_per_epoch,
                               size_t max_comp_chunk_bytes,
                               size_t page_size,
                               uint64_t chunks_per_shard_append);

  // Compute gather + perm LUTs for a batch of active_count epochs.
  // pool_epochs[a] gives the compressed-pool epoch index for active slot a.
  // Perm uses epoch-major shard order:
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
  };

  // Unified per-batch aggregate layout: collects per-LOD segment info plus
  // the totals needed to size unified scratch and gather buffers.
  struct batch_aggregate_layout
  {
    uint8_t nlod;
    size_t max_comp_chunk_bytes; // shared across LODs (uniform pool stride)
    struct lod_segment lods[LOD_MAX_LEVELS];

    uint64_t total_batch_chunks;   // sum of n_active * chunks_per_epoch
    uint64_t total_batch_covering; // sum of n_active * covering_count
    size_t total_data_bytes;       // total aggregate capacity across all LODs
  };

  // Compact CPU/GPU aggregate layout.
  int batch_aggregate_layout_init_compact(
    struct batch_aggregate_layout* out,
    const struct aggregate_layout* per_lod,
    const uint32_t* per_lod_n_active,
    uint8_t nlod);

  // Build the host shadow of the unified offset/size arrays for a fixed-size
  // aggregate.  Covering entries without a real chunk remain zero-sized and
  // offsets stay absolute in the compact device aggregate.
  int aggregate_fixed_host_index(const struct batch_aggregate_layout* layout,
                                 const struct aggregate_layout* per_lod,
                                 size_t fixed_chunk_bytes,
                                 size_t* offsets,
                                 size_t* chunk_sizes);

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
