#pragma once

#include "stream/types.aggregate.h"
#include <stddef.h>

struct threadpool;
struct shard_state;

// Pre-allocated workspace for zero-allocation aggregation.
struct aggregate_cpu_workspace
{
  uint32_t* perm;         // [M] ravel permutation (set per-call for shared use)
  size_t* permuted_sizes; // [C] scratch, zeroed each call
  size_t* offsets;        // [C+1] reused each call
  size_t* chunk_sizes;    // [C] reused each call
  void* data;             // output buffer (worst-case capacity)
  size_t data_capacity;
};

// Init workspace for a single layout: allocates all buffers, precomputes perm.
int
aggregate_cpu_workspace_init(struct aggregate_cpu_workspace* ws,
                             const struct aggregate_layout* layout);

void
aggregate_cpu_workspace_free(struct aggregate_cpu_workspace* ws);

// Same as aggregate_cpu but uses pre-allocated workspace. Zero mallocs.
// result->data/offsets/chunk_sizes point into ws — caller must NOT free them.
int
aggregate_cpu_into(const void* compressed,
                   const size_t* comp_sizes,
                   const struct aggregate_layout* layout,
                   struct aggregate_cpu_workspace* ws,
                   struct aggregate_result* result,
                   struct threadpool* pool);

// Batch variant: aggregate n_active epochs at once using gather LUT.
// gather[n_active * M]: maps batch input index → compressed chunk index.
// ws->perm must be the batch perm [n_active * M] (precomputed, interleaved).
// ws buffers sized for n_active * C covering positions.
// compressed_base / comp_sizes_base: full compressed buffer (all epochs).
int
aggregate_cpu_batch_into(const void* compressed_base,
                         const size_t* comp_sizes_base,
                         const uint32_t* gather,
                         const struct aggregate_layout* layout,
                         uint32_t n_active,
                         struct aggregate_cpu_workspace* ws,
                         struct aggregate_result* result,
                         struct threadpool* pool);

// Inputs to aggregate_cpu_batch_into_unified. Grouped because the call has
// many independent arguments; designated initializers at the call site keep
// each argument labeled.
struct aggregate_cpu_inputs
{
  // Compressed-pool inputs.
  const void* compressed_base;
  const size_t* comp_sizes_base;
  const uint32_t* gather; // [total_batch_chunks]

  // Layouts: unified batch + per-LOD geometry.
  const struct batch_aggregate_layout* layout;
  const struct aggregate_layout* per_lod_layouts; // [layout->nlod]

  // Tail-carry inputs (carry-over mode only). Both NULL when page_size == 0.
  // shards_by_lod[lv] is the shard_state for LOD lv. Each entry is independent;
  // the function does not assume a contiguous backing array.
  struct shard_state* const* shards_by_lod; // [layout->nlod]
  size_t* const* h_tail_bytes;              // [layout->nlod]

  // Workspace (reused) and outputs (per-LOD result views).
  struct aggregate_cpu_workspace* ws;
  struct aggregate_result* per_lod_results; // [layout->nlod]
  struct threadpool* pool;
};

// Unified per-batch aggregate. Gathers compressed chunks from across all LODs
// into the shared, page-aligned data buffer. Per-LOD result views are written
// into per_lod_results so the caller's deliver-per-LOD path can hand each
// segment to its sink.
//
// When layout->page_size > 0, lays out per-shard `shard_capacity`-sized
// regions matching the GPU tail-carry path: each shard's first chunk is
// anchored at `si*shard_capacity + h_tail_bytes[lv][si]`, and the leading
// tail bytes are copied from `shards_by_lod[lv]->shards[si].tail_buf`. When
// page_size is zero, falls back to a single contiguous prefix-sum per LOD.
int
aggregate_cpu_batch_into_unified(const struct aggregate_cpu_inputs* in);
