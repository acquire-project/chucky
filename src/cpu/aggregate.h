#pragma once

#include "stream/types.aggregate.h"
#include <stddef.h>

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
                   int nthreads);

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
                         int nthreads);

// Init a batch-shaped workspace for the worst-case unified layout. Allocates
// the unified data buffer page-aligned so deliver-time write_direct guards
// hold even with O_DIRECT shard sinks. Caller is expected to size the
// scratch arrays for the maximum batch (typically batch_active_count active
// epochs across every LOD).
int
aggregate_cpu_workspace_init_batch(struct aggregate_cpu_workspace* ws,
                                   uint64_t max_total_chunks,
                                   uint64_t max_total_covering,
                                   size_t max_total_data_bytes,
                                   size_t alignment);

// Free a batch-shaped workspace. Mirrors aggregate_cpu_workspace_free but
// uses platform_aligned_free for the data buffer.
void
aggregate_cpu_workspace_free_batch(struct aggregate_cpu_workspace* ws);

// Unified per-batch aggregate. Single OpenMP loop over total_batch_chunks
// gathers compressed chunks from across all LODs into the shared, page-
// aligned data buffer. Per-LOD result views are written into per_lod_results
// so the caller's deliver-per-LOD path can hand each segment to its sink.
int
aggregate_cpu_batch_into_unified(
  const void* compressed_base,
  const size_t* comp_sizes_base,
  const uint32_t* gather,    // [total_batch_chunks]
  const uint8_t* source_lod, // [total_batch_chunks]
  const struct batch_aggregate_layout* layout,
  struct aggregate_cpu_workspace* ws,
  struct aggregate_result* per_lod_results, // [layout->nlod]
  int nthreads);
