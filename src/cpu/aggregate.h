#pragma once

#include "types.aggregate.h"
#include <stddef.h>

// Aggregate compressed chunks into shard order on CPU.
//
// compressed: input buffer, chunk i at offset i * max_comp_chunk_bytes.
// comp_sizes[M]: actual compressed size of each chunk.
// layout: pre-computed aggregate layout (from aggregate_layout_compute).
// result: output — caller must free result->data, result->offsets,
//         result->chunk_sizes via aggregate_cpu_result_free.
//
// Returns 0 on success.
int aggregate_cpu(const void* compressed,
                  const size_t* comp_sizes,
                  const struct aggregate_layout* layout,
                  struct aggregate_result* result);

void aggregate_cpu_result_free(struct aggregate_result* result);

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
int aggregate_cpu_workspace_init(struct aggregate_cpu_workspace* ws,
                                 const struct aggregate_layout* layout);

void aggregate_cpu_workspace_free(struct aggregate_cpu_workspace* ws);

// Same as aggregate_cpu but uses pre-allocated workspace. Zero mallocs.
// result->data/offsets/chunk_sizes point into ws — caller must NOT free them.
int aggregate_cpu_into(const void* compressed,
                       const size_t* comp_sizes,
                       const struct aggregate_layout* layout,
                       struct aggregate_cpu_workspace* ws,
                       struct aggregate_result* result);

// Batch variant: aggregate n_active epochs at once using gather LUT.
// gather[n_active * M]: maps batch input index → compressed chunk index.
// ws->perm must be the batch perm [n_active * M] (precomputed, interleaved).
// ws buffers sized for n_active * C covering positions.
// compressed_base / comp_sizes_base: full compressed buffer (all epochs).
int aggregate_cpu_batch_into(const void* compressed_base,
                              const size_t* comp_sizes_base,
                              const uint32_t* gather,
                              const struct aggregate_layout* layout,
                              uint32_t n_active,
                              struct aggregate_cpu_workspace* ws,
                              struct aggregate_result* result);
