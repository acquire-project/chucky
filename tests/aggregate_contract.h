#pragma once

// Contract verification for `struct aggregate_result`.
//
// CPU and GPU aggregators both produce an `aggregate_result` consumed by
// `deliver_to_shards_batch`. The semantics MUST be the same on both sides;
// otherwise delivery — which is the single shared consumer — silently
// addresses memory differently depending on which producer ran. This file
// makes the contract explicit and testable.
//
// Contract (carry-over mode, layout->page_size > 0):
//   1. result->data is the base of the LOD's per-shard region buffer.
//      Inside it, shard si occupies bytes
//         [si * shard_capacity, (si+1) * shard_capacity).
//   2. result->offsets[j] is the byte offset of chunk j relative to
//      result->data. Chunks are laid out in shard-permuted, epoch-major
//      order: shard 0 epoch 0..K-1, shard 1 epoch 0..K-1, ...
//   3. The first chunk of shard si lands at
//         si * shard_capacity + tail_in[si]
//      where tail_in[si] is the leading-tail length carried in from the
//      previous batch (0 on the first batch). The leading-tail bytes
//      occupy [si*shard_capacity, si*shard_capacity + tail_in[si]).
//   4. Within a shard, chunks pack tightly:
//         offsets[j+1] - offsets[j] == result->chunk_sizes[j]
//      for every j strictly inside the shard.
//   5. Each chunk fits inside its shard's region:
//         offsets[j] + chunk_sizes[j] <= (si+1) * shard_capacity.
//   6. result->chunk_sizes[j] is the real (pre-padding) compressed size
//      of chunk j, NOT the padded value used for byte placement.
//
// Contract (legacy contiguous mode, layout->page_size == 0):
//   1. result->data is the base of the LOD's data buffer.
//   2. result->offsets is one tightly-packed prefix sum starting at 0.
//   3. offsets[j+1] - offsets[j] == result->chunk_sizes[j] everywhere.
//
// The verifier runs in-process, returns 0 on success and a non-zero count
// of distinct violations on failure (each violation logged via log_error).
//
// Caller-supplied tail_in array (carry-over mode only):
//   tail_in[si] for si in [0, num_shards). May be NULL to assert no
//   leading tails on this batch (equivalent to all zeros).

#include <stddef.h>
#include <stdint.h>

struct aggregate_result;
struct aggregate_layout;

// Verify the carry-over contract. Asserts the layout is in carry-over mode
// (page_size > 0, shard_capacity > 0). n_active is the number of active
// epochs for this LOD in this batch.
int
verify_aggregate_result_carryover(const struct aggregate_result* result,
                                  const struct aggregate_layout* layout,
                                  const size_t* tail_in,
                                  uint32_t n_active);

// Verify the legacy contiguous contract. Asserts page_size == 0.
int
verify_aggregate_result_contiguous(const struct aggregate_result* result,
                                   const struct aggregate_layout* layout,
                                   uint32_t n_active);
