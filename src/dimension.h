// User-facing API for configuring the dimension array that describes
// array shape, chunking, sharding, storage order, and downsampling.
#pragma once

#include <stddef.h>
#include <stdint.h>

struct dimension
{
  uint64_t size; // 0 means unbounded (dim 0 only: stream indefinitely)
  uint64_t chunk_size;
  uint64_t chunks_per_shard; // 0 means all chunks along this dimension
                             // (must be > 0 when size == 0)
  const char* name;          // optional label (e.g. "x"), may be NULL
  int downsample;            // include in LOD pyramid
  uint8_t storage_position;  // position in storage layout (0=outermost).
                             // dims[0].storage_position must be 0.
                             // Must be a valid permutation of 0..rank-1.
};

// Initialize dims from a name string and sizes array.
// Each character in names is one dimension name. strlen(names) = rank.
// Sets: name, size, chunk_size=size, storage_position=identity,
//       chunks_per_shard=0, downsample=0.
// Returns rank. 0 on error.
uint8_t
dims_create(struct dimension* dims, const char* names, const uint64_t* sizes);

// Set storage order from a string of dimension names.
// Each character in order names a dimension; its position is the storage
// position assigned to that dimension. strlen(order) must equal rank.
// The first character must name dims[0] (the append dimension is pinned).
// Pass NULL for identity (0,1,2,...).
// Returns 0 on success, non-zero on error.
int
dims_set_storage_order(struct dimension* dims, uint8_t rank, const char* order);

// Sets downsample=1 on dims whose name character appears in names.
// Other dims set to downsample=0.
void
dims_set_downsample_by_name(struct dimension* dims,
                            uint8_t rank,
                            const char* names);

// Set chunk_size for each dimension directly.
// chunk_sizes has rank elements. Each must be > 0.
void
dims_set_chunk_sizes(struct dimension* dims,
                     uint8_t rank,
                     const uint64_t* chunk_sizes);

// Set chunks_per_shard to achieve target shard counts.
// shard_counts has rank elements. 0 means "skip" (don't modify).
// Requires chunk_size to be set first.
void
dims_set_shard_counts(struct dimension* dims,
                      uint8_t rank,
                      const uint64_t* shard_counts);

// Choose shard geometry from a byte floor, concurrency target, and
// append-shard floor, respecting the backend parts cap as a hard constraint.
//
// Policy (two-phase):
//   Phase A — fill target_concurrent_shards. Integer-greedy across inner
//     dims (d >= n_append): each step grows the dim with the largest
//     remaining n_chunks[d]/shards[d] ratio while Π shards[d] <= target.
//   Phase B — enforce parts budget. If inner_cps_prod is too big for the
//     parts cap given the required cps_append, keep splitting inner dims
//     past the target (target_concurrent_shards is soft). Picks the inner
//     dim with the largest current cps each step.
//
//   Outer append dim (d = 0): chunks_per_shard maximized within
//     MAX_PARTS_PER_SHARD / (inner_cps_prod · others_prod) and <= n_chunks[0].
//     When min_append_shards > 1, capped at floor(n_chunks[0] / N) —
//     authoritative over the byte floor. When min_shard_bytes > 0, Phase B
//     reserves budget for cps_append >= cps_floor when achievable; if not,
//     min_shard_bytes silently yields (soft).
//   Inner append dims (d in 1..na-1): pass through at chunks_per_shard =
//     n_chunks[d].
//
// Requires chunk_size to be set first (e.g. via dims_budget_chunk_bytes).
// target_concurrent_shards of 0 is treated as 1.
// min_append_shards of 0 or 1 is treated as "no minimum".
//
// Returns:
//   0 = success
//   1 = min_shard_bytes < chunk_bytes (floor meaningless below one chunk).
//       Caller can retry with smaller chunks or no floor.
//   2 = parts budget infeasible even with inner fully split. Caller can retry
//       with larger chunks, lower target_concurrent_shards, or lower
//       min_append_shards.
//   3 = invalid argument (null dims, rank==0, zero chunk_size, zero bpe).
int
dims_set_shard_geometry(struct dimension* dims,
                        uint8_t rank,
                        size_t min_shard_bytes,
                        uint32_t target_concurrent_shards,
                        uint32_t min_append_shards,
                        size_t bytes_per_element);

// Combined chunk + shard layout policy.
//
// When chunk_ratios != NULL: runs dims_budget_chunk_bytes first.
// Always runs dims_set_shard_geometry second. No ordering concerns
// for callers.
struct dims_layout_policy
{
  size_t bytes_per_element;
  size_t target_chunk_bytes; // ignored when chunk_ratios == NULL
  const int* chunk_ratios;   // NULL = leave chunk_size unchanged
  size_t min_shard_bytes;
  uint32_t target_concurrent_shards;
  uint32_t min_append_shards; // 0 = no minimum
};

int
dims_set_layout(struct dimension* dims,
                uint8_t rank,
                const struct dims_layout_policy* p);

// Distribute target_chunk_bytes across dims using power-of-2 ratios.
//
// ratios[i] > 0  -> bit-budget participant with this weight.
// ratios[i] == 0 -> chunk_size = 1 (no bits allocated).
// ratios[i] == -1-> pin chunk_size at dims[i].size. If dims[i].size == 0
//                  (unbounded dim 0), treated as weight=1: the dim absorbs
//                  the remaining bit budget. Only dim 0 may be unbounded.
//
// Bit allocation is greedy over participants; remaining element budget is
// nelem / prod(pinned sizes). Participant chunk_size is clamped to dim size
// for bounded dims (no clamp for unbounded dim 0).
//
// Returns:
//   0 = success.
//   1 = invalid input: bytes_per_element == 0, target_chunk_bytes <
//       bytes_per_element (budget smaller than one element), or pinned dims
//       alone exceed the budget.
int
dims_budget_chunk_bytes(struct dimension* dims,
                        uint8_t rank,
                        size_t target_chunk_bytes,
                        size_t bytes_per_element,
                        const int* ratios);
