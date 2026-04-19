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

// Distribute nelem across dims using power-of-2 ratios.
//
// ratios[i] > 0  -> participates in the bit budget with this weight.
// ratios[i] == 0 -> chunk_size = 1 (no bits allocated).
// ratios[i] == -1-> pin chunk_size at dims[i].size. If dims[i].size == 0
//                  (unbounded dim 0), treated as weight=1: the dim absorbs
//                  the remaining bit budget. Only dim 0 may be unbounded.
//
// Bit allocation is greedy over budget participants. The remaining element
// budget for participants is nelem / prod(pinned sizes).
void
dims_budget_chunk_size(struct dimension* dims,
                       uint8_t rank,
                       uint64_t nelem,
                       const int* ratios);

// Set chunks_per_shard to achieve target shard counts.
// shard_counts has rank elements. 0 means "skip" (don't modify).
// Requires chunk_size to be set first.
void
dims_set_shard_counts(struct dimension* dims,
                      uint8_t rank,
                      const uint64_t* shard_counts);

// Choose shard geometry from a byte floor and a concurrency bound.
//
// Policy:
//   - Inner dims (d >= n_append): integer-greedy allocation of shard count
//     to dims, bounded so the product <= target_concurrent_shards (no pow2
//     rounding). Each step increments the inner dim with the largest
//     remaining n_chunks[d]/shards[d] ratio while staying within
//     n_chunks[d].
//   - Outer append dim (d = 0): chunks_per_shard is maximized subject to
//     (a) chunks_per_shard_total <= MAX_PARTS_PER_SHARD (backend parts cap)
//     and (b) cps <= n_chunks[0] (dim extent). When min_append_shards > 1
//     it's an authoritative cap (ceildiv(n_chunks[0], cps) >= N); it wins
//     over the byte floor if they conflict. When unset, min_shard_bytes is
//     a byte floor: cps >= ceildiv(min_shard_bytes, row_bytes).
//     Unbounded dim 0 (size == 0) ignores the min_append_shards cap.
//   - Inner append dims (d in 1..na-1): pass through at chunks_per_shard =
//     n_chunks[d] so the downstream product (config.c) evaluates correctly.
//
// Requires chunk_size to be set first (e.g. via dims_budget_chunk_bytes).
// target_concurrent_shards of 0 is treated as 1 (no multiplexing).
// min_append_shards of 0 is treated as 1 (no minimum).
//
// Returns 0 on success, non-zero if min_shard_bytes < chunk_bytes (floor
// is meaningless below one chunk).
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
// Like dims_budget_chunk_size but accepts a byte target instead of elements.
// Computes nelem = target_chunk_bytes / bytes_per_element, then delegates.
void
dims_budget_chunk_bytes(struct dimension* dims,
                        uint8_t rank,
                        size_t target_chunk_bytes,
                        size_t bytes_per_element,
                        const int* ratios);
