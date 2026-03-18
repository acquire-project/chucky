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
// ratio 0 -> chunk_size = 1 (that dim doesn't contribute to chunk volume).
// For non-zero ratios: bits_per_part = ceil(log2(nelem) / sum(ratios)).
// Each dim gets chunk_size = 1 << (ratio[i] * bits_per_part).
// Remainder bits (may be negative) go to the lowest-indexed non-zero-ratio dim.
void
dims_budget_chunk_size(struct dimension* dims,
                       uint8_t rank,
                       uint64_t nelem,
                       const uint8_t* ratios);

// Set chunks_per_shard to achieve target shard counts.
// shard_counts has rank elements. 0 means "skip" (don't modify).
// Requires chunk_size to be set first.
void
dims_set_shard_counts(struct dimension* dims,
                      uint8_t rank,
                      const uint64_t* shard_counts);

// Distribute target_chunk_bytes across dims using power-of-2 ratios.
// Like dims_budget_chunk_size but accepts a byte target instead of elements.
// Computes nelem = target_chunk_bytes / bytes_per_element, then delegates.
void
dims_budget_chunk_bytes(struct dimension* dims,
                        uint8_t rank,
                        size_t target_chunk_bytes,
                        size_t bytes_per_element,
                        const uint8_t* ratios);

// Callback: estimate total bytes required for the current dim configuration.
// Returns 0 on success, non-zero on error.
typedef int (*dims_estimate_fn)(void* ctx, size_t* estimated_bytes_out);

// Find the largest power-of-2 chunk size (starting from target_chunk_bytes)
// that fits within budget_bytes according to the estimate callback.
//
// Algorithm: repeatedly calls dims_budget_chunk_bytes to set chunk sizes,
// then estimate(ctx, &est) to check total usage. Halves target_chunk_bytes
// until est <= budget_bytes.
//
// Modifies dims in place. On success, dims hold the chosen chunk sizes.
// Returns 0 on success, non-zero if no chunk size fits (even 1 element/chunk).
int
dims_advise(struct dimension* dims,
            uint8_t rank,
            size_t target_chunk_bytes,
            size_t bytes_per_element,
            const uint8_t* ratios,
            size_t budget_bytes,
            dims_estimate_fn estimate,
            void* estimate_ctx);

// Print a summary table of the dimension configuration.
void
dims_print(const struct dimension* dims, uint8_t rank);
