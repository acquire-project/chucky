// User-facing API for configuring the dimension array that describes
// array shape, chunking, sharding, storage order, and downsampling.
#pragma once

#include <stddef.h>
#include <stdint.h>

enum dimension_axis_type
{
  dimension_axis_space = 0, // default (zero-init = space)
  dimension_axis_time,
  dimension_axis_channel,
  dimension_axis_other,
};

// OME-NGFF v0.5 axis metadata: unit, scale, and type.
// Only consumed by zarr metadata writers.
struct omengff_axis
{
  const char* unit; // axis unit (e.g. "micrometer"),
                    // NULL defaults to "index" in metadata
  double scale;     // physical pixel scale for coordinateTransformations
                    // (must be non-negative; 0 treated as 1.0)
  enum dimension_axis_type type; // space, time, channel, or other
};

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
  struct omengff_axis ngff;  // OME-NGFF v0.5 axis metadata
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

// Compute n_append: the number of leftmost append dimensions.
// Append dims form the leftmost contiguous prefix of dims where
// chunk_size == 1. If the rightmost append dim candidate has downsample set,
// the prefix stops there (only rightmost append dim may be accumulator-
// downsampled). Always returns at least 1.
uint8_t
dims_n_append(const struct dimension* dims, uint8_t rank);

// Validate a dimension array. Returns 0 on success, non-zero on error.
// Checks: chunk_size > 0, storage_position is a valid permutation with
// append dims pinned to positions 0..n_append-1, and unbounded dim 0 has
// chunks_per_shard > 0.
int
dims_validate(const struct dimension* dims, uint8_t rank);

// Print a summary table of the dimension configuration.
void
dims_print(const struct dimension* dims, uint8_t rank);
