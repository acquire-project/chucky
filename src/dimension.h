#pragma once

#include <stdint.h>

struct dimension;

// Initialize dims from a name string and sizes array.
// Each character in names is one dimension name. strlen(names) = rank.
// Sets: name, size, tile_size=size, storage_position=identity,
//       tiles_per_shard=0, downsample=0.
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

// Set tile_size for each dimension directly.
// tile_sizes has rank elements. Each must be > 0.
void
dims_set_tile_sizes(struct dimension* dims,
                    uint8_t rank,
                    const uint64_t* tile_sizes);

// Distribute nelem across dims using power-of-2 ratios.
//
// ratio 0 -> tile_size = 1 (that dim doesn't contribute to tile volume).
// For non-zero ratios: bits_per_part = ceil(log2(nelem) / sum(ratios)).
// Each dim gets tile_size = 1 << (ratio[i] * bits_per_part).
// Remainder bits (may be negative) go to the lowest-indexed non-zero-ratio dim.
void
dims_budget_tile_size(struct dimension* dims,
                      uint8_t rank,
                      uint64_t nelem,
                      const uint8_t* ratios);

// Set tiles_per_shard to achieve target shard counts.
// shard_counts has rank elements. 0 means "skip" (don't modify).
// Requires tile_size to be set first.
void
dims_set_shard_counts(struct dimension* dims,
                      uint8_t rank,
                      const uint64_t* shard_counts);

// Print a summary table of the dimension configuration.
void
dims_print(const struct dimension* dims, uint8_t rank);
