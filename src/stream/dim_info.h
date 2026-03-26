#pragma once

#include "dimension.h"
#include <stdint.h>

// View into a contiguous range of dimensions (beg/end pointer pair).
struct dim_slice
{
  const struct dimension* beg;
  const struct dimension* end;
};

static inline uint8_t
dim_slice_len(struct dim_slice s)
{
  return (uint8_t)(s.end - s.beg);
}

// Resolved partition of dims into append and inner groups.
// Constructed once by dim_info_init(); immutable after.
//
// Append dims: leftmost contiguous prefix with chunk_size == 1.
// Inner dims: everything else.
// The slices point into the original dimension array (no copies).
struct dim_info
{
  struct dim_slice append; // dims[0 .. n_append)
  struct dim_slice inner;  // dims[n_append .. rank)

  int append_downsample;       // rightmost append dim has downsample
  uint32_t lod_mask;           // bitmask: inner dims with downsample=1
  uint64_t inner_append_count; // prod(chunk_count[d] for d=1..n_append-1)
};

static inline uint8_t
dim_info_n_append(const struct dim_info* info)
{
  return dim_slice_len(info->append);
}

static inline uint8_t
dim_info_rank(const struct dim_info* info)
{
  return dim_slice_len(info->append) + dim_slice_len(info->inner);
}

// Absolute index of a dimension pointer within the original array.
static inline int
dim_index(const struct dim_info* info, const struct dimension* d)
{
  return (int)(d - info->append.beg);
}

// Partition dims into append/inner, validate constraints, precompute
// derived values. Returns 0 on success.
//
// Validates: chunk_size > 0, storage_position is valid permutation with
// append dims pinned, only dim 0 may be unbounded, only rightmost
// append dim may have downsample.
int
dim_info_init(struct dim_info* info,
              const struct dimension* dims,
              uint8_t rank);
