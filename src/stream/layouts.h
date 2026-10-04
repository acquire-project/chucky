#pragma once

#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "stream/dim_info.h"
#include "stream/types.aggregate.h"

#include <stddef.h>
#include <stdint.h>

struct tile_stream_layout
{
  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];

  // Packed acquisition extents within one epoch. Append dimensions span one
  // chunk; inner dimensions retain their logical size, including partial edges.
  uint64_t input_shape[HALF_MAX_RANK];

  uint64_t chunk_elements;
  uint64_t chunk_stride;
  uint64_t chunks_per_epoch;
  uint64_t epoch_elements; // logical input elements, excluding chunk padding
  size_t chunk_pool_bytes;
};

// Chunk alignment bytes do not affect whether the logical shape fills chunks.
static inline int
layout_has_partial_chunks(const struct tile_stream_layout* layout)
{
  return layout->epoch_elements !=
         layout->chunks_per_epoch * layout->chunk_elements;
}

#ifdef __cplusplus
extern "C"
{
#endif

  // A factored logical-to-chunk map: input_shape[last] column offsets followed
  // by epoch_elements / input_shape[last] row offsets. All offsets are uint64_t
  // elements. This is the same row-plus-column lookup used by LOD chunk
  // scatter. Returns 0 bytes if the table size overflows size_t.
  size_t chunk_scatter_lut_bytes(const struct tile_stream_layout* layout);
  void chunk_scatter_lut_build(const struct tile_stream_layout* layout,
                               uint64_t* lut);

#ifdef __cplusplus
}
#endif

// Per-level pre-computed layout information (CPU only, no GPU pointers).
struct level_layout_info
{
  struct aggregate_layout agg_layout;
  uint32_t batch_active_count;
  uint64_t chunks_per_shard_append;
  uint64_t chunks_per_shard_inner;
  uint64_t chunks_per_shard_total;
  uint64_t shard_inner_count;
};

// All pre-computed layout data from CPU-only math.
// Produced by compute_stream_layouts, consumed by the create path
// and the memory estimate path.
//
// Owns dims_owned[]: a deep copy of the caller's config.dimensions (including
// duplicated name strings) so dim_info slices and later metadata reads don't
// depend on the caller keeping its dimensions array alive.
struct computed_stream_layouts
{
  struct dimension dims_owned[HALF_MAX_RANK]; // owned copy of config dims
  uint8_t rank;
  struct dim_info
    dims; // resolved append/inner partition (points into dims_owned)
  struct lod_plan plan; // owned if enable_multiscale
  struct tile_stream_layout layouts[LOD_MAX_LEVELS]; // [0] = L0
  struct level_geometry levels;
  uint32_t epochs_per_batch;
  size_t max_output_size;
  struct level_layout_info per_level[LOD_MAX_LEVELS];
};
