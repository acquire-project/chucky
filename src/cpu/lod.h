#pragma once

#include "lod_plan.h"
#include "types.lod.h"
#include "types.stream.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Scatter linear input into morton-ordered LOD L0 buffer.
  // dst must have room for batch_count * lod_counts[0] elements.
  int lod_cpu_scatter(const struct lod_plan* p,
                      const void* src,
                      void* dst,
                      enum lod_dtype dtype);

  // Reduce across LOD levels in-place.
  // values buffer holds all levels: total = levels.ends[nlod-1] elements.
  int lod_cpu_reduce(const struct lod_plan* p,
                     void* values,
                     enum lod_dtype dtype,
                     enum lod_reduce_method method);

  // Scatter + reduce (allocates output). Returns 0 on success.
  int lod_cpu_compute(const struct lod_plan* p,
                      const void* src,
                      void** out_values,
                      enum lod_dtype dtype,
                      enum lod_reduce_method method);

  // Scatter level `lv` from morton-ordered values into chunk pool using
  // the given tile_stream_layout (lifted shape/strides for that level).
  // batch_chunk_offset: offset (in elements) into chunk pool for each batch.
  int lod_cpu_morton_to_chunks(const struct lod_plan* p,
                               const void* values,
                               void* chunk_pool,
                               int lv,
                               const struct tile_stream_layout* layout,
                               const uint64_t* batch_chunk_offsets,
                               enum lod_dtype dtype);

  // Dim0 fold: accumulate inner-reduced data (levels 1+) from the morton
  // buffer into the accumulator. On first call (counts[lv]==0) copies;
  // subsequent calls reduce (mean/min/max).
  // accum: buffer sized to sum(batch_count * lod_counts[lv]) for lv=1..nlod-1.
  // counts[nlod]: per-level fold count (caller increments after this call).
  int lod_cpu_dim0_fold(const struct lod_plan* p,
                        const void* morton_values,
                        void* accum,
                        const uint32_t* counts,
                        enum lod_dtype dtype,
                        enum lod_reduce_method method);

  // Dim0 emit: finalize accumulator for level lv back to morton buffer.
  // For float mean: divides by count. For int mean/min/max: copies.
  int lod_cpu_dim0_emit(const struct lod_plan* p,
                        void* morton_values,
                        const void* accum,
                        int lv,
                        uint32_t count,
                        enum lod_dtype dtype,
                        enum lod_reduce_method method);

#ifdef __cplusplus
}
#endif
