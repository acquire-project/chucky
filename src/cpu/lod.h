#pragma once

#include "lod/lod_plan.h"
#include "stream/layouts.h"
#include "types.lod.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Reduce across LOD levels in-place.
  // values buffer holds all levels: total = levels.ends[nlod-1] elements.
  int lod_cpu_reduce(const struct lod_plan* p,
                     void* values,
                     enum dtype dtype,
                     enum lod_reduce_method method);

  // Build morton-to-chunk-pool LUT for level lv.
  // chunk_lut must have room for lod_nelem[lv] entries.
  void lod_cpu_build_chunk_lut(const struct lod_plan* p,
                               int lv,
                               const struct tile_stream_layout* layout,
                               uint32_t* chunk_lut);

  // Scatter level `lv` from morton-ordered values into chunk pool using
  // the given tile_stream_layout (lifted shape/strides for that level).
  // chunk_lut: precomputed via lod_cpu_build_chunk_lut (or NULL to build
  //            internally — legacy path for standalone use).
  // batch_chunk_offset: offset (in elements) into chunk pool for each batch.
  int lod_cpu_morton_to_chunks(const struct lod_plan* p,
                               const void* values,
                               void* chunk_pool,
                               int lv,
                               const struct tile_stream_layout* layout,
                               const uint32_t* chunk_lut,
                               const uint64_t* batch_chunk_offsets,
                               enum dtype dtype);

  // Dim0 fold: accumulate inner-reduced data (levels 1+) from the morton
  // buffer into the accumulator. On first call (counts[lv]==0) copies;
  // subsequent calls reduce (mean/min/max).
  // accum: buffer sized to sum(batch_count * lod_nelem[lv]) for lv=1..nlod-1.
  // counts[nlod]: per-level fold count (caller increments after this call).
  int lod_cpu_dim0_fold(const struct lod_plan* p,
                        const void* morton_values,
                        void* accum,
                        const uint32_t* counts,
                        enum dtype dtype,
                        enum lod_reduce_method method);

  // Dim0 emit: finalize accumulator for level lv back to morton buffer.
  // For float mean: divides by count. For int mean/min/max: copies.
  int lod_cpu_dim0_emit(const struct lod_plan* p,
                        void* morton_values,
                        const void* accum,
                        int lv,
                        uint32_t count,
                        enum dtype dtype,
                        enum lod_reduce_method method);

  // Build scatter LUT for L0: maps morton position to source linear offset
  // within one batch. lut must have room for lod_nelem[0] entries.
  // Computed once at init, reused every epoch.
  void lod_cpu_build_scatter_lut(const struct lod_plan* p, uint32_t* lut);

  // Build per-batch offsets into the linear source for scatter/gather.
  // offsets must have room for batch_count entries.
  void lod_cpu_build_scatter_batch_offsets(const struct lod_plan* p,
                                           uint64_t* offsets);

  // Gather linear input into morton-ordered LOD L0 buffer using
  // precomputed LUT. Sequential writes, random reads.
  // Replaces lod_cpu_scatter for the hot path.
  int lod_cpu_gather(const struct lod_plan* p,
                     const void* src,
                     void* dst,
                     const uint32_t* scatter_lut,
                     const uint64_t* batch_offsets,
                     enum dtype dtype);

#ifdef __cplusplus
}
#endif
