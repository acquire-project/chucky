#pragma once

#include "dtype.h"
#include "types.lod.h"
#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // CSR-based reduce: precomputed starts/indices LUT.
  // d_values is a single flat allocation; src_offset/dst_offset are element
  // counts into it.
  int lod_reduce_csr(CUdeviceptr d_values,
                     CUdeviceptr d_starts,
                     CUdeviceptr d_indices,
                     enum dtype dtype,
                     enum lod_reduce_method method,
                     uint64_t src_offset,
                     uint64_t dst_offset,
                     uint64_t src_segment_size,
                     uint64_t dst_segment_size,
                     uint64_t batch_count,
                     CUstream stream);

  int lod_build_chunk_scatter_lut(CUdeviceptr d_chunk_lut,
                                  CUdeviceptr d_lod_shape,
                                  CUdeviceptr d_lod_chunk_sizes,
                                  CUdeviceptr d_lod_chunk_strides,
                                  int lod_ndim,
                                  const uint64_t* lod_shape_host,
                                  uint64_t lod_count,
                                  CUstream stream);

  int lod_morton_to_chunks_lut(CUdeviceptr d_chunks,
                               CUdeviceptr d_morton,
                               CUdeviceptr d_chunk_lut,
                               CUdeviceptr d_fixed_dims_chunk_offsets,
                               enum dtype dtype,
                               uint64_t lod_count,
                               uint64_t fixed_dims_count,
                               CUstream stream);

  int lod_build_gather_lut(CUdeviceptr d_src_lut,
                           CUdeviceptr d_lod_shape,
                           CUdeviceptr d_lod_strides,
                           int lod_ndim,
                           const uint64_t* lod_shape_host,
                           uint64_t lod_count,
                           CUstream stream);

  int lod_gather_lut(CUdeviceptr d_dst,
                     CUdeviceptr d_src,
                     CUdeviceptr d_src_lut,
                     CUdeviceptr d_fixed_dims_offsets,
                     enum dtype dtype,
                     uint64_t lod_count,
                     uint64_t fixed_dims_count,
                     CUstream stream);

  int lod_accum_emit(CUdeviceptr d_dst,
                     CUdeviceptr d_accum,
                     enum dtype dtype,
                     enum lod_reduce_method method,
                     uint64_t n_elements,
                     uint32_t count,
                     CUstream stream);

  int lod_accum_fold_fused(CUdeviceptr d_accum,
                           CUdeviceptr d_new_data,
                           CUdeviceptr d_level_ids,
                           CUdeviceptr d_counts,
                           enum dtype dtype,
                           enum lod_reduce_method method,
                           uint64_t n_elements,
                           CUstream stream);

  // Build CSR reduce LUT on GPU for one level transition.
  // d_starts: [dst_total+1] uint64, d_indices: [src_total] uint64.
  // Both must be pre-allocated by the caller.
  struct level_dims;
  int lod_build_csr_gpu(CUdeviceptr d_starts,
                        CUdeviceptr d_indices,
                        const struct level_dims* src,
                        const struct level_dims* dst,
                        CUstream stream);

#ifdef __cplusplus
}
#endif
