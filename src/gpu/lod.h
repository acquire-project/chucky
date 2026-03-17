#pragma once

#include "types.lod.h"
#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  int lod_fill_ends_gpu(CUdeviceptr d_ends,
                        int ndim,
                        CUdeviceptr d_child_shape,
                        CUdeviceptr d_parent_shape,
                        const uint64_t* child_shape_host,
                        const uint64_t* parent_shape_host,
                        uint64_t n_parents,
                        CUstream stream);

  // returns 0 on success, non-zero on failure
  int lod_reduce(CUdeviceptr d_values,
                 CUdeviceptr d_ends,
                 enum lod_dtype dtype,
                 enum lod_reduce_method method,
                 uint64_t src_offset,
                 uint64_t dst_offset,
                 uint64_t src_lod_count,
                 uint64_t dst_lod_count,
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
                               CUdeviceptr d_batch_chunk_offsets,
                               enum lod_dtype dtype,
                               uint64_t lod_count,
                               uint64_t batch_count,
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
                     CUdeviceptr d_batch_offsets,
                     enum lod_dtype dtype,
                     uint64_t lod_count,
                     uint64_t batch_count,
                     CUstream stream);

  int lod_accum_emit(CUdeviceptr d_dst,
                     CUdeviceptr d_accum,
                     enum lod_dtype dtype,
                     enum lod_reduce_method method,
                     uint64_t n_elements,
                     uint32_t count,
                     CUstream stream);

  int lod_accum_fold_fused(CUdeviceptr d_accum,
                           CUdeviceptr d_new_data,
                           CUdeviceptr d_level_ids,
                           CUdeviceptr d_counts,
                           enum lod_dtype dtype,
                           enum lod_reduce_method method,
                           uint64_t n_elements,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
