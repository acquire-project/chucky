#pragma once

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum lod_dtype
  {
    lod_dtype_u16,
    lod_dtype_f32,
  };

  struct m2t_layout
  {
    enum lod_dtype dtype;
    int ndim;
    CUdeviceptr d_full_shape;
    int lod_ndim;
    uint32_t lod_mask;
    CUdeviceptr d_lod_shape;
    uint64_t lod_count;
    uint64_t n_elements;
    int lod_nlod;
    CUdeviceptr d_lifted_shape;
    CUdeviceptr d_lifted_strides;
  };

  void lod_scatter(CUdeviceptr d_dst,
                   CUdeviceptr d_src,
                   enum lod_dtype dtype,
                   int ndim,
                   uint64_t n_elements,
                   CUdeviceptr d_full_shape,
                   CUdeviceptr d_lod_shape,
                   int lod_ndim,
                   const uint64_t* lod_shape_host,
                   uint32_t lod_mask,
                   uint64_t lod_count,
                   CUstream stream);

  void lod_fill_ends_gpu(CUdeviceptr d_ends,
                         int ndim,
                         CUdeviceptr d_child_shape,
                         CUdeviceptr d_parent_shape,
                         const uint64_t* child_shape_host,
                         const uint64_t* parent_shape_host,
                         uint64_t n_parents,
                         CUstream stream);

  void lod_reduce(CUdeviceptr d_values,
                  CUdeviceptr d_ends,
                  enum lod_dtype dtype,
                  uint64_t src_offset,
                  uint64_t dst_offset,
                  uint64_t src_lod_count,
                  uint64_t dst_lod_count,
                  uint64_t batch_count,
                  CUstream stream);

  // Pre-compute lod_nlod (= ceil_log2(max(lod_shape))) for m2t_layout.
  int lod_m2t_lod_nlod(int lod_ndim, const uint64_t* lod_shape_host);

  // Morton-to-tile scatter: reads morton-ordered LOD data and writes into
  // tile-pool layout using lifted strides. See struct m2t_layout.
  void lod_morton_to_tiles(CUdeviceptr d_tiles,
                           CUdeviceptr d_morton,
                           const struct m2t_layout* layout,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
