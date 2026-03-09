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

  enum lod_reduce_method
  {
    lod_reduce_mean,
    lod_reduce_min,
    lod_reduce_max,
    lod_reduce_median,
    lod_reduce_max_suppressed, // 2nd highest value
    lod_reduce_min_suppressed, // 2nd lowest value
  };

  // returns 1 on succes, 0 on failure
  int lod_scatter(CUdeviceptr d_dst,
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

  // returns 1 on succes, 0 on failure
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

  // Build a LUT mapping lod_linear_index -> morton_rank.
  // d_lut: device buffer of lod_count uint32_t entries.
  void lod_build_scatter_lut(CUdeviceptr d_lut,
                             CUdeviceptr d_lod_shape,
                             int lod_ndim,
                             const uint64_t* lod_shape_host,
                             uint64_t lod_count,
                             CUstream stream);

  // Scatter using a precomputed morton rank LUT.
  void lod_scatter_lut(CUdeviceptr d_dst,
                       CUdeviceptr d_src,
                       CUdeviceptr d_lut,
                       enum lod_dtype dtype,
                       int ndim,
                       uint64_t n_elements,
                       CUdeviceptr d_full_shape,
                       CUdeviceptr d_lod_shape,
                       int lod_ndim,
                       uint32_t lod_mask,
                       uint64_t lod_count,
                       CUstream stream);

  // Build a tile-scatter LUT: tile_lut[morton_pos] = tile_pool_offset.
  // The offset is the contribution from LOD dims only (using lifted strides).
  // d_fwd_lut: forward LUT (lod_linear -> morton_pos), lod_count entries.
  // d_lod_shape: lod_ndim uint64_t entries.
  // d_lod_tile_sizes: lod_ndim uint64_t entries (tile size per LOD dim).
  // d_lod_tile_strides: 2*lod_ndim int64_t entries (grid, within) pairs.
  // d_tile_lut: device buffer of lod_count uint32_t entries.
  void lod_build_tile_scatter_lut(CUdeviceptr d_tile_lut,
                                  CUdeviceptr d_fwd_lut,
                                  CUdeviceptr d_lod_shape,
                                  CUdeviceptr d_lod_tile_sizes,
                                  CUdeviceptr d_lod_tile_strides,
                                  int lod_ndim,
                                  uint64_t lod_count,
                                  CUstream stream);

  // Morton-to-tile scatter using precomputed LUTs.
  // d_tile_lut: lod_count uint32_t entries (morton_pos -> tile_pool_lod_offset).
  // d_batch_tile_offsets: batch_count uint32_t entries (batch -> tile_pool_batch_offset).
  void lod_morton_to_tiles_lut(CUdeviceptr d_tiles,
                               CUdeviceptr d_morton,
                               CUdeviceptr d_tile_lut,
                               CUdeviceptr d_batch_tile_offsets,
                               enum lod_dtype dtype,
                               uint64_t lod_count,
                               uint64_t batch_count,
                               CUstream stream);

  // Build an inverse LUT for gather: src_lut[morton_pos] = src_lod_offset.
  // The source offset is the C-order position contribution from LOD dims.
  // d_fwd_lut: forward LUT (lod_linear -> morton_pos), lod_count entries.
  // d_lod_strides: lod_ndim uint64_t entries, C-order strides of LOD dims
  //                in the full array.
  // d_src_lut: device buffer of lod_count uint32_t entries.
  // Caller must ensure all offsets fit in uint32_t.
  void lod_build_gather_lut(CUdeviceptr d_src_lut,
                            CUdeviceptr d_fwd_lut,
                            CUdeviceptr d_lod_shape,
                            CUdeviceptr d_lod_strides,
                            int lod_ndim,
                            uint64_t lod_count,
                            CUstream stream);

  // Gather using precomputed inverse LUT: iterate in Morton output order
  // for coalesced writes, scattered reads.
  // d_src_lut: lod_count uint32_t entries (morton_pos -> src_lod_offset).
  // d_batch_offsets: batch_count uint32_t entries mapping batch_index to
  //                  the C-order source offset from batch dims.
  // Caller must ensure all offsets fit in uint32_t.
  void lod_gather_lut(CUdeviceptr d_dst,
                      CUdeviceptr d_src,
                      CUdeviceptr d_src_lut,
                      CUdeviceptr d_batch_offsets,
                      enum lod_dtype dtype,
                      uint64_t lod_count,
                      uint64_t batch_count,
                      CUstream stream);

  // Fold one epoch's spatial LOD data into a running accumulator.
  // For mean: accumulator is wider type (u32 for u16), running sum.
  // For min/max: accumulator is native type, running min/max.
  // count: 0 = first epoch (initialize), >0 = fold.
  void lod_accum_fold(CUdeviceptr d_accum,
                      CUdeviceptr d_new_data,
                      enum lod_dtype dtype,
                      enum lod_reduce_method method,
                      uint64_t n_elements,
                      uint32_t count,
                      CUstream stream);

  // Finalize a mean accumulator: divide running sum by count, store in native
  // type.  d_dst and d_accum may alias when method is min/max (same type).
  // For mean: d_accum is wider (u32 for u16 input), d_dst is native type.
  void lod_accum_emit(CUdeviceptr d_dst,
                      CUdeviceptr d_accum,
                      enum lod_dtype dtype,
                      enum lod_reduce_method method,
                      uint64_t n_elements,
                      uint32_t count,
                      CUstream stream);

  // Fused fold over all LOD levels 1+ in a single kernel launch.
  // Each thread looks up its level from d_level_ids (u8) and reads the
  // corresponding count from d_counts to decide write vs fold.
  // d_accum and d_new_data have the same packed layout (levels 1+ only).
  void lod_accum_fold_fused(CUdeviceptr d_accum,
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
