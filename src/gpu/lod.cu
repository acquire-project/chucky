#include "defs.limits.h"
#include "dtype.h"
#include "gpu/lod.h"
#include "lod/lod_plan.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

// min/max overloads for __half (not provided by CUDA math functions).
__device__ inline __half
min(__half a, __half b)
{
  return __hmin(a, b);
}
__device__ inline __half
max(__half a, __half b)
{
  return __hmax(a, b);
}

#define FOR_EACH_DTYPE(X)                                                      \
  X(dtype_u8, uint8_t)                                                         \
  X(dtype_u16, uint16_t)                                                       \
  X(dtype_u32, uint32_t)                                                       \
  X(dtype_u64, uint64_t)                                                       \
  X(dtype_i8, int8_t)                                                          \
  X(dtype_i16, int16_t)                                                        \
  X(dtype_i32, int32_t)                                                        \
  X(dtype_i64, int64_t)                                                        \
  X(dtype_f16, __half)                                                         \
  X(dtype_f32, float)                                                          \
  X(dtype_f64, double)

// Widened accumulator for lod_reduce (register-only).
template<typename T>
struct reduce_acc
{
  using type = T;
};
template<>
struct reduce_acc<uint8_t>
{
  using type = uint32_t;
};
template<>
struct reduce_acc<uint16_t>
{
  using type = uint32_t;
};
template<>
struct reduce_acc<uint32_t>
{
  using type = uint64_t;
};
template<>
struct reduce_acc<int8_t>
{
  using type = int32_t;
};
template<>
struct reduce_acc<int16_t>
{
  using type = int32_t;
};
template<>
struct reduce_acc<int32_t>
{
  using type = int64_t;
};
template<>
struct reduce_acc<__half>
{
  using type = float;
};
// u64, i64, float, double: default (type = T)

#define LOD_BLOCK 256

// Number of elements in a chunk of `chunk_size` starting at `start`
// within a dimension of `dimension_size`, clamped to the boundary.
__device__ static uint64_t
clamped_chunk_extent_d(uint64_t dimension_size,
                       uint64_t start,
                       uint64_t chunk_size)
{
  if (start >= dimension_size)
    return 0;
  uint64_t remaining = dimension_size - start;
  return (remaining < chunk_size) ? remaining : chunk_size;
}

// Index of the low child node for dimension d at the given tree level,
// derived from the coordinate without maintaining a prefix array.
//   level < nlod:  top bits of coord with LSB cleared
//   level >= nlod: coord shifted into the child grid
__device__ static uint64_t
tree_child_low(uint64_t coord, int level, int nlod)
{
  if (level < nlod) {
    int bit_index = nlod - 1 - level;
    return (coord >> (bit_index + 1)) << 1;
  }
  return coord << (level - nlod + 1);
}

// Morton rank from coordinates in register arrays.
//
// shape:   [ndim]  shape (read from registers / L1 cache)
// coords:  [ndim]  per-thread coordinates (registers)
// products: allocated on stack as [NdimMax+1] (registers)
template<int NdimMax>
__device__ static uint64_t
morton_rank_d(int ndim,
              const uint64_t* shape,
              int nlod,
              const uint64_t* coords,
              int depth)
{
  for (int d = 0; d < ndim; ++d) {
    uint64_t coord = coords[d];
    int coord_bits = coord > 0 ? (64 - __clzll(coord)) : 0;
    if (coord_bits > nlod)
      nlod = coord_bits;
  }

  int total_levels = nlod + depth;
  uint64_t rank = 0;
  uint64_t products[NdimMax + 1];

  for (int level = 0; level < total_levels; ++level) {
    uint64_t block_size = 1ull << (total_levels - 1 - level);

    int digit = 0;

    // Compute subtree-width products and extract the Morton digit.
    products[0] = 1;
    for (int d = 0; d < ndim; ++d) {
      uint64_t coord = coords[d];
      uint64_t node_low = tree_child_low(coord, level, nlod);
      if (level < nlod)
        digit |= (int)((coord >> (nlod - 1 - level)) & 1) << d;
      uint64_t extent_low =
        clamped_chunk_extent_d(shape[d], node_low * block_size, block_size);
      uint64_t extent_high = clamped_chunk_extent_d(
        shape[d], (node_low + 1) * block_size, block_size);
      products[d + 1] = products[d] * (extent_low + extent_high);
    }

    // For each set bit in digit, count elements in the low subtree.
    uint64_t suffix = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      uint64_t coord = coords[d];
      uint64_t node_low = tree_child_low(coord, level, nlod);
      int bit = (digit >> d) & 1;
      uint64_t extent_low =
        clamped_chunk_extent_d(shape[d], node_low * block_size, block_size);
      if (bit == 1)
        rank += suffix * extent_low * products[d];
      uint64_t extent_chosen = clamped_chunk_extent_d(
        shape[d], (node_low + (uint64_t)bit) * block_size, block_size);
      suffix *= extent_chosen;
    }
  }

  return rank;
}

// Morton rank without materializing a coordinate array.
// Each coord is derived on the fly: (linear / lin_stride[r]) % lin_shape[r]
// then right-shifted by `shift` (0 = identity, 1 = halve for LOD downsample).
// remap[d] maps morton dim d to the index into lin_stride/lin_shape.
// If remap is NULL, identity mapping is used.
template<int NdimMax>
__device__ static uint64_t
morton_rank_linear(int ndim,
                   const uint64_t* shape,
                   int nlod_init,
                   uint64_t linear,
                   const uint64_t* lin_stride,
                   const uint64_t* lin_shape,
                   const int* remap,
                   int shift)
{
  int nlod = nlod_init;
  for (int d = 0; d < ndim; ++d) {
    int si = remap ? remap[d] : d;
    uint64_t coord = ((linear / lin_stride[si]) % lin_shape[si]) >> shift;
    int coord_bits = coord > 0 ? (64 - __clzll(coord)) : 0;
    if (coord_bits > nlod)
      nlod = coord_bits;
  }

  int total_levels = nlod;
  uint64_t rank = 0;
  uint64_t products[NdimMax + 1];

  for (int level = 0; level < total_levels; ++level) {
    uint64_t block_size = 1ull << (total_levels - 1 - level);
    int digit = 0;

    products[0] = 1;
    for (int d = 0; d < ndim; ++d) {
      int si = remap ? remap[d] : d;
      uint64_t coord = ((linear / lin_stride[si]) % lin_shape[si]) >> shift;
      uint64_t node_low = tree_child_low(coord, level, nlod);
      if (level < nlod)
        digit |= (int)((coord >> (nlod - 1 - level)) & 1) << d;
      uint64_t extent_low =
        clamped_chunk_extent_d(shape[d], node_low * block_size, block_size);
      uint64_t extent_high = clamped_chunk_extent_d(
        shape[d], (node_low + 1) * block_size, block_size);
      products[d + 1] = products[d] * (extent_low + extent_high);
    }

    uint64_t suffix = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      int si = remap ? remap[d] : d;
      uint64_t coord = ((linear / lin_stride[si]) % lin_shape[si]) >> shift;
      uint64_t node_low = tree_child_low(coord, level, nlod);
      int bit = (digit >> d) & 1;
      uint64_t extent_low =
        clamped_chunk_extent_d(shape[d], node_low * block_size, block_size);
      if (bit == 1)
        rank += suffix * extent_low * products[d];
      uint64_t extent_chosen = clamped_chunk_extent_d(
        shape[d], (node_low + (uint64_t)bit) * block_size, block_size);
      suffix *= extent_chosen;
    }
  }

  return rank;
}

// --- Build gather LUT kernel ---
// Builds inv_lut: src_lut[morton_pos] = src_lod_offset, where
// src_lod_offset is the C-order contribution from LOD dimensions.

template<int NdimMax>
__global__ void
lod_build_gather_lut_k(uint32_t* __restrict__ src_lut,
                       int lod_ndim,
                       const uint64_t* __restrict__ lod_shape,
                       const uint64_t* __restrict__ lod_strides,
                       int lod_nlod,
                       uint64_t lod_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= lod_count)
    return;

  uint64_t coords[NdimMax];
  uint64_t remainder = gid;
  uint64_t src_offset = 0;
  for (int d = lod_ndim - 1; d >= 0; --d) {
    uint64_t coord = remainder % lod_shape[d];
    remainder /= lod_shape[d];
    coords[d] = coord;
    src_offset += coord * lod_strides[d];
  }

  uint64_t morton_pos =
    morton_rank_d<NdimMax>(lod_ndim, lod_shape, lod_nlod, coords, 0);
  src_lut[morton_pos] = (uint32_t)src_offset;
}

// --- Gather kernel: shared-memory tiled, u32-aliased, coalesced stores ---
// For types <= 4 bytes: pack into u32 via memcpy, coalesced stores via smem.
// For 8-byte types: simple per-element path (no packing).

template<typename T>
__global__ void __launch_bounds__(256, 4)
  lod_gather_lut_k(T* __restrict__ dst,
                   const T* __restrict__ src,
                   const uint32_t* __restrict__ src_lut,
                   const uint32_t* __restrict__ fixed_dims_offsets,
                   uint64_t lod_count,
                   uint64_t total)
{
  if constexpr (sizeof(T) > sizeof(uint32_t)) {
    // 8-byte types: simple per-element, no shared memory packing
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total)
      return;
    uint64_t batch = gid / lod_count;
    uint64_t morton_pos = gid % lod_count;
    dst[gid] = src[(uint64_t)fixed_dims_offsets[batch] + src_lut[morton_pos]];
  } else {
    constexpr int T_PER_U32 = sizeof(uint32_t) / sizeof(T);
    constexpr int TILE_U32 =
      (1 << 12) / sizeof(uint32_t); // 1024 u32 slots = 4KB
    constexpr int TILE_ELEMENTS = TILE_U32 * T_PER_U32;

    __shared__ uint32_t tile[TILE_U32];

    const int tid = threadIdx.x;
    const uint64_t block_base = (uint64_t)blockIdx.x * TILE_ELEMENTS;

    // Phase 1: Gather scattered reads, pack into u32 via memcpy, write to smem
    {
      uint64_t gid0 = block_base + (uint64_t)tid * T_PER_U32;
      uint64_t batch = gid0 / lod_count;
      uint64_t morton_pos = gid0 % lod_count;

      for (int i = tid; i < TILE_U32; i += blockDim.x) {
        uint64_t gid = block_base + (uint64_t)i * T_PER_U32;
        uint32_t packed = 0;

        for (int k = 0; k < T_PER_U32; ++k) {
          if (gid + k < total) {
            T val =
              src[(uint64_t)fixed_dims_offsets[batch] + src_lut[morton_pos]];
            memcpy((char*)&packed + k * sizeof(T), &val, sizeof(T));
          }
          morton_pos++;
          if (morton_pos >= lod_count) {
            morton_pos = 0;
            batch++;
          }
        }
        tile[i] = packed;

        // Advance to next iteration: skip (blockDim.x - 1) * T_PER_U32 elements
        morton_pos += (uint64_t)(blockDim.x - 1) * T_PER_U32;
        while (morton_pos >= lod_count) {
          morton_pos -= lod_count;
          batch++;
        }
      }
    }

    __syncthreads();

    // Phase 2: Coalesced u32 stores
    {
      uint32_t* dst_u32 = (uint32_t*)dst;
      uint64_t base_u32 = block_base / T_PER_U32;

      for (int i = tid; i < TILE_U32; i += blockDim.x) {
        uint64_t elem_idx = block_base + (uint64_t)i * T_PER_U32;
        if (elem_idx + T_PER_U32 <= total) {
          dst_u32[base_u32 + i] = tile[i];
        } else if (elem_idx < total) {
          T* tile_T = (T*)tile;
          for (int k = 0; k < T_PER_U32; ++k)
            if (elem_idx + k < total)
              dst[elem_idx + k] = tile_T[i * T_PER_U32 + k];
        }
      }
    }
  }
}

// --- Build chunk-scatter LUT kernel ---
// Builds chunk_lut: chunk_lut[morton_pos] = chunk_pool_offset from LOD dims.
// Uses forward LUT to map lod_linear -> morton_pos, then decomposes
// lod_linear into coords and computes lifted-stride offset.

template<int NdimMax>
__global__ void
lod_build_chunk_scatter_lut_k(uint32_t* __restrict__ chunk_lut,
                              int lod_ndim,
                              const uint64_t* __restrict__ lod_shape,
                              const uint64_t* __restrict__ lod_chunk_sizes,
                              const int64_t* __restrict__ lod_chunk_strides,
                              int lod_nlod,
                              uint64_t lod_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= lod_count)
    return;

  uint64_t coords[NdimMax];
  uint64_t remainder = gid;
  int64_t offset = 0;
  for (int d = lod_ndim - 1; d >= 0; --d) {
    uint64_t coord = remainder % lod_shape[d];
    remainder /= lod_shape[d];
    coords[d] = coord;
    uint64_t chunk_idx = coord / lod_chunk_sizes[d];
    uint64_t within = coord % lod_chunk_sizes[d];
    offset += (int64_t)chunk_idx * lod_chunk_strides[2 * d];
    offset += (int64_t)within * lod_chunk_strides[2 * d + 1];
  }

  uint64_t morton_pos =
    morton_rank_d<NdimMax>(lod_ndim, lod_shape, lod_nlod, coords, 0);
  chunk_lut[morton_pos] = (uint32_t)offset;
}

// --- Morton-to-chunk scatter kernel using LUT ---
// Sequential reads from morton buffer, LUT-directed writes to chunk pool.

template<typename T>
__global__ void
lod_morton_to_chunks_lut_k(
  T* __restrict__ dst,
  const T* __restrict__ src,
  const uint32_t* __restrict__ chunk_lut,
  const uint32_t* __restrict__ fixed_dims_chunk_offsets,
  uint64_t lod_count,
  uint64_t total)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= total)
    return;

  uint64_t batch = gid / lod_count;
  uint64_t morton_pos = gid % lod_count;
  dst[(uint64_t)fixed_dims_chunk_offsets[batch] + chunk_lut[morton_pos]] =
    src[gid];
}

// --- Shared memory size helpers ---

// --- Dispatch helpers ---

template<int NdimMax>
static void
lod_build_gather_lut_launch(CUdeviceptr d_src_lut,
                            CUdeviceptr d_lod_shape,
                            CUdeviceptr d_lod_strides,
                            int lod_ndim,
                            int lod_nlod,
                            uint64_t lod_count,
                            CUstream stream)
{
  const int grid_size = (int)((lod_count + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_build_gather_lut_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint32_t*)d_src_lut,
                                          lod_ndim,
                                          (const uint64_t*)d_lod_shape,
                                          (const uint64_t*)d_lod_strides,
                                          lod_nlod,
                                          lod_count);
}

extern "C" int
lod_build_gather_lut(CUdeviceptr d_src_lut,
                     CUdeviceptr d_lod_shape,
                     CUdeviceptr d_lod_strides,
                     int lod_ndim,
                     const uint64_t* lod_shape_host,
                     uint64_t lod_count,
                     CUstream stream)
{
  int lod_nlod = ceil_log2(max_shape(lod_ndim, lod_shape_host));

#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_gather_lut_launch<maxdim>(d_src_lut,                             \
                                        d_lod_shape,                           \
                                        d_lod_strides,                         \
                                        lod_ndim,                              \
                                        lod_nlod,                              \
                                        lod_count,                             \
                                        stream);                               \
    return 0;                                                                  \
  }

  XXX(4);
  XXX(8);
  XXX(16);
  XXX(32);
#undef XXX
  return 1;
}

template<int NdimMax>
static void
lod_build_chunk_scatter_lut_launch(CUdeviceptr d_chunk_lut,
                                   CUdeviceptr d_lod_shape,
                                   CUdeviceptr d_lod_chunk_sizes,
                                   CUdeviceptr d_lod_chunk_strides,
                                   int lod_ndim,
                                   int lod_nlod,
                                   uint64_t lod_count,
                                   CUstream stream)
{
  const int grid_size = (int)((lod_count + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_build_chunk_scatter_lut_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint32_t*)d_chunk_lut,
                                          lod_ndim,
                                          (const uint64_t*)d_lod_shape,
                                          (const uint64_t*)d_lod_chunk_sizes,
                                          (const int64_t*)d_lod_chunk_strides,
                                          lod_nlod,
                                          lod_count);
}

extern "C" int
lod_build_chunk_scatter_lut(CUdeviceptr d_chunk_lut,
                            CUdeviceptr d_lod_shape,
                            CUdeviceptr d_lod_chunk_sizes,
                            CUdeviceptr d_lod_chunk_strides,
                            int lod_ndim,
                            const uint64_t* lod_shape_host,
                            uint64_t lod_count,
                            CUstream stream)
{
  int lod_nlod = ceil_log2(max_shape(lod_ndim, lod_shape_host));

#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_chunk_scatter_lut_launch<maxdim>(d_chunk_lut,                    \
                                               d_lod_shape,                    \
                                               d_lod_chunk_sizes,              \
                                               d_lod_chunk_strides,            \
                                               lod_ndim,                       \
                                               lod_nlod,                       \
                                               lod_count,                      \
                                               stream);                        \
    return 0;                                                                  \
  }

  XXX(4);
  XXX(8);
#undef XXX
  return 1;
}

template<typename T>
static void
lod_morton_to_chunks_lut_launch(CUdeviceptr d_chunks,
                                CUdeviceptr d_morton,
                                CUdeviceptr d_chunk_lut,
                                CUdeviceptr d_fixed_dims_chunk_offsets,
                                uint64_t lod_count,
                                uint64_t fixed_dims_count,
                                CUstream stream)
{
  const uint64_t total = fixed_dims_count * lod_count;
  const int grid_size = (int)((total + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_morton_to_chunks_lut_k<T><<<grid_size, LOD_BLOCK, 0, stream>>>(
    (T*)d_chunks,
    (const T*)d_morton,
    (const uint32_t*)d_chunk_lut,
    (const uint32_t*)d_fixed_dims_chunk_offsets,
    lod_count,
    total);
}

extern "C" int
lod_morton_to_chunks_lut(CUdeviceptr d_chunks,
                         CUdeviceptr d_morton,
                         CUdeviceptr d_chunk_lut,
                         CUdeviceptr d_fixed_dims_chunk_offsets,
                         enum dtype dtype,
                         uint64_t lod_count,
                         uint64_t fixed_dims_count,
                         CUstream stream)
{
#define DISPATCH(D, T)                                                         \
  if (dtype == D) {                                                            \
    lod_morton_to_chunks_lut_launch<T>(d_chunks,                               \
                                       d_morton,                               \
                                       d_chunk_lut,                            \
                                       d_fixed_dims_chunk_offsets,             \
                                       lod_count,                              \
                                       fixed_dims_count,                       \
                                       stream);                                \
    return 0;                                                                  \
  }
  FOR_EACH_DTYPE(DISPATCH)
#undef DISPATCH
  return 1;
}

template<typename T>
static void
lod_gather_lut_launch(CUdeviceptr d_dst,
                      CUdeviceptr d_src,
                      CUdeviceptr d_src_lut,
                      CUdeviceptr d_fixed_dims_offsets,
                      uint64_t lod_count,
                      uint64_t fixed_dims_count,
                      CUstream stream)
{
  const uint64_t total = fixed_dims_count * lod_count;
  int grid_size;

  if constexpr (sizeof(T) > sizeof(uint32_t)) {
    grid_size = (int)((total + LOD_BLOCK - 1) / LOD_BLOCK);
  } else {
    constexpr int T_PER_U32 = sizeof(uint32_t) / sizeof(T);
    constexpr int TILE_ELEMENTS =
      ((1 << 12) / (int)sizeof(uint32_t)) * T_PER_U32;
    grid_size = (int)((total + TILE_ELEMENTS - 1) / TILE_ELEMENTS);
  }

  lod_gather_lut_k<T>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((T*)d_dst,
                                          (const T*)d_src,
                                          (const uint32_t*)d_src_lut,
                                          (const uint32_t*)d_fixed_dims_offsets,
                                          lod_count,
                                          total);
}

extern "C" int
lod_gather_lut(CUdeviceptr d_dst,
               CUdeviceptr d_src,
               CUdeviceptr d_src_lut,
               CUdeviceptr d_fixed_dims_offsets,
               enum dtype dtype,
               uint64_t lod_count,
               uint64_t fixed_dims_count,
               CUstream stream)
{
#define DISPATCH(D, T)                                                         \
  if (dtype == D) {                                                            \
    lod_gather_lut_launch<T>(d_dst,                                            \
                             d_src,                                            \
                             d_src_lut,                                        \
                             d_fixed_dims_offsets,                             \
                             lod_count,                                        \
                             fixed_dims_count,                                 \
                             stream);                                          \
    return 0;                                                                  \
  }
  FOR_EACH_DTYPE(DISPATCH)
#undef DISPATCH
  return 1;
}

// Type trait: is this a floating-point type (including __half)?
template<typename T>
struct is_fp
{
  static constexpr bool value = false;
};
template<>
struct is_fp<__half>
{
  static constexpr bool value = true;
};
template<>
struct is_fp<float>
{
  static constexpr bool value = true;
};
template<>
struct is_fp<double>
{
  static constexpr bool value = true;
};

// --- Accumulator emit kernel (dim0 LOD) ---
// Finalizes accumulator to native type.
// For integer mean: accumulator already holds pre-divided result; just copy.
// For float mean: accumulator holds raw sum; divide here.

template<typename T, enum lod_reduce_method Method>
__global__ void
lod_accum_emit_k(T* __restrict__ dst,
                 const T* __restrict__ accum,
                 uint64_t n_elements,
                 uint32_t count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= n_elements)
    return;

  // For integer mean: accumulator already holds sum>>level (pre-divided
  // via overflow_safe_add_shift in fold). Just copy.
  // For float mean: accumulator holds raw sum; divide here.
  if constexpr (Method == lod_reduce_mean && is_fp<T>::value)
    dst[gid] = (T)(accum[gid] / (T)count);
  else
    dst[gid] = accum[gid];
}

extern "C" int
lod_accum_emit(CUdeviceptr d_dst,
               CUdeviceptr d_accum,
               enum dtype dtype,
               enum lod_reduce_method method,
               uint64_t n_elements,
               uint32_t count,
               CUstream stream)
{
  const int grid = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);

#define LAUNCH_EMIT(T, M)                                                      \
  lod_accum_emit_k<T, M><<<grid, LOD_BLOCK, 0, stream>>>(                      \
    (T*)d_dst, (const T*)d_accum, n_elements, count)

#define EMIT_METHOD(T)                                                         \
  switch (method) {                                                            \
    case lod_reduce_mean:                                                      \
      LAUNCH_EMIT(T, lod_reduce_mean);                                         \
      return 0;                                                                \
    case lod_reduce_min:                                                       \
      LAUNCH_EMIT(T, lod_reduce_min);                                          \
      return 0;                                                                \
    case lod_reduce_max:                                                       \
      LAUNCH_EMIT(T, lod_reduce_max);                                          \
      return 0;                                                                \
    default:                                                                   \
      return 1;                                                                \
  }

#define DISPATCH(D, T)                                                         \
  case D:                                                                      \
    EMIT_METHOD(T);                                                            \
    break;
  switch (dtype) {
    FOR_EACH_DTYPE(DISPATCH)
  }
#undef DISPATCH
#undef EMIT_METHOD
#undef LAUNCH_EMIT
  return 1;
}

// --- Fused accumulator fold kernel (dim0 LOD) ---
// Single launch over all LOD levels 1+. Each thread reads its level from
// d_level_ids and the corresponding count from d_counts.
//
// Mean accumulation uses an overflow-safe running sum. The final count
// is always 2^level, so the division is a right-shift by `level` bits.
// To avoid overflow in native-width accumulators we pre-divide each
// addend: (a + b) >> s  =  (a>>s + b>>s) + ((a&mask + b&mask) >> s).
// The accumulator stores the partial mean (sum >> s) so emit just copies.

// Overflow-safe (a + b) >> s for integer types.
// Float types: just add (emit divides later).
// 64-bit integers: just add (overflow unlikely for LOD window sizes).
template<typename T>
__device__ static T
overflow_safe_add_shift(T a, T b, int s)
{
  if constexpr (is_fp<T>::value || sizeof(T) >= 8) {
    return a + b;
  } else {
    T mask = (T)((1u << s) - 1);
    return (T)((a >> s) + (b >> s) + (((a & mask) + (b & mask)) >> s));
  }
}

template<typename T, enum lod_reduce_method Method>
__global__ void
lod_accum_fold_fused_k(T* __restrict__ accum,
                       const T* __restrict__ new_data,
                       const uint8_t* __restrict__ level_ids,
                       const uint32_t* __restrict__ counts,
                       uint64_t n_elements)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= n_elements)
    return;

  uint32_t count = counts[level_ids[gid]];
  T val = new_data[gid];

  if (count == 0) {
    accum[gid] = val;
  } else {
    if constexpr (Method == lod_reduce_mean) {
      int s = (int)level_ids[gid]; // shift = level = log2(period)
      accum[gid] = overflow_safe_add_shift<T>(accum[gid], val, s);
    } else if constexpr (Method == lod_reduce_min)
      accum[gid] = min(accum[gid], val);
    else if constexpr (Method == lod_reduce_max)
      accum[gid] = max(accum[gid], val);
  }
}

extern "C" int
lod_accum_fold_fused(CUdeviceptr d_accum,
                     CUdeviceptr d_new_data,
                     CUdeviceptr d_level_ids,
                     CUdeviceptr d_counts,
                     enum dtype dtype,
                     enum lod_reduce_method method,
                     uint64_t n_elements,
                     CUstream stream)
{
  const int grid = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);

#define LAUNCH_FUSED(T, M)                                                     \
  lod_accum_fold_fused_k<T, M>                                                 \
    <<<grid, LOD_BLOCK, 0, stream>>>((T*)d_accum,                              \
                                     (const T*)d_new_data,                     \
                                     (const uint8_t*)d_level_ids,              \
                                     (const uint32_t*)d_counts,                \
                                     n_elements)

#define FUSED_METHOD(T)                                                        \
  switch (method) {                                                            \
    case lod_reduce_mean:                                                      \
      LAUNCH_FUSED(T, lod_reduce_mean);                                        \
      return 0;                                                                \
    case lod_reduce_min:                                                       \
      LAUNCH_FUSED(T, lod_reduce_min);                                         \
      return 0;                                                                \
    case lod_reduce_max:                                                       \
      LAUNCH_FUSED(T, lod_reduce_max);                                         \
      return 0;                                                                \
    default:                                                                   \
      return 1;                                                                \
  }

#define DISPATCH(D, T)                                                         \
  case D:                                                                      \
    FUSED_METHOD(T);                                                           \
    break;
  switch (dtype) {
    FOR_EACH_DTYPE(DISPATCH)
  }
#undef DISPATCH
#undef FUSED_METHOD
#undef LAUNCH_FUSED
  return 1;
}

// --- GPU CSR builder ---
// Build CSR starts/indices arrays on GPU for one level transition.
// Mirrors the CPU build_reduce_csr logic in lod_plan.c but avoids
// per-thread coordinate arrays by deriving each coord from a linear
// index on the fly via div/mod.

#include <cub/device/device_scan.cuh>

struct csr_level_params
{
  int src_lod_ndim;
  uint64_t src_lod_shape[LOD_MAX_NDIM];
  uint64_t
    src_lod_stride[LOD_MAX_NDIM]; // stride[k] = prod(src_lod_shape[0..k-1])
  uint64_t src_lod_nelem;
  int src_nlod; // max ceil_log2(src_lod_shape[k])

  int src_fixed_ndim;
  uint64_t src_fixed_shape[LOD_MAX_NDIM];
  uint64_t src_fixed_stride[LOD_MAX_NDIM]; // stride[k] =
                                           // prod(src_fixed_shape[k+1..n-1])
  uint64_t src_fixed_count;

  int dst_lod_ndim;
  uint64_t dst_lod_shape[LOD_MAX_NDIM];
  uint64_t dst_lod_nelem;
  uint64_t dst_fixed_count;
  int dst_nlod; // max ceil_log2(dst_lod_shape[k])

  int dst_fixed_ndim;
  uint64_t dst_fixed_shape[LOD_MAX_NDIM];

  // Per dst fixed dim: where the coord comes from.
  //   is_lod[k]=0 → src fixed dim at src_fixed_src_index[k]
  //   is_lod[k]=1 → dropped LOD dim at src_lod index dst_fixed_src_index[k]
  int dst_fixed_src_is_lod[LOD_MAX_NDIM];
  int dst_fixed_src_index[LOD_MAX_NDIM];

  // Per dst LOD dim: which src LOD dim provides the coord (halved).
  int dst_lod_src_index[LOD_MAX_NDIM];
};

// Map each source element to its destination element and histogram.
// No per-thread coordinate arrays — all coords derived from scalars.
template<int NdimMax>
__global__ void
csr_map_k(uint64_t* __restrict__ d_map,
          uint64_t* __restrict__ d_counts,
          const csr_level_params* __restrict__ pp,
          uint64_t src_total)
{
  const uint64_t gi = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gi >= src_total)
    return;

  const csr_level_params& p = *pp;
  uint64_t src_batch = gi / p.src_lod_nelem;
  uint64_t src_enum = gi % p.src_lod_nelem;

  // dst_bi: ravel dst fixed coords without storing a coord vector.
  uint64_t dst_bi = 0;
  for (int k = 0; k < p.dst_fixed_ndim; ++k) {
    uint64_t coord;
    if (p.dst_fixed_src_is_lod[k]) {
      int si = p.dst_fixed_src_index[k];
      coord = ((src_enum / p.src_lod_stride[si]) % p.src_lod_shape[si]) / 2;
    } else {
      int si = p.dst_fixed_src_index[k];
      coord = (src_batch / p.src_fixed_stride[si]) % p.src_fixed_shape[si];
    }
    dst_bi = dst_bi * p.dst_fixed_shape[k] + coord;
  }

  // dst_morton: morton rank of halved non-dropped LOD coords.
  uint64_t dst_morton = 0;
  if (p.dst_lod_ndim > 0)
    dst_morton = morton_rank_linear<NdimMax>(p.dst_lod_ndim,
                                             p.dst_lod_shape,
                                             p.dst_nlod,
                                             src_enum,
                                             p.src_lod_stride,
                                             p.src_lod_shape,
                                             p.dst_lod_src_index,
                                             1);

  uint64_t dst_elem = dst_bi * p.dst_lod_nelem + dst_morton;

  // src_morton: morton rank of src LOD coords (no remap, no shift).
  uint64_t src_morton = morton_rank_linear<NdimMax>(p.src_lod_ndim,
                                                    p.src_lod_shape,
                                                    p.src_nlod,
                                                    src_enum,
                                                    p.src_lod_stride,
                                                    p.src_lod_shape,
                                                    nullptr,
                                                    0);
  uint64_t src_elem = src_batch * p.src_lod_nelem + src_morton;

  d_map[gi] = dst_elem;
  d_map[src_total + gi] = src_elem;

  atomicAdd((unsigned long long*)&d_counts[dst_elem], 1ull);
}

// Scatter source indices into CSR indices array.
// d_write_pos is initialized as a copy of d_starts, so atomic returns an
// absolute position into d_indices — no additional offset needed.
__global__ void
csr_scatter_k(uint64_t* __restrict__ d_indices,
              uint64_t* __restrict__ d_write_pos,
              const uint64_t* __restrict__ d_map,
              uint64_t src_total)
{
  const uint64_t gi = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gi >= src_total)
    return;

  uint64_t dst_elem = d_map[gi];
  uint64_t src_elem = d_map[src_total + gi];
  uint64_t pos = atomicAdd((unsigned long long*)&d_write_pos[dst_elem], 1ull);
  d_indices[pos] = src_elem;
}

template<int NdimMax>
static int
lod_build_csr_gpu_launch(CUdeviceptr d_starts,
                         CUdeviceptr d_indices,
                         const csr_level_params* p,
                         CUstream stream)
{
  uint64_t src_total = p->src_fixed_count * p->src_lod_nelem;
  uint64_t dst_total = p->dst_fixed_count * p->dst_lod_nelem;

  if (src_total == 0 || dst_total == 0) {
    CUresult r =
      cuMemsetD8Async(d_starts, 0, (dst_total + 1) * sizeof(uint64_t), stream);
    if (r != CUDA_SUCCESS)
      return 1;
    return 0;
  }

  uint64_t grid64 = (src_total + LOD_BLOCK - 1) / LOD_BLOCK;
  if (grid64 > (uint64_t)INT_MAX) {
    log_error("csr: src_total %llu exceeds grid limit",
              (unsigned long long)src_total);
    return 1;
  }
  if (dst_total > (uint64_t)INT_MAX) {
    log_error("csr: dst_total %llu exceeds CUB int limit",
              (unsigned long long)dst_total);
    return 1;
  }
  int grid = (int)grid64;

  CUdeviceptr d_map = 0, d_counts = 0, d_write_pos = 0;
  CUdeviceptr d_cub_temp = 0, d_params = 0;
  int ret = 1;

  CUresult r;
  r = cuMemAlloc(&d_params, sizeof(csr_level_params));
  if (r != CUDA_SUCCESS)
    goto cleanup;
  r = cuMemcpyHtoD(d_params, p, sizeof(csr_level_params));
  if (r != CUDA_SUCCESS)
    goto cleanup;
  r = cuMemAlloc(&d_map, 2 * src_total * sizeof(uint64_t));
  if (r != CUDA_SUCCESS)
    goto cleanup;
  r = cuMemAlloc(&d_counts, dst_total * sizeof(uint64_t));
  if (r != CUDA_SUCCESS)
    goto cleanup;
  r = cuMemsetD8Async(d_counts, 0, dst_total * sizeof(uint64_t), stream);
  if (r != CUDA_SUCCESS)
    goto cleanup;

  // Step 1+2: Map + histogram.
  csr_map_k<NdimMax>
    <<<grid, LOD_BLOCK, 0, stream>>>((uint64_t*)d_map,
                                     (uint64_t*)d_counts,
                                     (const csr_level_params*)d_params,
                                     src_total);

  // Step 3: Prefix sum on counts → starts.
  {
    size_t cub_temp_bytes = 0;
    cudaError_t ce = cub::DeviceScan::ExclusiveSum(nullptr,
                                                   cub_temp_bytes,
                                                   (uint64_t*)d_counts,
                                                   (uint64_t*)d_starts,
                                                   (int)dst_total,
                                                   stream);
    if (ce != cudaSuccess) {
      log_error("CUB ExclusiveSum query failed: %d", (int)ce);
      goto cleanup;
    }
    if (cub_temp_bytes > 0) {
      r = cuMemAlloc(&d_cub_temp, cub_temp_bytes);
      if (r != CUDA_SUCCESS)
        goto cleanup;
    }
    ce = cub::DeviceScan::ExclusiveSum((void*)d_cub_temp,
                                       cub_temp_bytes,
                                       (uint64_t*)d_counts,
                                       (uint64_t*)d_starts,
                                       (int)dst_total,
                                       stream);
    if (ce != cudaSuccess) {
      log_error("CUB ExclusiveSum exec failed: %d", (int)ce);
      goto cleanup;
    }
  }

  // Sentinel: starts[dst_total] = src_total.  Writes a non-overlapping
  // element past the prefix-sum output, so no sync needed before this.
  // Synchronous copy — 8 bytes, safe from stack, negligible at init time.
  r = cuMemcpyHtoD(
    d_starts + dst_total * sizeof(uint64_t), &src_total, sizeof(uint64_t));
  if (r != CUDA_SUCCESS)
    goto cleanup;

  // Step 4: Scatter into indices.
  r = cuMemAlloc(&d_write_pos, dst_total * sizeof(uint64_t));
  if (r != CUDA_SUCCESS)
    goto cleanup;
  r = cuMemcpyDtoDAsync(
    d_write_pos, d_starts, dst_total * sizeof(uint64_t), stream);
  if (r != CUDA_SUCCESS)
    goto cleanup;

  csr_scatter_k<<<grid, LOD_BLOCK, 0, stream>>>((uint64_t*)d_indices,
                                                (uint64_t*)d_write_pos,
                                                (const uint64_t*)d_map,
                                                src_total);

  ret = 0;

cleanup:
  if (d_map)
    cuMemFree(d_map);
  if (d_counts)
    cuMemFree(d_counts);
  if (d_write_pos)
    cuMemFree(d_write_pos);
  if (d_cub_temp)
    cuMemFree(d_cub_temp);
  if (d_params)
    cuMemFree(d_params);
  return ret;
}

// Helper: find index of `d` in an int array. Returns -1 if not found.
static int
find_dim_index(const int* arr, int n, int d)
{
  for (int k = 0; k < n; ++k)
    if (arr[k] == d)
      return k;
  return -1;
}

static int
ceil_log2_h(uint64_t v)
{
  if (v <= 1)
    return 0;
#ifdef _MSC_VER
  unsigned long idx;
  _BitScanReverse64(&idx, v - 1);
  return (int)(idx + 1);
#else
  return 64 - __builtin_clzll(v - 1);
#endif
}

extern "C" int
lod_build_csr_gpu(CUdeviceptr d_starts,
                  CUdeviceptr d_indices,
                  const struct level_dims* src,
                  const struct level_dims* dst,
                  CUstream stream)
{
  uint32_t dropped_mask = src->lod_mask & ~dst->lod_mask;

  csr_level_params p;
  memset(&p, 0, sizeof(p));

  // Source LOD dims: shape, strides, nlod.
  p.src_lod_ndim = src->lod_ndim;
  p.src_lod_nelem = src->lod_nelem;
  p.src_nlod = 0;
  {
    uint64_t stride = 1;
    for (int k = 0; k < src->lod_ndim; ++k) {
      uint64_t s = src->dim[src->lod_to_dim[k]].size;
      p.src_lod_shape[k] = s;
      p.src_lod_stride[k] = stride;
      stride *= s;
      int bits = ceil_log2_h(s);
      if (bits > p.src_nlod)
        p.src_nlod = bits;
    }
  }

  // Source fixed dims: shape, strides (row-major, highest stride first).
  p.src_fixed_ndim = src->fixed_dims_ndim;
  p.src_fixed_count = src->fixed_dims_count;
  {
    uint64_t stride = 1;
    for (int k = src->fixed_dims_ndim - 1; k >= 0; --k) {
      p.src_fixed_shape[k] = src->fixed_dims_shape[k];
      p.src_fixed_stride[k] = stride;
      stride *= src->fixed_dims_shape[k];
    }
  }

  // Destination LOD dims: shape, nlod, src index mapping.
  p.dst_lod_ndim = dst->lod_ndim;
  p.dst_lod_nelem = dst->lod_nelem;
  p.dst_fixed_count = dst->fixed_dims_count;
  p.dst_nlod = 0;
  for (int k = 0; k < dst->lod_ndim; ++k) {
    uint64_t s = dst->dim[dst->lod_to_dim[k]].size;
    p.dst_lod_shape[k] = s;
    int bits = ceil_log2_h(s);
    if (bits > p.dst_nlod)
      p.dst_nlod = bits;
    int d = dst->lod_to_dim[k];
    int si = find_dim_index(src->lod_to_dim, src->lod_ndim, d);
    if (si < 0) {
      log_error("dst lod dim %d not found in src lod", d);
      return 1;
    }
    p.dst_lod_src_index[k] = si;
  }

  // Destination fixed dims: shape + source lookup.
  p.dst_fixed_ndim = dst->fixed_dims_ndim;
  for (int k = 0; k < dst->fixed_dims_ndim; ++k) {
    p.dst_fixed_shape[k] = dst->fixed_dims_shape[k];
    int d = dst->fixed_dim_to_dim[k];
    if (dropped_mask & (1u << d)) {
      p.dst_fixed_src_is_lod[k] = 1;
      int si = find_dim_index(src->lod_to_dim, src->lod_ndim, d);
      if (si < 0) {
        log_error("dropped dim %d not found in src lod", d);
        return 1;
      }
      p.dst_fixed_src_index[k] = si;
    } else {
      p.dst_fixed_src_is_lod[k] = 0;
      int si = find_dim_index(src->fixed_dim_to_dim, src->fixed_dims_ndim, d);
      if (si < 0) {
        log_error("fixed dim %d not found in src fixed", d);
        return 1;
      }
      p.dst_fixed_src_index[k] = si;
    }
  }

  int max_ndim = src->lod_ndim;
  if (dst->lod_ndim > max_ndim)
    max_ndim = dst->lod_ndim;

#define DISPATCH_CSR(maxdim)                                                   \
  if (max_ndim <= maxdim)                                                      \
    return lod_build_csr_gpu_launch<maxdim>(d_starts, d_indices, &p, stream);

  DISPATCH_CSR(4);
  DISPATCH_CSR(8);
  DISPATCH_CSR(16);
  DISPATCH_CSR(32);
#undef DISPATCH_CSR
  return 1;
}

// --- CSR reduce kernel ---

template<typename T, typename Acc, enum lod_reduce_method Method>
__global__ void
lod_reduce_csr_k(T* __restrict__ values,
                 const uint64_t* __restrict__ starts,
                 const uint64_t* __restrict__ indices,
                 uint64_t src_offset,
                 uint64_t dst_offset,
                 uint64_t src_segment_size,
                 uint64_t dst_segment_size,
                 uint64_t total)
{
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total)
    return;

  const uint64_t batch = gid / dst_segment_size;
  const uint64_t element = gid % dst_segment_size;

  const uint64_t src_base = src_offset + batch * src_segment_size;
  const uint64_t dst_base = dst_offset + batch * dst_segment_size;

  uint64_t start = starts[element];
  uint64_t end = starts[element + 1];
  if (start >= end) {
    values[dst_base + element] = (T)0;
    return;
  }
#pragma nv_diag_suppress 177, 550
  uint64_t len = end - start;
#pragma nv_diag_default 177, 550

  T result;

  if constexpr (Method == lod_reduce_mean) {
    Acc sum = (Acc)0;
    for (uint64_t j = start; j < end; ++j)
      sum += (Acc)values[src_base + indices[j]];
    result = (T)(sum / (Acc)len);
  } else if constexpr (Method == lod_reduce_min) {
    T best = values[src_base + indices[start]];
    for (uint64_t j = start + 1; j < end; ++j) {
      T v = values[src_base + indices[j]];
      if (v < best)
        best = v;
    }
    result = best;
  } else if constexpr (Method == lod_reduce_max) {
    T best = values[src_base + indices[start]];
    for (uint64_t j = start + 1; j < end; ++j) {
      T v = values[src_base + indices[j]];
      if (v > best)
        best = v;
    }
    result = best;
  } else if constexpr (Method == lod_reduce_median) {
    T buf[16];
    uint64_t n = (len <= 16) ? len : 16;
    for (uint64_t j = 0; j < n; ++j)
      buf[j] = values[src_base + indices[start + j]];
    for (uint64_t i = 1; i < n; ++i) {
      T key = buf[i];
      uint64_t k = i;
      while (k > 0 && buf[k - 1] > key) {
        buf[k] = buf[k - 1];
        --k;
      }
      buf[k] = key;
    }
    result = buf[n / 2];
  } else if constexpr (Method == lod_reduce_max_suppressed) {
    T top1 = values[src_base + indices[start]];
    T top2 = top1;
    if (len > 1) {
      T v = values[src_base + indices[start + 1]];
      if (v >= top1) {
        top2 = top1;
        top1 = v;
      } else {
        top2 = v;
      }
      for (uint64_t j = start + 2; j < end; ++j) {
        v = values[src_base + indices[j]];
        if (v >= top1) {
          top2 = top1;
          top1 = v;
        } else if (v > top2)
          top2 = v;
      }
    }
    result = top2;
  } else if constexpr (Method == lod_reduce_min_suppressed) {
    T bot1 = values[src_base + indices[start]];
    T bot2 = bot1;
    if (len > 1) {
      T v = values[src_base + indices[start + 1]];
      if (v <= bot1) {
        bot2 = bot1;
        bot1 = v;
      } else {
        bot2 = v;
      }
      for (uint64_t j = start + 2; j < end; ++j) {
        v = values[src_base + indices[j]];
        if (v <= bot1) {
          bot2 = bot1;
          bot1 = v;
        } else if (v < bot2)
          bot2 = v;
      }
    }
    result = bot2;
  }

  values[dst_base + element] = result;
}

extern "C" int
lod_reduce_csr(CUdeviceptr d_values,
               CUdeviceptr d_starts,
               CUdeviceptr d_indices,
               enum dtype dtype,
               enum lod_reduce_method method,
               uint64_t src_offset,
               uint64_t dst_offset,
               uint64_t src_segment_size,
               uint64_t dst_segment_size,
               uint64_t batch_count,
               CUstream stream)
{
  const uint64_t total = batch_count * dst_segment_size;
  if (total == 0)
    return 0;
  const int block_size = 256;
  const uint64_t grid64 = (total + block_size - 1) / block_size;
  if (grid64 > (uint64_t)INT_MAX) {
    log_error("lod_reduce_csr: grid_size %llu exceeds INT_MAX",
              (unsigned long long)grid64);
    return 1;
  }
  const int grid_size = (int)grid64;

#define LAUNCH_CSR(Type, Acc, Method)                                          \
  case Method:                                                                 \
    lod_reduce_csr_k<Type, Acc, Method>                                        \
      <<<grid_size, block_size, 0, stream>>>((Type*)d_values,                  \
                                             (const uint64_t*)d_starts,        \
                                             (const uint64_t*)d_indices,       \
                                             src_offset,                       \
                                             dst_offset,                       \
                                             src_segment_size,                 \
                                             dst_segment_size,                 \
                                             total);                           \
    return 0;

#define CSR_METHODS(Type, Acc)                                                 \
  switch (method) {                                                            \
    LAUNCH_CSR(Type, Acc, lod_reduce_mean);                                    \
    LAUNCH_CSR(Type, Acc, lod_reduce_min);                                     \
    LAUNCH_CSR(Type, Acc, lod_reduce_max);                                     \
    LAUNCH_CSR(Type, Acc, lod_reduce_median);                                  \
    LAUNCH_CSR(Type, Acc, lod_reduce_max_suppressed);                          \
    LAUNCH_CSR(Type, Acc, lod_reduce_min_suppressed);                          \
  }

#define DISPATCH(D, T)                                                         \
  case D:                                                                      \
    CSR_METHODS(T, reduce_acc<T>::type);                                       \
    break;
  switch (dtype) {
    FOR_EACH_DTYPE(DISPATCH)
  }
#undef DISPATCH
#undef CSR_METHODS
#undef LAUNCH_CSR
  return 1;
}
