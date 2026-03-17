#include "index.ops.h"
#include "lod.h"
#include "prelude.h"

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>

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
  X(lod_dtype_u8, uint8_t)                                                     \
  X(lod_dtype_u16, uint16_t)                                                   \
  X(lod_dtype_u32, uint32_t)                                                   \
  X(lod_dtype_u64, uint64_t)                                                   \
  X(lod_dtype_i8, int8_t)                                                      \
  X(lod_dtype_i16, int16_t)                                                    \
  X(lod_dtype_i32, int32_t)                                                    \
  X(lod_dtype_i64, int64_t)                                                    \
  X(lod_dtype_f16, __half)                                                     \
  X(lod_dtype_f32, float)                                                      \
  X(lod_dtype_f64, double)

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

// Advance coordinates to the next Morton position using register arrays.
template<int NdimMax>
__device__ static void
morton_next_d(int ndim, int nlod, uint64_t* coords)
{
  for (int bit = 0; bit < nlod; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      uint64_t mask = 1ull << bit;
      coords[d] ^= mask;
      if (coords[d] & mask)
        return;
    }
  }
  for (int d = 0; d < ndim; ++d)
    coords[d] = 0;
  coords[0] = 1ull << nlod;
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
                   const uint32_t* __restrict__ batch_offsets,
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
    dst[gid] = src[(uint64_t)batch_offsets[batch] + src_lut[morton_pos]];
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
            T val = src[(uint64_t)batch_offsets[batch] + src_lut[morton_pos]];
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
lod_morton_to_chunks_lut_k(T* __restrict__ dst,
                           const T* __restrict__ src,
                           const uint32_t* __restrict__ chunk_lut,
                           const uint32_t* __restrict__ batch_chunk_offsets,
                           uint64_t lod_count,
                           uint64_t total)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= total)
    return;

  uint64_t batch = gid / lod_count;
  uint64_t morton_pos = gid % lod_count;
  dst[(uint64_t)batch_chunk_offsets[batch] + chunk_lut[morton_pos]] = src[gid];
}

// --- Fill ends kernel (templated on NdimMax) ---

template<int NdimMax>
__global__ void
lod_fill_ends_k(uint64_t* __restrict__ ends,
                int ndim,
                const uint64_t* __restrict__ child_shape,
                const uint64_t* __restrict__ parent_shape,
                int parent_nlod,
                int child_nlod,
                uint64_t n_parents)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= n_parents)
    return;

  // Load shapes into registers
  uint64_t r_parent_shape[NdimMax];
  uint64_t r_child_shape[NdimMax];
  for (int d = 0; d < ndim; ++d) {
    r_parent_shape[d] = parent_shape[d];
    r_child_shape[d] = child_shape[d];
  }

  uint64_t coords[NdimMax];
  {
    uint64_t remainder = gid;
    for (int d = 0; d < ndim; ++d) {
      coords[d] = remainder % r_parent_shape[d];
      remainder /= r_parent_shape[d];
    }
  }

  uint64_t pos =
    morton_rank_d<NdimMax>(ndim, r_parent_shape, parent_nlod, coords, 0);

  morton_next_d<NdimMax>(ndim, parent_nlod, coords);

  uint64_t val =
    morton_rank_d<NdimMax>(ndim, r_child_shape, child_nlod, coords, 1);

  ends[pos] = val;
}

// --- Reduce kernel (templated on data type and reduce method) ---
// Accumulator type: uint32 for u16, float for f32.

template<typename T, typename Acc, enum lod_reduce_method Method>
__global__ void
lod_reduce_k(T* __restrict__ values,
             const uint64_t* __restrict__ ends,
             uint64_t src_offset,
             uint64_t dst_offset,
             uint64_t src_lod_count,
             uint64_t dst_lod_count,
             uint64_t batch_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t total = batch_count * dst_lod_count;
  if (gid >= total)
    return;

  const uint64_t batch = gid / dst_lod_count;
  const uint64_t element = gid % dst_lod_count;

  const uint64_t src_base = src_offset + batch * src_lod_count;
  const uint64_t dst_base = dst_offset + batch * dst_lod_count;

  uint64_t start = (element > 0) ? ends[element - 1] : 0;
  uint64_t end = ends[element];
#pragma nv_diag_suppress 177, 550 // len unused in min/max constexpr branches
  uint64_t len = end - start;
#pragma nv_diag_default 177, 550

  T result;

  if constexpr (Method == lod_reduce_mean) {
    Acc sum = (Acc)0;
    for (uint64_t j = start; j < end; ++j)
      sum += (Acc)values[src_base + j];
    result = (T)(sum / (Acc)len);
  } else if constexpr (Method == lod_reduce_min) {
    T best = values[src_base + start];
    for (uint64_t j = start + 1; j < end; ++j) {
      T v = values[src_base + j];
      if (v < best)
        best = v;
    }
    result = best;
  } else if constexpr (Method == lod_reduce_max) {
    T best = values[src_base + start];
    for (uint64_t j = start + 1; j < end; ++j) {
      T v = values[src_base + j];
      if (v > best)
        best = v;
    }
    result = best;
  } else if constexpr (Method == lod_reduce_median) {
    // Window sizes are small (typically 2-8), insertion sort is fine.
    T buf[16];
    uint64_t n = (len <= 16) ? len : 16;
    for (uint64_t j = 0; j < n; ++j)
      buf[j] = values[src_base + start + j];
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
    // 2nd highest value
    T top1 = values[src_base + start];
    T top2 = values[src_base + start];
    if (len > 1) {
      T v = values[src_base + start + 1];
      if (v >= top1) {
        top2 = top1;
        top1 = v;
      } else {
        top2 = v;
      }
      for (uint64_t j = start + 2; j < end; ++j) {
        v = values[src_base + j];
        if (v >= top1) {
          top2 = top1;
          top1 = v;
        } else if (v > top2) {
          top2 = v;
        }
      }
    }
    result = top2;
  } else if constexpr (Method == lod_reduce_min_suppressed) {
    // 2nd lowest value
    T bot1 = values[src_base + start];
    T bot2 = values[src_base + start];
    if (len > 1) {
      T v = values[src_base + start + 1];
      if (v <= bot1) {
        bot2 = bot1;
        bot1 = v;
      } else {
        bot2 = v;
      }
      for (uint64_t j = start + 2; j < end; ++j) {
        v = values[src_base + j];
        if (v <= bot1) {
          bot2 = bot1;
          bot1 = v;
        } else if (v < bot2) {
          bot2 = v;
        }
      }
    }
    result = bot2;
  }

  values[dst_base + element] = result;
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
                                CUdeviceptr d_batch_chunk_offsets,
                                uint64_t lod_count,
                                uint64_t batch_count,
                                CUstream stream)
{
  const uint64_t total = batch_count * lod_count;
  const int grid_size = (int)((total + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_morton_to_chunks_lut_k<T><<<grid_size, LOD_BLOCK, 0, stream>>>(
    (T*)d_chunks,
    (const T*)d_morton,
    (const uint32_t*)d_chunk_lut,
    (const uint32_t*)d_batch_chunk_offsets,
    lod_count,
    total);
}

extern "C" int
lod_morton_to_chunks_lut(CUdeviceptr d_chunks,
                         CUdeviceptr d_morton,
                         CUdeviceptr d_chunk_lut,
                         CUdeviceptr d_batch_chunk_offsets,
                         enum lod_dtype dtype,
                         uint64_t lod_count,
                         uint64_t batch_count,
                         CUstream stream)
{
#define DISPATCH(D, T)                                                         \
  if (dtype == D) {                                                            \
    lod_morton_to_chunks_lut_launch<T>(d_chunks,                               \
                                       d_morton,                               \
                                       d_chunk_lut,                            \
                                       d_batch_chunk_offsets,                  \
                                       lod_count,                              \
                                       batch_count,                            \
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
                      CUdeviceptr d_batch_offsets,
                      uint64_t lod_count,
                      uint64_t batch_count,
                      CUstream stream)
{
  const uint64_t total = batch_count * lod_count;
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
                                          (const uint32_t*)d_batch_offsets,
                                          lod_count,
                                          total);
}

extern "C" int
lod_gather_lut(CUdeviceptr d_dst,
               CUdeviceptr d_src,
               CUdeviceptr d_src_lut,
               CUdeviceptr d_batch_offsets,
               enum lod_dtype dtype,
               uint64_t lod_count,
               uint64_t batch_count,
               CUstream stream)
{
#define DISPATCH(D, T)                                                         \
  if (dtype == D) {                                                            \
    lod_gather_lut_launch<T>(d_dst,                                            \
                             d_src,                                            \
                             d_src_lut,                                        \
                             d_batch_offsets,                                  \
                             lod_count,                                        \
                             batch_count,                                      \
                             stream);                                          \
    return 0;                                                                  \
  }
  FOR_EACH_DTYPE(DISPATCH)
#undef DISPATCH
  return 1;
}

template<int NdimMax>
static void
lod_fill_ends_launch(CUdeviceptr d_ends,
                     int ndim,
                     CUdeviceptr d_child_shape,
                     CUdeviceptr d_parent_shape,
                     int parent_nlod,
                     int child_nlod,
                     uint64_t n_parents,
                     CUstream stream)
{
  const int grid_size = (int)((n_parents + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_fill_ends_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint64_t*)d_ends,
                                          ndim,
                                          (const uint64_t*)d_child_shape,
                                          (const uint64_t*)d_parent_shape,
                                          parent_nlod,
                                          child_nlod,
                                          n_parents);
}

extern "C" int
lod_fill_ends_gpu(CUdeviceptr d_ends,
                  int ndim,
                  CUdeviceptr d_child_shape,
                  CUdeviceptr d_parent_shape,
                  const uint64_t* child_shape_host,
                  const uint64_t* parent_shape_host,
                  uint64_t n_parents,
                  CUstream stream)
{
  int parent_nlod = ceil_log2(max_shape(ndim, parent_shape_host));
  int child_nlod = ceil_log2(max_shape(ndim, child_shape_host));

#define XXX(maxdim)                                                            \
  if (ndim <= maxdim) {                                                        \
    lod_fill_ends_launch<maxdim>(d_ends,                                       \
                                 ndim,                                         \
                                 d_child_shape,                                \
                                 d_parent_shape,                               \
                                 parent_nlod,                                  \
                                 child_nlod,                                   \
                                 n_parents,                                    \
                                 stream);                                      \
    return 0;                                                                  \
  }

  XXX(4);
  XXX(8);
#undef XXX
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
               enum lod_dtype dtype,
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
                     enum lod_dtype dtype,
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

extern "C" int
lod_reduce(CUdeviceptr d_values,
           CUdeviceptr d_ends,
           enum lod_dtype dtype,
           enum lod_reduce_method method,
           uint64_t src_offset,
           uint64_t dst_offset,
           uint64_t src_lod_count,
           uint64_t dst_lod_count,
           uint64_t batch_count,
           CUstream stream)
{
  const uint64_t total = batch_count * dst_lod_count;
  const int block_size = 256;
  const int grid_size = (int)((total + block_size - 1) / block_size);

#define LAUNCH_REDUCE(Type, Acc, Method)                                       \
  case Method:                                                                 \
    lod_reduce_k<Type, Acc, Method>                                            \
      <<<grid_size, block_size, 0, stream>>>((Type*)d_values,                  \
                                             (const uint64_t*)d_ends,          \
                                             src_offset,                       \
                                             dst_offset,                       \
                                             src_lod_count,                    \
                                             dst_lod_count,                    \
                                             batch_count);                     \
    return 0;

#define REDUCE_METHODS(Type, Acc)                                              \
  switch (method) {                                                            \
    LAUNCH_REDUCE(Type, Acc, lod_reduce_mean);                                 \
    LAUNCH_REDUCE(Type, Acc, lod_reduce_min);                                  \
    LAUNCH_REDUCE(Type, Acc, lod_reduce_max);                                  \
    LAUNCH_REDUCE(Type, Acc, lod_reduce_median);                               \
    LAUNCH_REDUCE(Type, Acc, lod_reduce_max_suppressed);                       \
    LAUNCH_REDUCE(Type, Acc, lod_reduce_min_suppressed);                       \
  }

#define DISPATCH(D, T)                                                         \
  case D:                                                                      \
    REDUCE_METHODS(T, reduce_acc<T>::type);                                    \
    break;
  switch (dtype) {
    FOR_EACH_DTYPE(DISPATCH)
  }
#undef DISPATCH
#undef REDUCE_METHODS
#undef LAUNCH_REDUCE
  return 1;
}
