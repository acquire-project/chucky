#include "lod.h"

#include <assert.h>
#include <stdint.h>

#define LOD_BLOCK 256

static int
ceil_log2_h(uint64_t v)
{
  if (v <= 1)
    return 0;
  v -= 1;
  int n = 0;
  while (v) {
    v >>= 1;
    ++n;
  }
  return n;
}

static uint64_t
max_shape_h(int ndim, const uint64_t* shape)
{
  uint64_t max = 0;
  for (int d = 0; d < ndim; ++d)
    if (shape[d] > max)
      max = shape[d];
  return max;
}

// Number of elements in a tile of `tile_size` starting at `start`
// within a dimension of `dimension_size`, clamped to the boundary.
__device__ static uint64_t
clamped_tile_extent_d(uint64_t dimension_size,
                      uint64_t start,
                      uint64_t tile_size)
{
  if (start >= dimension_size)
    return 0;
  uint64_t remaining = dimension_size - start;
  return (remaining < tile_size) ? remaining : tile_size;
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
    uint64_t tile_size = 1ull << (total_levels - 1 - level);

    int digit = 0;

    // Compute subtree-width products and extract the Morton digit.
    products[0] = 1;
    for (int d = 0; d < ndim; ++d) {
      uint64_t coord = coords[d];
      uint64_t node_low = tree_child_low(coord, level, nlod);
      if (level < nlod)
        digit |= (int)((coord >> (nlod - 1 - level)) & 1) << d;
      uint64_t extent_low =
        clamped_tile_extent_d(shape[d], node_low * tile_size, tile_size);
      uint64_t extent_high =
        clamped_tile_extent_d(shape[d], (node_low + 1) * tile_size, tile_size);
      products[d + 1] = products[d] * (extent_low + extent_high);
    }

    // For each set bit in digit, count elements in the low subtree.
    uint64_t suffix = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      uint64_t coord = coords[d];
      uint64_t node_low = tree_child_low(coord, level, nlod);
      int bit = (digit >> d) & 1;
      uint64_t extent_low =
        clamped_tile_extent_d(shape[d], node_low * tile_size, tile_size);
      if (bit == 1)
        rank += suffix * extent_low * products[d];
      uint64_t extent_chosen = clamped_tile_extent_d(
        shape[d], (node_low + (uint64_t)bit) * tile_size, tile_size);
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
// Caller must ensure dst is 4-byte aligned.

template<typename T>
__global__ void __launch_bounds__(256, 4)
  lod_gather_lut_k(T* __restrict__ dst,
                   const T* __restrict__ src,
                   const uint32_t* __restrict__ src_lut,
                   const uint32_t* __restrict__ batch_offsets,
                   uint64_t lod_count,
                   uint64_t total)
{
  constexpr int T_PER_U32 = sizeof(uint32_t) / sizeof(T);
  constexpr int TILE_U32 = (1 << 12) / sizeof(uint32_t); // 1024 u32 slots = 4KB
  constexpr int TILE_ELEMENTS = TILE_U32 * T_PER_U32;

  __shared__ uint32_t tile[TILE_U32];

  const int tid = threadIdx.x;
  const uint64_t block_base = (uint64_t)blockIdx.x * TILE_ELEMENTS;

  // Phase 1: Gather scattered reads, pack into u32, write to shared memory
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
          if constexpr (sizeof(T) == 4)
            packed = __float_as_uint(val);
          else
            packed |= ((uint32_t)val) << (k * sizeof(T) * 8);
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

// --- Build tile-scatter LUT kernel ---
// Builds tile_lut: tile_lut[morton_pos] = tile_pool_offset from LOD dims.
// Uses forward LUT to map lod_linear -> morton_pos, then decomposes
// lod_linear into coords and computes lifted-stride offset.

template<int NdimMax>
__global__ void
lod_build_tile_scatter_lut_k(uint32_t* __restrict__ tile_lut,
                             int lod_ndim,
                             const uint64_t* __restrict__ lod_shape,
                             const uint64_t* __restrict__ lod_tile_sizes,
                             const int64_t* __restrict__ lod_tile_strides,
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
    uint64_t tile_idx = coord / lod_tile_sizes[d];
    uint64_t within = coord % lod_tile_sizes[d];
    offset += (int64_t)tile_idx * lod_tile_strides[2 * d];
    offset += (int64_t)within * lod_tile_strides[2 * d + 1];
  }

  uint64_t morton_pos =
    morton_rank_d<NdimMax>(lod_ndim, lod_shape, lod_nlod, coords, 0);
  tile_lut[morton_pos] = (uint32_t)offset;
}

// --- Morton-to-tile scatter kernel using LUT ---
// Sequential reads from morton buffer, LUT-directed writes to tile pool.

template<typename T>
__global__ void
lod_morton_to_tiles_lut_k(T* __restrict__ dst,
                          const T* __restrict__ src,
                          const uint32_t* __restrict__ tile_lut,
                          const uint32_t* __restrict__ batch_tile_offsets,
                          uint64_t lod_count,
                          uint64_t total)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= total)
    return;

  uint64_t batch = gid / lod_count;
  uint64_t morton_pos = gid % lod_count;
  dst[(uint64_t)batch_tile_offsets[batch] + tile_lut[morton_pos]] = src[gid];
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

extern "C" void
lod_build_gather_lut(CUdeviceptr d_src_lut,
                     CUdeviceptr d_lod_shape,
                     CUdeviceptr d_lod_strides,
                     int lod_ndim,
                     const uint64_t* lod_shape_host,
                     uint64_t lod_count,
                     CUstream stream)
{
  int lod_nlod = ceil_log2_h(max_shape_h(lod_ndim, lod_shape_host));

#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_gather_lut_launch<maxdim>(d_src_lut,                             \
                                        d_lod_shape,                           \
                                        d_lod_strides,                         \
                                        lod_ndim,                              \
                                        lod_nlod,                              \
                                        lod_count,                             \
                                        stream);                               \
    return;                                                                    \
  }

  XXX(4);
  XXX(8);
#undef XXX
}

template<int NdimMax>
static void
lod_build_tile_scatter_lut_launch(CUdeviceptr d_tile_lut,
                                  CUdeviceptr d_lod_shape,
                                  CUdeviceptr d_lod_tile_sizes,
                                  CUdeviceptr d_lod_tile_strides,
                                  int lod_ndim,
                                  int lod_nlod,
                                  uint64_t lod_count,
                                  CUstream stream)
{
  const int grid_size = (int)((lod_count + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_build_tile_scatter_lut_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint32_t*)d_tile_lut,
                                          lod_ndim,
                                          (const uint64_t*)d_lod_shape,
                                          (const uint64_t*)d_lod_tile_sizes,
                                          (const int64_t*)d_lod_tile_strides,
                                          lod_nlod,
                                          lod_count);
}

extern "C" void
lod_build_tile_scatter_lut(CUdeviceptr d_tile_lut,
                           CUdeviceptr d_lod_shape,
                           CUdeviceptr d_lod_tile_sizes,
                           CUdeviceptr d_lod_tile_strides,
                           int lod_ndim,
                           const uint64_t* lod_shape_host,
                           uint64_t lod_count,
                           CUstream stream)
{
  int lod_nlod = ceil_log2_h(max_shape_h(lod_ndim, lod_shape_host));

#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_tile_scatter_lut_launch<maxdim>(d_tile_lut,                      \
                                              d_lod_shape,                     \
                                              d_lod_tile_sizes,                \
                                              d_lod_tile_strides,              \
                                              lod_ndim,                        \
                                              lod_nlod,                        \
                                              lod_count,                       \
                                              stream);                         \
    return;                                                                    \
  }

  XXX(4);
  XXX(8);
#undef XXX
}

template<typename T>
static void
lod_morton_to_tiles_lut_launch(CUdeviceptr d_tiles,
                               CUdeviceptr d_morton,
                               CUdeviceptr d_tile_lut,
                               CUdeviceptr d_batch_tile_offsets,
                               uint64_t lod_count,
                               uint64_t batch_count,
                               CUstream stream)
{
  const uint64_t total = batch_count * lod_count;
  const int grid_size = (int)((total + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_morton_to_tiles_lut_k<T>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((T*)d_tiles,
                                          (const T*)d_morton,
                                          (const uint32_t*)d_tile_lut,
                                          (const uint32_t*)d_batch_tile_offsets,
                                          lod_count,
                                          total);
}

extern "C" void
lod_morton_to_tiles_lut(CUdeviceptr d_tiles,
                        CUdeviceptr d_morton,
                        CUdeviceptr d_tile_lut,
                        CUdeviceptr d_batch_tile_offsets,
                        enum lod_dtype dtype,
                        uint64_t lod_count,
                        uint64_t batch_count,
                        CUstream stream)
{
#define XXX(target_dtype, T)                                                   \
  if (dtype == target_dtype) {                                                 \
    lod_morton_to_tiles_lut_launch<T>(d_tiles,                                 \
                                      d_morton,                                \
                                      d_tile_lut,                              \
                                      d_batch_tile_offsets,                    \
                                      lod_count,                               \
                                      batch_count,                             \
                                      stream);                                 \
    return;                                                                    \
  }

  XXX(lod_dtype_u16, uint16_t);
  XXX(lod_dtype_f32, float);
#undef XXX
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
  constexpr int T_PER_U32 = sizeof(uint32_t) / sizeof(T);
  constexpr int TILE_ELEMENTS = ((1 << 12) / (int)sizeof(uint32_t)) * T_PER_U32;
  const int grid_size = (int)((total + TILE_ELEMENTS - 1) / TILE_ELEMENTS);

  lod_gather_lut_k<T>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((T*)d_dst,
                                          (const T*)d_src,
                                          (const uint32_t*)d_src_lut,
                                          (const uint32_t*)d_batch_offsets,
                                          lod_count,
                                          total);
}

extern "C" void
lod_gather_lut(CUdeviceptr d_dst,
               CUdeviceptr d_src,
               CUdeviceptr d_src_lut,
               CUdeviceptr d_batch_offsets,
               enum lod_dtype dtype,
               uint64_t lod_count,
               uint64_t batch_count,
               CUstream stream)
{
#define XXX(target_dtype, T)                                                   \
  if (dtype == target_dtype) {                                                 \
    lod_gather_lut_launch<T>(d_dst,                                            \
                             d_src,                                            \
                             d_src_lut,                                        \
                             d_batch_offsets,                                  \
                             lod_count,                                        \
                             batch_count,                                      \
                             stream);                                          \
    return;                                                                    \
  }

  XXX(lod_dtype_u16, uint16_t);
  XXX(lod_dtype_f32, float);
#undef XXX
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

extern "C" void
lod_fill_ends_gpu(CUdeviceptr d_ends,
                  int ndim,
                  CUdeviceptr d_child_shape,
                  CUdeviceptr d_parent_shape,
                  const uint64_t* child_shape_host,
                  const uint64_t* parent_shape_host,
                  uint64_t n_parents,
                  CUstream stream)
{
  int parent_nlod = ceil_log2_h(max_shape_h(ndim, parent_shape_host));
  int child_nlod = ceil_log2_h(max_shape_h(ndim, child_shape_host));

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
    return;                                                                    \
  }

  XXX(4);
  XXX(8);
#undef XXX
}

// --- Accumulator emit kernel (dim0 LOD) ---
// Finalizes accumulator to native type.
// For mean: divide sum by count. For min/max: just copy.

template<typename T, typename Acc, enum lod_reduce_method Method>
__global__ void
lod_accum_emit_k(T* __restrict__ dst,
                 const Acc* __restrict__ accum,
                 uint64_t n_elements,
                 uint32_t count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= n_elements)
    return;

  if constexpr (Method == lod_reduce_mean)
    dst[gid] = (T)(accum[gid] / (Acc)count);
  else
    dst[gid] = (T)accum[gid];
}

extern "C" void
lod_accum_emit(CUdeviceptr d_dst,
               CUdeviceptr d_accum,
               enum lod_dtype dtype,
               enum lod_reduce_method method,
               uint64_t n_elements,
               uint32_t count,
               CUstream stream)
{
  const int grid = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);

#define LAUNCH_EMIT(T, Acc, M)                                                 \
  lod_accum_emit_k<T, Acc, M><<<grid, LOD_BLOCK, 0, stream>>>(                 \
    (T*)d_dst, (const Acc*)d_accum, n_elements, count)

#define EMIT_METHOD(T, Acc)                                                    \
  switch (method) {                                                            \
    case lod_reduce_mean:                                                      \
      LAUNCH_EMIT(T, Acc, lod_reduce_mean);                                    \
      return;                                                                  \
    case lod_reduce_min:                                                       \
      LAUNCH_EMIT(T, T, lod_reduce_min);                                       \
      return;                                                                  \
    case lod_reduce_max:                                                       \
      LAUNCH_EMIT(T, T, lod_reduce_max);                                       \
      return;                                                                  \
    default:                                                                   \
      return;                                                                  \
  }

  switch (dtype) {
    case lod_dtype_u16:
      EMIT_METHOD(uint16_t, uint32_t);
      break;
    case lod_dtype_f32:
      EMIT_METHOD(float, float);
      break;
  }

#undef LAUNCH_EMIT
#undef EMIT_METHOD
}

// --- Fused accumulator fold kernel (dim0 LOD) ---
// Single launch over all LOD levels 1+. Each thread reads its level from
// d_level_ids and the corresponding count from d_counts.

template<typename T, typename Acc, enum lod_reduce_method Method>
__global__ void
lod_accum_fold_fused_k(Acc* __restrict__ accum,
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
    accum[gid] = (Acc)val;
  } else {
    if constexpr (Method == lod_reduce_mean)
      accum[gid] += (Acc)val;
    else if constexpr (Method == lod_reduce_min)
      accum[gid] = min(accum[gid], (Acc)val);
    else if constexpr (Method == lod_reduce_max)
      accum[gid] = max(accum[gid], (Acc)val);
  }
}

extern "C" void
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

#define LAUNCH_FUSED(T, Acc, M)                                                \
  lod_accum_fold_fused_k<T, Acc, M>                                            \
    <<<grid, LOD_BLOCK, 0, stream>>>((Acc*)d_accum,                            \
                                     (const T*)d_new_data,                     \
                                     (const uint8_t*)d_level_ids,              \
                                     (const uint32_t*)d_counts,                \
                                     n_elements)

#define FUSED_METHOD(T, Acc)                                                   \
  switch (method) {                                                            \
    case lod_reduce_mean:                                                      \
      LAUNCH_FUSED(T, Acc, lod_reduce_mean);                                   \
      return;                                                                  \
    case lod_reduce_min:                                                       \
      LAUNCH_FUSED(T, T, lod_reduce_min);                                      \
      return;                                                                  \
    case lod_reduce_max:                                                       \
      LAUNCH_FUSED(T, T, lod_reduce_max);                                      \
      return;                                                                  \
    default:                                                                   \
      return;                                                                  \
  }

  switch (dtype) {
    case lod_dtype_u16:
      FUSED_METHOD(uint16_t, uint32_t);
      break;
    case lod_dtype_f32:
      FUSED_METHOD(float, float);
      break;
  }

#undef LAUNCH_FUSED
#undef FUSED_METHOD
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

#define CASE(Type, Acc, Method)                                                \
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

#define CASE2(Dtype, Type, Acc)                                                \
  case Dtype:                                                                  \
    XXX(Type, Acc);                                                            \
    break;

#define XXX(Type, Acc)                                                         \
  switch (method) {                                                            \
    CASE(Type, Acc, lod_reduce_mean);                                          \
    CASE(Type, Acc, lod_reduce_min);                                           \
    CASE(Type, Acc, lod_reduce_max);                                           \
    CASE(Type, Acc, lod_reduce_median);                                        \
    CASE(Type, Acc, lod_reduce_max_suppressed);                                \
    CASE(Type, Acc, lod_reduce_min_suppressed);                                \
  }

  switch (dtype) {
    CASE2(lod_dtype_u16, uint16_t, uint32_t);
    CASE2(lod_dtype_f32, float, float);
  }

#undef CASE
#undef CASE2
#undef XXX
  return 1; // error - invalid parameter
}
