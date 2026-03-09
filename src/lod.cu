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

// --- Scatter kernel (templated on dtype and NdimMax) ---

template<typename T, int NdimMax>
__global__ void
lod_scatter_k(T* __restrict__ dst,
              const T* __restrict__ src,
              int ndim,
              uint64_t n_elements,
              const uint64_t* __restrict__ full_shape,
              const uint64_t* __restrict__ lod_shape,
              int lod_ndim,
              int lod_nlod,
              uint32_t lod_mask,
              uint64_t lod_count)
{
  const int tid = threadIdx.x;
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + tid;
  if (gid >= n_elements)
    return;

  // Decompose gid in C-order (dim ndim-1 fastest) to match row-major data
  // layout. Assign LOD coords in reverse so lod_coords[k] pairs with
  // lod_shape[k].
  uint64_t lod_coords[NdimMax];
  uint64_t remainder = gid;
  uint64_t batch_index = 0, batch_stride = 1;
  int lod_dim = lod_ndim - 1;
  for (int d = ndim - 1; d >= 0; --d) {
    uint64_t coord = remainder % full_shape[d];
    remainder /= full_shape[d];
    if ((lod_mask >> d) & 1) {
      lod_coords[lod_dim] = coord;
      lod_dim--;
    } else {
      batch_index += coord * batch_stride;
      batch_stride *= full_shape[d];
    }
  }

  uint64_t pos =
    morton_rank_d<NdimMax>(lod_ndim, lod_shape, lod_nlod, lod_coords, 0);

  dst[batch_index * lod_count + pos] = src[gid];
}

// --- Build scatter LUT kernel ---
// Maps lod_linear_index (row-major within lod_shape) to morton_rank.

template<int NdimMax>
__global__ void
lod_build_scatter_lut_k(uint32_t* __restrict__ lut,
                        int ndim,
                        const uint64_t* __restrict__ lod_shape,
                        int lod_nlod,
                        uint64_t lod_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= lod_count)
    return;

  uint64_t coords[NdimMax];
  uint64_t remainder = gid;
  for (int d = ndim - 1; d >= 0; --d) {
    coords[d] = remainder % lod_shape[d];
    remainder /= lod_shape[d];
  }

  lut[gid] =
    (uint32_t)morton_rank_d<NdimMax>(ndim, lod_shape, lod_nlod, coords, 0);
}

// --- Scatter kernel using precomputed LUT ---

template<typename T, int NdimMax>
__global__ void
lod_scatter_lut_k(T* __restrict__ dst,
                  const T* __restrict__ src,
                  const uint32_t* __restrict__ lut,
                  int ndim,
                  uint64_t n_elements,
                  const uint64_t* __restrict__ full_shape,
                  const uint64_t* __restrict__ lod_shape,
                  int lod_ndim,
                  uint32_t lod_mask,
                  uint64_t lod_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= n_elements)
    return;

  uint64_t lod_coords[NdimMax];
  uint64_t remainder = gid;
  uint64_t batch_index = 0, batch_stride = 1;
  int lod_dim = lod_ndim - 1;

  for (int d = ndim - 1; d >= 0; --d) {
    uint64_t coord = remainder % full_shape[d];
    remainder /= full_shape[d];
    if ((lod_mask >> d) & 1) {
      lod_coords[lod_dim] = coord;
      lod_dim--;
    } else {
      batch_index += coord * batch_stride;
      batch_stride *= full_shape[d];
    }
  }

  // Ravel lod_coords to row-major index within lod_shape
  uint64_t lod_linear = lod_coords[0];
  for (int d = 1; d < lod_ndim; ++d)
    lod_linear = lod_linear * lod_shape[d] + lod_coords[d];

  uint32_t pos = lut[lod_linear];
  dst[batch_index * lod_count + pos] = src[gid];
}

// --- Build gather LUT kernel ---
// Builds inv_lut: src_lut[morton_pos] = src_lod_offset, where
// src_lod_offset is the C-order contribution from LOD dimensions.

template<int NdimMax>
__global__ void
lod_build_gather_lut_k(uint32_t* __restrict__ src_lut,
                       const uint32_t* __restrict__ fwd_lut,
                       int lod_ndim,
                       const uint64_t* __restrict__ lod_shape,
                       const uint64_t* __restrict__ lod_strides,
                       uint64_t lod_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= lod_count)
    return;

  // Decompose gid (= lod_linear) into lod_coords, compute src offset
  uint64_t remainder = gid;
  uint64_t src_offset = 0;
  for (int d = lod_ndim - 1; d >= 0; --d) {
    uint64_t coord = remainder % lod_shape[d];
    remainder /= lod_shape[d];
    src_offset += coord * lod_strides[d];
  }

  src_lut[fwd_lut[gid]] = (uint32_t)src_offset;
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
        if (morton_pos >= lod_count) { morton_pos = 0; batch++; }
      }
      tile[i] = packed;

      // Advance to next iteration: skip (blockDim.x - 1) * T_PER_U32 elements
      morton_pos += (uint64_t)(blockDim.x - 1) * T_PER_U32;
      while (morton_pos >= lod_count) { morton_pos -= lod_count; batch++; }
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
                              const uint32_t* __restrict__ fwd_lut,
                              int lod_ndim,
                              const uint64_t* __restrict__ lod_shape,
                              const uint64_t* __restrict__ lod_tile_sizes,
                              const int64_t* __restrict__ lod_tile_strides,
                              uint64_t lod_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= lod_count)
    return;

  // Decompose gid (= lod_linear) into lod_coords, compute tile offset
  uint64_t remainder = gid;
  int64_t offset = 0;
  for (int d = lod_ndim - 1; d >= 0; --d) {
    uint64_t coord = remainder % lod_shape[d];
    remainder /= lod_shape[d];
    uint64_t tile_idx = coord / lod_tile_sizes[d];
    uint64_t within = coord % lod_tile_sizes[d];
    offset += (int64_t)tile_idx * lod_tile_strides[2 * d];
    offset += (int64_t)within * lod_tile_strides[2 * d + 1];
  }

  tile_lut[fwd_lut[gid]] = (uint32_t)offset;
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
  uint64_t len = end - start;

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

// --- Morton-to-tile scatter kernel ---
// Each thread handles one element in the level's full shape.
// It computes the morton position (to read from morton buffer)
// and the tile-pool position (to write using lifted strides).

template<typename T, int NdimMax>
__global__ void
lod_morton_to_tiles_k(T* __restrict__ tiles,
                      const T* __restrict__ morton,
                      int ndim,
                      const uint64_t* __restrict__ full_shape,
                      int lod_ndim,
                      uint32_t lod_mask,
                      const uint64_t* __restrict__ lod_shape,
                      int lod_nlod,
                      uint64_t lod_count,
                      uint64_t n_elements,
                      int lifted_rank,
                      const uint64_t* __restrict__ lifted_shape,
                      const int64_t* __restrict__ lifted_strides)
{
  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + threadIdx.x;
  if (gid >= n_elements)
    return;

  // Load shapes and strides into registers
  uint64_t r_full_shape[NdimMax];
  uint64_t r_lod_shape[NdimMax];
  uint64_t r_lifted_shape[2 * NdimMax];
  int64_t r_lifted_strides[2 * NdimMax];
  for (int d = 0; d < ndim; ++d)
    r_full_shape[d] = full_shape[d];
  for (int d = 0; d < lod_ndim; ++d)
    r_lod_shape[d] = lod_shape[d];
  for (int d = 0; d < lifted_rank; ++d) {
    r_lifted_shape[d] = lifted_shape[d];
    r_lifted_strides[d] = lifted_strides[d];
  }

  // Decompose gid in C-order (dim ndim-1 fastest) to match row-major data
  // layout.
  uint64_t remainder = gid;
  uint64_t coords[NdimMax];
  for (int d = ndim - 1; d >= 0; --d) {
    coords[d] = remainder % r_full_shape[d];
    remainder /= r_full_shape[d];
  }

  // Separate batch and LOD coords, compute batch index.
  // Assign LOD coords in reverse so lod_coords[k] pairs with lod_shape[k].
  uint64_t lod_coords[NdimMax];
  uint64_t batch_index = 0, batch_stride = 1;
  int lod_dim = lod_ndim - 1;
  for (int d = ndim - 1; d >= 0; --d) {
    if ((lod_mask >> d) & 1) {
      lod_coords[lod_dim] = coords[d];
      lod_dim--;
    } else {
      batch_index += coords[d] * batch_stride;
      batch_stride *= r_full_shape[d];
    }
  }

  // Morton rank → read position
  uint64_t mpos =
    morton_rank_d<NdimMax>(lod_ndim, r_lod_shape, lod_nlod, lod_coords, 0);
  T val = morton[batch_index * lod_count + mpos];

  // Tile-pool position via lifted strides
  // lifted shape: (tile_count[D-1], tile_size[D-1], ..., tile_count[0],
  // tile_size[0]) For coordinate coords[d]:
  //   tile_index[d] = coords[d] / tile_size[d]
  //   within_tile[d] = coords[d] % tile_size[d]
  // lifted_strides[2*i] = stride for tile_index of dim i (reversed)
  // lifted_strides[2*i+1] = stride for within_tile of dim i (reversed)
  int64_t tile_pos = 0;
  for (int i = 0; i < ndim; ++i) {
    uint64_t tile_idx = coords[i] / r_lifted_shape[2 * i + 1];
    uint64_t within = coords[i] % r_lifted_shape[2 * i + 1];
    tile_pos += (int64_t)tile_idx * r_lifted_strides[2 * i];
    tile_pos += (int64_t)within * r_lifted_strides[2 * i + 1];
  }

  tiles[tile_pos] = val;
}

// --- Shared memory size helpers ---

// --- Dispatch helpers ---

template<typename T, int NdimMax>
static void
lod_scatter_launch(CUdeviceptr d_dst,
                   CUdeviceptr d_src,
                   int ndim,
                   uint64_t n_elements,
                   CUdeviceptr d_full_shape,
                   CUdeviceptr d_lod_shape,
                   int lod_ndim,
                   int lod_nlod,
                   uint32_t lod_mask,
                   uint64_t lod_count,
                   CUstream stream)
{
  const int grid_size = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_scatter_k<T, NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((T*)d_dst,
                                          (const T*)d_src,
                                          ndim,
                                          n_elements,
                                          (const uint64_t*)d_full_shape,
                                          (const uint64_t*)d_lod_shape,
                                          lod_ndim,
                                          lod_nlod,
                                          lod_mask,
                                          lod_count);
}

// FIXME: this should return 0 on error, otherwise 1.
extern "C" int
lod_scatter(CUdeviceptr d_dst,
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
            CUstream stream)
{
  int lod_nlod = ceil_log2_h(max_shape_h(lod_ndim, lod_shape_host));

#define XXX(target_dtype, T, maxdim)                                           \
  if (dtype == target_dtype && ndim <= maxdim) {                               \
    lod_scatter_launch<T, maxdim>(d_dst,                                       \
                                  d_src,                                       \
                                  ndim,                                        \
                                  n_elements,                                  \
                                  d_full_shape,                                \
                                  d_lod_shape,                                 \
                                  lod_ndim,                                    \
                                  lod_nlod,                                    \
                                  lod_mask,                                    \
                                  lod_count,                                   \
                                  stream);                                     \
    return 1;                                                                  \
  }

  XXX(lod_dtype_u16, uint16_t, 4);
  XXX(lod_dtype_f32, float, 4);
  XXX(lod_dtype_u16, uint16_t, 8);
  XXX(lod_dtype_f32, float, 8);
#undef XXX
  return 0; // failure
}

template<int NdimMax>
static void
lod_build_scatter_lut_launch(CUdeviceptr d_lut,
                             CUdeviceptr d_lod_shape,
                             int lod_ndim,
                             int lod_nlod,
                             uint64_t lod_count,
                             CUstream stream)
{
  const int grid_size = (int)((lod_count + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_build_scatter_lut_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint32_t*)d_lut,
                                          lod_ndim,
                                          (const uint64_t*)d_lod_shape,
                                          lod_nlod,
                                          lod_count);
}

extern "C" void
lod_build_scatter_lut(CUdeviceptr d_lut,
                      CUdeviceptr d_lod_shape,
                      int lod_ndim,
                      const uint64_t* lod_shape_host,
                      uint64_t lod_count,
                      CUstream stream)
{
  int lod_nlod = ceil_log2_h(max_shape_h(lod_ndim, lod_shape_host));

#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_scatter_lut_launch<maxdim>(                                      \
      d_lut, d_lod_shape, lod_ndim, lod_nlod, lod_count, stream);              \
    return;                                                                    \
  }

  XXX(4);
  XXX(8);
#undef XXX
}

template<typename T, int NdimMax>
static void
lod_scatter_lut_launch(CUdeviceptr d_dst,
                       CUdeviceptr d_src,
                       CUdeviceptr d_lut,
                       int ndim,
                       uint64_t n_elements,
                       CUdeviceptr d_full_shape,
                       CUdeviceptr d_lod_shape,
                       int lod_ndim,
                       uint32_t lod_mask,
                       uint64_t lod_count,
                       CUstream stream)
{
  const int grid_size = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_scatter_lut_k<T, NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((T*)d_dst,
                                          (const T*)d_src,
                                          (const uint32_t*)d_lut,
                                          ndim,
                                          n_elements,
                                          (const uint64_t*)d_full_shape,
                                          (const uint64_t*)d_lod_shape,
                                          lod_ndim,
                                          lod_mask,
                                          lod_count);
}

extern "C" void
lod_scatter_lut(CUdeviceptr d_dst,
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
                CUstream stream)
{
#define XXX(target_dtype, T, maxdim)                                           \
  if (dtype == target_dtype && ndim <= maxdim) {                               \
    lod_scatter_lut_launch<T, maxdim>(d_dst,                                   \
                                      d_src,                                   \
                                      d_lut,                                   \
                                      ndim,                                    \
                                      n_elements,                              \
                                      d_full_shape,                            \
                                      d_lod_shape,                             \
                                      lod_ndim,                                \
                                      lod_mask,                                \
                                      lod_count,                               \
                                      stream);                                 \
    return;                                                                    \
  }

  XXX(lod_dtype_u16, uint16_t, 4);
  XXX(lod_dtype_f32, float, 4);
  XXX(lod_dtype_u16, uint16_t, 8);
  XXX(lod_dtype_f32, float, 8);
#undef XXX
}

template<int NdimMax>
static void
lod_build_gather_lut_launch(CUdeviceptr d_src_lut,
                            CUdeviceptr d_fwd_lut,
                            CUdeviceptr d_lod_shape,
                            CUdeviceptr d_lod_strides,
                            int lod_ndim,
                            uint64_t lod_count,
                            CUstream stream)
{
  const int grid_size = (int)((lod_count + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_build_gather_lut_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint32_t*)d_src_lut,
                                          (const uint32_t*)d_fwd_lut,
                                          lod_ndim,
                                          (const uint64_t*)d_lod_shape,
                                          (const uint64_t*)d_lod_strides,
                                          lod_count);
}

extern "C" void
lod_build_gather_lut(CUdeviceptr d_src_lut,
                     CUdeviceptr d_fwd_lut,
                     CUdeviceptr d_lod_shape,
                     CUdeviceptr d_lod_strides,
                     int lod_ndim,
                     uint64_t lod_count,
                     CUstream stream)
{
#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_gather_lut_launch<maxdim>(d_src_lut,                             \
                                        d_fwd_lut,                             \
                                        d_lod_shape,                           \
                                        d_lod_strides,                         \
                                        lod_ndim,                              \
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
                                   CUdeviceptr d_fwd_lut,
                                   CUdeviceptr d_lod_shape,
                                   CUdeviceptr d_lod_tile_sizes,
                                   CUdeviceptr d_lod_tile_strides,
                                   int lod_ndim,
                                   uint64_t lod_count,
                                   CUstream stream)
{
  const int grid_size = (int)((lod_count + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_build_tile_scatter_lut_k<NdimMax>
    <<<grid_size, LOD_BLOCK, 0, stream>>>((uint32_t*)d_tile_lut,
                                          (const uint32_t*)d_fwd_lut,
                                          lod_ndim,
                                          (const uint64_t*)d_lod_shape,
                                          (const uint64_t*)d_lod_tile_sizes,
                                          (const int64_t*)d_lod_tile_strides,
                                          lod_count);
}

extern "C" void
lod_build_tile_scatter_lut(CUdeviceptr d_tile_lut,
                            CUdeviceptr d_fwd_lut,
                            CUdeviceptr d_lod_shape,
                            CUdeviceptr d_lod_tile_sizes,
                            CUdeviceptr d_lod_tile_strides,
                            int lod_ndim,
                            uint64_t lod_count,
                            CUstream stream)
{
#define XXX(maxdim)                                                            \
  if (lod_ndim <= maxdim) {                                                    \
    lod_build_tile_scatter_lut_launch<maxdim>(d_tile_lut,                      \
                                              d_fwd_lut,                       \
                                              d_lod_shape,                     \
                                              d_lod_tile_sizes,                \
                                              d_lod_tile_strides,              \
                                              lod_ndim,                        \
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

extern "C" int
lod_morton_tile_nlod(int lod_ndim, const uint64_t* lod_shape_host)
{
  return ceil_log2_h(max_shape_h(lod_ndim, lod_shape_host));
}

template<typename T, int NdimMax>
static void
lod_morton_to_tiles_launch(CUdeviceptr d_tiles,
                           CUdeviceptr d_morton,
                           const struct morton_tile_layout* layout,
                           CUstream stream)
{
  int ndim = layout->ndim;
  int lod_ndim = layout->lod_ndim;
  int lifted_rank = 2 * ndim;
  uint64_t n_elements = layout->n_elements;
  const int grid_size = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);

  lod_morton_to_tiles_k<T, NdimMax><<<grid_size, LOD_BLOCK, 0, stream>>>(
    (T*)d_tiles,
    (const T*)d_morton,
    ndim,
    (const uint64_t*)layout->d_full_shape,
    lod_ndim,
    layout->lod_mask,
    (const uint64_t*)layout->d_lod_shape,
    layout->lod_nlod,
    layout->lod_count,
    n_elements,
    lifted_rank,
    (const uint64_t*)layout->d_lifted_shape,
    (const int64_t*)layout->d_lifted_strides);
}

extern "C" void
lod_morton_to_tiles(CUdeviceptr d_tiles,
                    CUdeviceptr d_morton,
                    const struct morton_tile_layout* layout,
                    CUstream stream)
{
#define XXX(target_dtype, T, maxdim)                                           \
  if (layout->dtype == target_dtype && layout->ndim <= maxdim) {               \
    lod_morton_to_tiles_launch<T, maxdim>(d_tiles, d_morton, layout, stream);  \
    return;                                                                    \
  }

  XXX(lod_dtype_u16, uint16_t, 4);
  XXX(lod_dtype_f32, float, 4);
  XXX(lod_dtype_u16, uint16_t, 8);
  XXX(lod_dtype_f32, float, 8);
#undef XXX
}

// Helper macro: dispatch over all methods for a given (Type, Acc) pair.
#define DISPATCH_METHOD(Type, Acc)                                             \
  switch (method) {                                                            \
    case lod_reduce_mean:                                                      \
      LAUNCH_REDUCE(Type, Acc, lod_reduce_mean);                               \
      break;                                                                   \
    case lod_reduce_min:                                                       \
      LAUNCH_REDUCE(Type, Acc, lod_reduce_min);                                \
      break;                                                                   \
    case lod_reduce_max:                                                       \
      LAUNCH_REDUCE(Type, Acc, lod_reduce_max);                                \
      break;                                                                   \
    case lod_reduce_median:                                                    \
      LAUNCH_REDUCE(Type, Acc, lod_reduce_median);                             \
      break;                                                                   \
    case lod_reduce_max_suppressed:                                            \
      LAUNCH_REDUCE(Type, Acc, lod_reduce_max_suppressed);                     \
      break;                                                                   \
    case lod_reduce_min_suppressed:                                            \
      LAUNCH_REDUCE(Type, Acc, lod_reduce_min_suppressed);                     \
      break;                                                                   \
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
