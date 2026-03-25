#include "gpu/transpose.h"
#include <assert.h>
#include <stdint.h>

__global__ void
fill_k(uint16_t* beg, uint16_t* end)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (beg + i < end) {
    beg[i] = i;
  }
}

extern "C" void
fill_u16(CUdeviceptr d_beg,
         CUdeviceptr d_end,
         int grid_size,
         int block_size,
         CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  fill_k<<<grid_size, block_size, 0, cuda_stream>>>((uint16_t*)d_beg,
                                                    (uint16_t*)d_end);
}

// Transpose indices kernel
// Each thread independently computes one output index using the add() algorithm
__global__ void
transpose_indices_k(uint64_t* d_out,
                    uint64_t beg,
                    uint64_t end,
                    uint8_t rank,
                    const uint64_t* shape,
                    const int64_t* strides)
{
  const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t input_idx = beg + gid;

  if (input_idx < end) {
    // Decompose input_idx into coordinates, then compute output offset
    uint64_t out = 0;
    uint64_t rest = input_idx;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out += coord * strides[d];
    }

    d_out[gid] = out;
  }
}

extern "C" void
transpose_indices(CUdeviceptr d_beg,
                  CUdeviceptr d_end,
                  uint64_t i_offset,
                  uint8_t rank,
                  const uint64_t* d_shape,
                  const int64_t* d_strides,
                  CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  uint64_t* out_beg = (uint64_t*)d_beg;
  uint64_t* out_end = (uint64_t*)d_end;
  const uint64_t n = (uint64_t)(out_end - out_beg);

  const int block_size = 256;
  const int grid_size = (int)((n + block_size - 1) / block_size);

  transpose_indices_k<<<grid_size, block_size, 0, cuda_stream>>>(
    out_beg, i_offset, i_offset + n, rank, d_shape, d_strides);
}

// Transpose data kernel - v0
// Uses shared memory to stage data before computing output positions.
//
// Requirements on d_src:
//   - Must be aligned to sizeof(uint32_t) bytes.
//   - Must be padded to a multiple of ELEMENTS_PER_BLOCK elements so that
//     all vectorized loads are in-bounds. Padding values are ignored (filtered
//     on store by src_size).
template<typename T>
__global__ void __launch_bounds__(256, 4) transpose_v0_k(T* d_dst,
                                                         const T* d_src,
                                                         uint64_t src_size,
                                                         uint64_t i_offset,
                                                         uint8_t rank,
                                                         const uint64_t* shape,
                                                         const int64_t* strides)
{
  constexpr int ELEMENTS_PER_BLOCK = (1 << 12) / sizeof(T); // 4KB

  __shared__ T shared_buf[ELEMENTS_PER_BLOCK];

  const int tid = threadIdx.x;
  const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;

  // Vectorized load into shared memory. For small types (< 4 bytes) we coalesce
  // through uint32_t loads; for types >= 4 bytes we load natively.
  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    constexpr int T_PER_LOAD = sizeof(uint32_t) / sizeof(T);
    constexpr int LOADS_PER_BLOCK = ELEMENTS_PER_BLOCK / T_PER_LOAD;
    static_assert(ELEMENTS_PER_BLOCK % T_PER_LOAD == 0);

    const uint32_t* src_u32 = (const uint32_t*)(d_src + block_offset);
    uint32_t* buf_u32 = (uint32_t*)shared_buf;
    for (int i = tid; i < LOADS_PER_BLOCK; i += blockDim.x)
      buf_u32[i] = src_u32[i];
  } else {
    const T* src = d_src + block_offset;
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
      shared_buf[i] = src[i];
  }

  __syncthreads();

  for (int i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
    int global_idx = block_offset + i;

    if (global_idx < src_size) {
      uint64_t input_idx = i_offset + global_idx;
      uint64_t out_offset = 0;
      uint64_t rest = input_idx;

      for (int d = rank - 1; d >= 0; --d) {
        const uint64_t coord = rest % shape[d];
        rest /= shape[d];
        out_offset += coord * strides[d];
      }

      // Store
      d_dst[out_offset] = shared_buf[i];
    }
  }
}

template<typename T>
static void
transpose_launch(CUdeviceptr d_dst_beg,
                 CUdeviceptr d_src_beg,
                 uint64_t src_bytes,
                 uint64_t i_offset,
                 uint8_t rank,
                 const uint64_t* d_shape,
                 const int64_t* d_strides,
                 CUstream stream)
{
  const uint64_t src_size = src_bytes / sizeof(T);
  if (src_size == 0)
    return;

  cudaStream_t cuda_stream = (cudaStream_t)stream;
  const int block_size = 256;
  const int elements_per_block = (1 << 12) / (int)sizeof(T);

  assert(d_src_beg %
           (sizeof(T) < sizeof(uint32_t) ? sizeof(uint32_t) : sizeof(T)) ==
         0);
  const int grid_size =
    (int)((src_size + elements_per_block - 1) / elements_per_block);

  transpose_v0_k<T><<<grid_size, block_size, 0, cuda_stream>>>(
    (T*)d_dst_beg, (T*)d_src_beg, src_size, i_offset, rank, d_shape, d_strides);
}

extern "C" void
transpose(CUdeviceptr d_dst_beg,
          CUdeviceptr d_src_beg,
          uint64_t src_bytes,
          uint8_t bpe,
          uint64_t i_offset,
          uint8_t rank,
          const uint64_t* d_shape,
          const int64_t* d_strides,
          CUstream stream)
{
  switch (bpe) {
    case 1:
      transpose_launch<uint8_t>(d_dst_beg,
                                d_src_beg,
                                src_bytes,
                                i_offset,
                                rank,
                                d_shape,
                                d_strides,
                                stream);
      break;
    case 2:
      transpose_launch<uint16_t>(d_dst_beg,
                                 d_src_beg,
                                 src_bytes,
                                 i_offset,
                                 rank,
                                 d_shape,
                                 d_strides,
                                 stream);
      break;
    case 4:
      transpose_launch<uint32_t>(d_dst_beg,
                                 d_src_beg,
                                 src_bytes,
                                 i_offset,
                                 rank,
                                 d_shape,
                                 d_strides,
                                 stream);
      break;
    case 8:
      transpose_launch<uint64_t>(d_dst_beg,
                                 d_src_beg,
                                 src_bytes,
                                 i_offset,
                                 rank,
                                 d_shape,
                                 d_strides,
                                 stream);
      break;
    default:
      assert(!"transpose: unsupported bpe");
      break;
  }
}
