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
//   - Must be aligned to sizeof(T).
//   - Loads take whole uint32_t words overlapping the requested elements and
//     ignore the surplus on store, so a d_src that is not word aligned needs
//     room back to the previous word boundary, and the buffer needs
//     TRANSPOSE_SOURCE_PAD_BYTES past src_size.
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
  constexpr int T_PER_LOAD =
    sizeof(T) < sizeof(uint32_t) ? (int)(sizeof(uint32_t) / sizeof(T)) : 1;
  static_assert(ELEMENTS_PER_BLOCK % T_PER_LOAD == 0);

  // One word of slack: an unaligned d_src shifts the whole block's load.
  __shared__ __align__(sizeof(uint32_t))
    T shared_buf[ELEMENTS_PER_BLOCK + T_PER_LOAD];

  const int tid = threadIdx.x;
  const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
  const uint64_t left = src_size - (uint64_t)block_offset;
  const int elements =
    left < ELEMENTS_PER_BLOCK ? (int)left : ELEMENTS_PER_BLOCK;

  // Elements between the word boundary below d_src and d_src itself. Always 0
  // for types of 4 bytes or more, which cannot straddle a word boundary.
  const int lead_elements =
    (int)(((uintptr_t)d_src & (sizeof(uint32_t) - 1)) / sizeof(T));

  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    const uint32_t* src_words =
      (const uint32_t*)(d_src - lead_elements) + block_offset / T_PER_LOAD;
    uint32_t* buf_words = (uint32_t*)shared_buf;
    const int words = (lead_elements + elements + T_PER_LOAD - 1) / T_PER_LOAD;
    for (int i = tid; i < words; i += blockDim.x)
      buf_words[i] = src_words[i];
  } else {
    const T* src = d_src + block_offset;
    for (int i = tid; i < elements; i += blockDim.x)
      shared_buf[i] = src[i];
  }

  __syncthreads();

  for (int i = tid; i < elements; i += blockDim.x) {
    uint64_t out_offset = 0;
    uint64_t rest = i_offset + block_offset + i;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out_offset += coord * strides[d];
    }

    d_dst[out_offset] = shared_buf[lead_elements + i];
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

  assert(d_src_beg % sizeof(T) == 0);
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
