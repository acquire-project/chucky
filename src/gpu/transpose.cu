#include "gpu/transpose.h"
#include <assert.h>
#include <stdint.h>

// Transpose data kernel - v0
// Each element's destination comes from its own index, so one launch covers the
// buffer however many epochs it spans.
template<typename T>
__global__ void __launch_bounds__(256, 4)
  transpose_v0_k(T* d_dst,
                 const T* d_src,
                 uint64_t src_size,
                 uint64_t i_offset,
                 uint64_t epoch_elements,
                 uint64_t epoch_stride,
                 uint8_t rank,
                 const uint64_t* shape,
                 const int64_t* strides)
{
  constexpr int ELEMENTS_PER_BLOCK = (1 << 12) / sizeof(T); // 4KB

  const int tid = threadIdx.x;
  const uint64_t block_offset = (uint64_t)blockIdx.x * ELEMENTS_PER_BLOCK;
  const uint64_t left = src_size - block_offset;
  const int elements =
    left < ELEMENTS_PER_BLOCK ? (int)left : ELEMENTS_PER_BLOCK;

  for (int i = tid; i < elements; i += blockDim.x) {
    uint64_t rest = i_offset + block_offset + i;
    uint64_t out_offset = (rest / epoch_elements) * epoch_stride;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out_offset += coord * strides[d];
    }

    d_dst[out_offset] = d_src[block_offset + i];
  }
}

template<typename T>
static void
transpose_launch(CUdeviceptr d_dst_beg,
                 CUdeviceptr d_src_beg,
                 uint64_t src_bytes,
                 uint64_t i_offset,
                 uint64_t epoch_elements,
                 uint64_t epoch_bytes,
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
  assert(epoch_bytes % sizeof(T) == 0);
  assert(epoch_elements > 0);
  const int grid_size =
    (int)((src_size + elements_per_block - 1) / elements_per_block);

  transpose_v0_k<T>
    <<<grid_size, block_size, 0, cuda_stream>>>((T*)d_dst_beg,
                                                (T*)d_src_beg,
                                                src_size,
                                                i_offset,
                                                epoch_elements,
                                                epoch_bytes / sizeof(T),
                                                rank,
                                                d_shape,
                                                d_strides);
}

extern "C" void
transpose(CUdeviceptr d_dst_beg,
          CUdeviceptr d_src_beg,
          uint64_t src_bytes,
          uint8_t bpe,
          uint64_t i_offset,
          uint64_t epoch_elements,
          uint64_t epoch_bytes,
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
                                epoch_elements,
                                epoch_bytes,
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
                                 epoch_elements,
                                 epoch_bytes,
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
                                 epoch_elements,
                                 epoch_bytes,
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
                                 epoch_elements,
                                 epoch_bytes,
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
