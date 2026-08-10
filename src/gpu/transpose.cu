#include "gpu/transpose.h"
#include <assert.h>
#include <stdint.h>

// Tiling the source per block measures faster than one element per thread.
template<typename T>
constexpr int ELEMENTS_PER_BLOCK = (1 << 12) / (int)sizeof(T);

// Transpose data kernel - v0
template<typename T>
__global__ void __launch_bounds__(256, 4)
  transpose_v0_k(T* d_dst,
                 const T* d_src,
                 uint64_t src_size,
                 uint64_t i_offset,
                 uint64_t epoch_elements,
                 uint64_t region_stride,
                 uint8_t rank,
                 const uint64_t* shape,
                 const int64_t* strides)
{
  const uint64_t block_offset = (uint64_t)blockIdx.x * ELEMENTS_PER_BLOCK<T>;
  const uint64_t left = src_size - block_offset;
  const int elements =
    left < ELEMENTS_PER_BLOCK<T> ? (int)left : ELEMENTS_PER_BLOCK<T>;

  for (int i = threadIdx.x; i < elements; i += blockDim.x) {
    uint64_t rest = i_offset + block_offset + i;
    uint64_t out_offset = (rest / epoch_elements) * region_stride;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out_offset += coord * strides[d];
    }

    d_dst[out_offset] = d_src[block_offset + i];
  }
}

struct transpose_args
{
  CUdeviceptr d_dst_beg;
  CUdeviceptr d_src_beg;
  uint64_t src_bytes;
  uint64_t i_offset;
  uint64_t epoch_elements;
  uint64_t region_bytes;
  uint8_t rank;
  const uint64_t* d_shape;
  const int64_t* d_strides;
  CUstream stream;
};

template<typename T>
static void
transpose_launch(const struct transpose_args& a)
{
  const uint64_t src_size = a.src_bytes / sizeof(T);
  if (src_size == 0)
    return;

  assert(a.d_src_beg % sizeof(T) == 0);
  assert(a.region_bytes % sizeof(T) == 0);
  assert(a.epoch_elements > 0);

  const int block_size = 256;
  const unsigned grid_size =
    (unsigned)((src_size + ELEMENTS_PER_BLOCK<T> - 1) / ELEMENTS_PER_BLOCK<T>);

  transpose_v0_k<T><<<grid_size, block_size, 0, (cudaStream_t)a.stream>>>(
    (T*)a.d_dst_beg,
    (T*)a.d_src_beg,
    src_size,
    a.i_offset,
    a.epoch_elements,
    a.region_bytes / sizeof(T),
    a.rank,
    a.d_shape,
    a.d_strides);
}

extern "C" void
transpose(CUdeviceptr d_dst_beg,
          CUdeviceptr d_src_beg,
          uint64_t src_bytes,
          uint8_t bpe,
          uint64_t i_offset,
          uint64_t epoch_elements,
          uint64_t region_bytes,
          uint8_t rank,
          const uint64_t* d_shape,
          const int64_t* d_strides,
          CUstream stream)
{
  const struct transpose_args a = { d_dst_beg, d_src_beg,      src_bytes,
                                    i_offset,  epoch_elements, region_bytes,
                                    rank,      d_shape,        d_strides,
                                    stream };
  switch (bpe) {
    case 1:
      transpose_launch<uint8_t>(a);
      break;
    case 2:
      transpose_launch<uint16_t>(a);
      break;
    case 4:
      transpose_launch<uint32_t>(a);
      break;
    case 8:
      transpose_launch<uint64_t>(a);
      break;
    default:
      assert(!"transpose: unsupported bpe");
      break;
  }
}
