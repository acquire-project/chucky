#include "gpu/blosc.shuffle.h"
#include "gpu/prelude.cuda.h"

__global__ static void
shuffle_kernel(const unsigned char* src,
               size_t src_stride,
               unsigned char* dst,
               size_t dst_stride,
               size_t chunk_bytes,
               size_t typesize,
               size_t batch_size)
{
  size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch_size * chunk_bytes;
  if (p >= total)
    return;

  const size_t chunk = p / chunk_bytes;
  const size_t out = p - chunk * chunk_bytes;
  const size_t nelem = chunk_bytes / typesize;
  const size_t complete = nelem * typesize;
  size_t in = out;
  if (out < complete) {
    const size_t byte = out / nelem;
    const size_t elem = out - byte * nelem;
    in = elem * typesize + byte;
  }
  dst[chunk * dst_stride + out] = src[chunk * src_stride + in];
}

__global__ static void
bitshuffle_kernel(const unsigned char* src,
                  size_t src_stride,
                  unsigned char* dst,
                  size_t dst_stride,
                  size_t chunk_bytes,
                  size_t typesize,
                  size_t batch_size)
{
  const size_t chunk = blockIdx.x;
  if (chunk >= batch_size)
    return;

  const size_t nelem = chunk_bytes / typesize;
  const size_t complete = nelem * typesize;
  const unsigned char* chunk_src = src + chunk * src_stride;
  unsigned char* chunk_dst = dst + chunk * dst_stride;

  // C-Blosc 1.x bitshuffle is all-or-nothing for complete elements: if their
  // count is not divisible by eight it copies the whole block unchanged.
  if ((nelem & 7) != 0) {
    for (size_t out = threadIdx.x; out < chunk_bytes; out += blockDim.x)
      chunk_dst[out] = chunk_src[out];
  } else {
    const size_t groups = nelem / 8;
    for (size_t out = threadIdx.x; out < complete; out += blockDim.x) {
      const size_t row = out / groups;
      const size_t group = out - row * groups;
      const size_t byte = row / 8;
      const unsigned bit = (unsigned)(row & 7);
      unsigned char packed = 0;
      for (unsigned elem = 0; elem < 8; ++elem) {
        const unsigned char value =
          chunk_src[(group * 8 + elem) * typesize + byte];
        packed |= (unsigned char)(((value >> bit) & 1u) << elem);
      }
      chunk_dst[out] = packed;
    }
    for (size_t out = complete + threadIdx.x; out < chunk_bytes;
         out += blockDim.x)
      chunk_dst[out] = chunk_src[out];
  }
}

extern "C" int
gpu_blosc_shuffle_async(const void* src,
                        size_t src_stride,
                        void* dst,
                        size_t dst_stride,
                        size_t chunk_bytes,
                        size_t typesize,
                        size_t batch_size,
                        CUstream stream)
{
  const size_t total = batch_size * chunk_bytes;
  const unsigned blocks = (unsigned)((total + 255) / 256);
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  return CUDA_LAUNCH(
    shuffle_kernel<<<blocks, 256, 0, cuda_stream>>>((const unsigned char*)src,
                                                    src_stride,
                                                    (unsigned char*)dst,
                                                    dst_stride,
                                                    chunk_bytes,
                                                    typesize,
                                                    batch_size));
}

extern "C" int
gpu_blosc_bitshuffle_async(const void* src,
                           size_t src_stride,
                           void* dst,
                           size_t dst_stride,
                           size_t chunk_bytes,
                           size_t typesize,
                           size_t batch_size,
                           CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  return CUDA_LAUNCH(bitshuffle_kernel<<<batch_size, 256, 0, cuda_stream>>>(
    (const unsigned char*)src,
    src_stride,
    (unsigned char*)dst,
    dst_stride,
    chunk_bytes,
    typesize,
    batch_size));
}
