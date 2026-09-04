#include "gpu/blosc.shuffle.h"
#include "gpu/prelude.cuda.h"

__global__ static void
shuffle_kernel(const unsigned char* src,
               size_t src_stride,
               unsigned char* dst,
               size_t dst_stride,
               size_t chunk_bytes,
               size_t block_bytes,
               size_t typesize,
               size_t batch_size)
{
  size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch_size * chunk_bytes;
  if (p >= total)
    return;

  const size_t chunk = p / chunk_bytes;
  const size_t chunk_out = p - chunk * chunk_bytes;
  const size_t block_offset = chunk_out / block_bytes * block_bytes;
  const size_t remaining = chunk_bytes - block_offset;
  const size_t nbytes = remaining < block_bytes ? remaining : block_bytes;
  const size_t out = chunk_out - block_offset;
  const size_t nelem = nbytes / typesize;
  const size_t complete = nelem * typesize;
  size_t in = out;
  if (out < complete) {
    const size_t byte = out / nelem;
    const size_t elem = out - byte * nelem;
    in = elem * typesize + byte;
  }
  dst[chunk * dst_stride + chunk_out] =
    src[chunk * src_stride + block_offset + in];
}

__global__ static void
bitshuffle_kernel(const unsigned char* src,
                  size_t src_stride,
                  unsigned char* dst,
                  size_t dst_stride,
                  size_t chunk_bytes,
                  size_t block_bytes,
                  size_t typesize,
                  size_t batch_size)
{
  const size_t blocks_per_chunk = (chunk_bytes + block_bytes - 1) / block_bytes;
  const size_t chunk = blockIdx.x / blocks_per_chunk;
  const size_t block_offset = (blockIdx.x % blocks_per_chunk) * block_bytes;
  if (chunk >= batch_size)
    return;

  const size_t remaining = chunk_bytes - block_offset;
  const size_t nbytes = remaining < block_bytes ? remaining : block_bytes;
  const size_t nelem = nbytes / typesize;
  const size_t complete = nelem * typesize;
  const unsigned char* chunk_src = src + chunk * src_stride + block_offset;
  unsigned char* chunk_dst = dst + chunk * dst_stride + block_offset;

  // C-Blosc 1.x bitshuffle is all-or-nothing for complete elements: if their
  // count is not divisible by eight it copies the whole block unchanged.
  if ((nelem & 7) != 0) {
    for (size_t out = threadIdx.x; out < nbytes; out += blockDim.x)
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
    for (size_t out = complete + threadIdx.x; out < nbytes; out += blockDim.x)
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
  return gpu_blosc_filter_blocks_async(CODEC_SHUFFLE_BYTE,
                                       src,
                                       src_stride,
                                       dst,
                                       dst_stride,
                                       chunk_bytes,
                                       chunk_bytes,
                                       typesize,
                                       batch_size,
                                       stream);
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
  return gpu_blosc_filter_blocks_async(CODEC_SHUFFLE_BIT,
                                       src,
                                       src_stride,
                                       dst,
                                       dst_stride,
                                       chunk_bytes,
                                       chunk_bytes,
                                       typesize,
                                       batch_size,
                                       stream);
}

extern "C" int
gpu_blosc_filter_blocks_async(enum codec_shuffle shuffle,
                              const void* src,
                              size_t src_stride,
                              void* dst,
                              size_t dst_stride,
                              size_t chunk_bytes,
                              size_t block_bytes,
                              size_t typesize,
                              size_t batch_size,
                              CUstream stream)
{
  if (!chunk_bytes || !block_bytes || !typesize || !batch_size)
    return 1;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  if (shuffle == CODEC_SHUFFLE_BYTE) {
    const size_t total = batch_size * chunk_bytes;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    return CUDA_LAUNCH(
      shuffle_kernel<<<blocks, 256, 0, cuda_stream>>>((const unsigned char*)src,
                                                      src_stride,
                                                      (unsigned char*)dst,
                                                      dst_stride,
                                                      chunk_bytes,
                                                      block_bytes,
                                                      typesize,
                                                      batch_size));
  }
  if (shuffle != CODEC_SHUFFLE_BIT)
    return 1;
  const size_t blocks =
    batch_size * ((chunk_bytes + block_bytes - 1) / block_bytes);
  return CUDA_LAUNCH(bitshuffle_kernel<<<blocks, 256, 0, cuda_stream>>>(
    (const unsigned char*)src,
    src_stride,
    (unsigned char*)dst,
    dst_stride,
    chunk_bytes,
    block_bytes,
    typesize,
    batch_size));
}
