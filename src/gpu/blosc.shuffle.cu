#include "gpu/blosc.shuffle.h"
#include "gpu/prelude.cuda.h"

__global__ static void
shuffle_kernel(const unsigned char* src,
               size_t src_stride,
               unsigned char* dst,
               size_t dst_stride,
               size_t chunk_bytes,
               size_t block_bytes,
               size_t blocks_per_chunk,
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
  dst[(chunk * blocks_per_chunk + chunk_out / block_bytes) * dst_stride + out] =
    src[chunk * src_stride + block_offset + in];
}

__global__ static void
prepare_block_kernel(const unsigned char* src,
                     size_t src_stride,
                     unsigned char* dst,
                     size_t dst_stride,
                     size_t chunk_bytes,
                     size_t block_bytes,
                     size_t blocks_per_chunk,
                     size_t typesize,
                     size_t batch_size,
                     enum codec_shuffle shuffle)
{
  const size_t chunk = blockIdx.x / blocks_per_chunk;
  const size_t block_offset = (blockIdx.x % blocks_per_chunk) * block_bytes;
  if (chunk >= batch_size)
    return;

  const size_t remaining = chunk_bytes - block_offset;
  const size_t nbytes = remaining < block_bytes ? remaining : block_bytes;
  const size_t nelem = nbytes / typesize;
  const size_t complete = nelem * typesize;
  const unsigned char* chunk_src = src + chunk * src_stride + block_offset;
  unsigned char* chunk_dst = dst + (size_t)blockIdx.x * dst_stride;

  // C-Blosc 1.x bitshuffle is all-or-nothing for complete elements: if their
  // count is not divisible by eight it copies the whole block unchanged.
  if (shuffle == CODEC_SHUFFLE_NONE || (nelem & 7) != 0) {
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
gpu_blosc_prepare_blocks_async(struct gpu_blosc_frame_layout layout,
                               struct gpu_blosc_input original,
                               void* prepared,
                               size_t block_stride,
                               size_t batch_size,
                               CUstream stream)
{
  if (!layout.chunk_bytes || !layout.block_bytes || !layout.typesize ||
      !batch_size || block_stride < layout.block_bytes)
    return 1;
  const size_t blocks_per_chunk =
    (layout.chunk_bytes + layout.block_bytes - 1) / layout.block_bytes;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  if (layout.shuffle == CODEC_SHUFFLE_BYTE) {
    const size_t total = batch_size * layout.chunk_bytes;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    return CUDA_LAUNCH(shuffle_kernel<<<blocks, 256, 0, cuda_stream>>>(
      (const unsigned char*)original.data,
      original.stride,
      (unsigned char*)prepared,
      block_stride,
      layout.chunk_bytes,
      layout.block_bytes,
      blocks_per_chunk,
      layout.typesize,
      batch_size));
  }
  if (layout.shuffle != CODEC_SHUFFLE_NONE &&
      layout.shuffle != CODEC_SHUFFLE_BIT)
    return 1;
  const size_t blocks = batch_size * blocks_per_chunk;
  return CUDA_LAUNCH(prepare_block_kernel<<<blocks, 256, 0, cuda_stream>>>(
    (const unsigned char*)original.data,
    original.stride,
    (unsigned char*)prepared,
    block_stride,
    layout.chunk_bytes,
    layout.block_bytes,
    blocks_per_chunk,
    layout.typesize,
    batch_size,
    layout.shuffle));
}
