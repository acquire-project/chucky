#include "gpu/blosc.shuffle.h"
#include "gpu/prelude.cuda.h"

__device__ static void
copy_tail(const unsigned char* src,
          unsigned char* dst,
          size_t complete,
          size_t nbytes)
{
  for (size_t out = complete + threadIdx.x; out < nbytes; out += blockDim.x)
    dst[out] = src[out];
}

template<bool byte_shuffle>
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

  if (byte_shuffle) {
    for (size_t out = threadIdx.x; out < complete; out += blockDim.x) {
      const size_t byte = out / nelem;
      const size_t elem = out - byte * nelem;
      chunk_dst[out] = chunk_src[elem * typesize + byte];
    }
    copy_tail(chunk_src, chunk_dst, complete, nbytes);
  } else if (shuffle == CODEC_SHUFFLE_NONE || (nelem & 7) != 0) {
    // C-Blosc 1.x bitshuffle copies a block if its complete-element count is
    // not divisible by eight.
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
    copy_tail(chunk_src, chunk_dst, complete, nbytes);
  }
}

// Compile byte shuffle separately to preserve the copy/bitshuffle kernel
// without adding filter branches to its inner traversal.
template<bool byte_shuffle>
static int
prepare_blocks_async(struct gpu_blosc_frame_layout layout,
                     struct gpu_blosc_input original,
                     void* prepared,
                     size_t block_stride,
                     size_t batch_size,
                     CUstream stream)
{
  const size_t blocks_per_chunk =
    (layout.chunk_bytes + layout.block_bytes - 1) / layout.block_bytes;
  const size_t blocks = batch_size * blocks_per_chunk;
  // Large byte-shuffle blocks need more warps when the batch has few blocks.
  const unsigned threads =
    byte_shuffle && layout.block_bytes >= 256 * 1024 ? 1024 : 256;
  return CUDA_LAUNCH(prepare_block_kernel<byte_shuffle>
                     <<<blocks, threads, 0, (cudaStream_t)stream>>>(
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
  if (layout.shuffle == CODEC_SHUFFLE_BYTE)
    return prepare_blocks_async<true>(
      layout, original, prepared, block_stride, batch_size, stream);
  if (layout.shuffle != CODEC_SHUFFLE_NONE &&
      layout.shuffle != CODEC_SHUFFLE_BIT)
    return 1;
  return prepare_blocks_async<false>(
    layout, original, prepared, block_stride, batch_size, stream);
}
