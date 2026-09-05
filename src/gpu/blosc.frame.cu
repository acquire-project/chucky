#include "gpu/blosc.frame.h"
#include "gpu/prelude.cuda.h"

#include <stdint.h>

#pragma nv_diag_suppress 221
#include <cub/block/block_scan.cuh>
#pragma nv_diag_default 221

__device__ static void
put_u32le(uint8_t* dst, uint32_t value)
{
  dst[0] = (uint8_t)value;
  dst[1] = (uint8_t)(value >> 8);
  dst[2] = (uint8_t)(value >> 16);
  dst[3] = (uint8_t)(value >> 24);
}

__global__ static void
block_offsets_kernel(const size_t* block_sizes,
                     size_t* block_offsets,
                     uint8_t* encoded,
                     size_t encoded_stride,
                     size_t* encoded_sizes,
                     size_t chunk_bytes,
                     size_t block_bytes,
                     size_t blocks_per_chunk,
                     size_t typesize,
                     uint8_t codec_format,
                     enum codec_shuffle shuffle,
                     bool force_copy)
{
  using Scan = cub::BlockScan<size_t, 256>;
  __shared__ Scan::TempStorage scan;
  __shared__ size_t end;
  const size_t chunk = blockIdx.x;
  if (threadIdx.x == 0)
    end = GPU_BLOSC_HEADER_BYTES + blocks_per_chunk * sizeof(uint32_t);
  __syncthreads();

  if (!force_copy) {
    for (size_t base = 0; base < blocks_per_chunk; base += blockDim.x) {
      const size_t block = base + threadIdx.x;
      const size_t i = chunk * blocks_per_chunk + block;
      size_t record_bytes = 0;
      if (block < blocks_per_chunk) {
        const size_t remaining = chunk_bytes - block * block_bytes;
        const size_t nbytes = remaining < block_bytes ? remaining : block_bytes;
        const size_t compressed = block_sizes[i];
        const size_t payload =
          compressed && compressed < nbytes ? compressed : nbytes;
        record_bytes = sizeof(uint32_t) + payload;
      }
      size_t offset, total;
      Scan(scan).ExclusiveSum(record_bytes, offset, total);
      if (block < blocks_per_chunk)
        block_offsets[i] = end + offset;
      __syncthreads();
      if (threadIdx.x == 0)
        end += total;
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    uint8_t* dst = encoded + chunk * encoded_stride;
    const bool compressed =
      !force_copy && end < chunk_bytes + GPU_BLOSC_HEADER_BYTES;
    uint8_t flags = (uint8_t)(0x10 | (codec_format << 5));
    if (shuffle == CODEC_SHUFFLE_BYTE)
      flags |= 0x01;
    else if (shuffle == CODEC_SHUFFLE_BIT)
      flags |= 0x04;
    if (!compressed)
      flags |= 0x02;
    const size_t size = compressed ? end : chunk_bytes + GPU_BLOSC_HEADER_BYTES;
    dst[0] = 2;
    dst[1] = 1;
    dst[2] = flags;
    dst[3] = (uint8_t)typesize;
    put_u32le(dst + 4, (uint32_t)chunk_bytes);
    put_u32le(dst + 8, (uint32_t)block_bytes);
    put_u32le(dst + 12, (uint32_t)size);
    encoded_sizes[chunk] = size;
  }
}

__global__ static void
pack_blocks_kernel(const uint8_t* original,
                   size_t original_stride,
                   const void* const* inputs,
                   const uint8_t* block_data,
                   size_t block_stride,
                   const size_t* block_sizes,
                   const size_t* block_offsets,
                   uint8_t* encoded,
                   size_t encoded_stride,
                   size_t chunk_bytes,
                   size_t block_bytes,
                   size_t blocks_per_chunk)
{
  const size_t i = blockIdx.x;
  const size_t chunk = i / blocks_per_chunk;
  const size_t block = i % blocks_per_chunk;
  const size_t block_offset = block * block_bytes;
  const size_t remaining = chunk_bytes - block_offset;
  const size_t nbytes = remaining < block_bytes ? remaining : block_bytes;
  uint8_t* frame = encoded + chunk * encoded_stride;
  const uint8_t* src;
  uint8_t* dst;
  size_t payload;
  if (frame[2] & 0x02) {
    src = original + chunk * original_stride + block_offset;
    dst = frame + GPU_BLOSC_HEADER_BYTES + block_offset;
    payload = nbytes;
  } else {
    const size_t compressed = block_sizes[i];
    const bool raw = compressed == 0 || compressed >= nbytes;
    payload = raw ? nbytes : compressed;
    src = raw ? (const uint8_t*)inputs[i] : block_data + i * block_stride;
    const size_t offset = block_offsets[i];
    if (threadIdx.x == 0) {
      put_u32le(frame + GPU_BLOSC_HEADER_BYTES + block * sizeof(uint32_t),
                (uint32_t)offset);
      put_u32le(frame + offset, (uint32_t)payload);
    }
    dst = frame + offset + sizeof(uint32_t);
  }
  for (size_t j = threadIdx.x; j < payload; j += blockDim.x)
    dst[j] = src[j];
}

extern "C" int
gpu_blosc_pack_async(struct gpu_blosc_frame_layout layout,
                     struct gpu_blosc_input original,
                     struct gpu_blosc_blocks blocks,
                     struct gpu_blosc_output encoded,
                     size_t batch_size,
                     int force_copy,
                     CUstream stream)
{
  if (layout.block_bytes == 0 || layout.block_bytes > layout.chunk_bytes)
    return 1;
  const size_t blocks_per_chunk =
    (layout.chunk_bytes + layout.block_bytes - 1) / layout.block_bytes;
  const uint8_t format = layout.codec == CODEC_BLOSC_LZ4 ? 1 : 4;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  if (CUDA_LAUNCH(block_offsets_kernel<<<batch_size, 256, 0, cuda_stream>>>(
        blocks.sizes,
        blocks.offsets,
        (uint8_t*)encoded.data,
        encoded.stride,
        encoded.sizes,
        layout.chunk_bytes,
        layout.block_bytes,
        blocks_per_chunk,
        layout.typesize,
        format,
        layout.shuffle,
        force_copy)))
    return 1;
  return CUDA_LAUNCH(
    pack_blocks_kernel<<<batch_size * blocks_per_chunk, 256, 0, cuda_stream>>>(
      (const uint8_t*)original.data,
      original.stride,
      blocks.inputs,
      (const uint8_t*)blocks.data,
      blocks.stride,
      blocks.sizes,
      blocks.offsets,
      (uint8_t*)encoded.data,
      encoded.stride,
      layout.chunk_bytes,
      layout.block_bytes,
      blocks_per_chunk));
}
