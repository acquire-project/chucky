#include "gpu/blosc.frame.h"
#include "gpu/prelude.cuda.h"

#pragma nv_diag_suppress 221
#include <cub/block/block_scan.cuh>
#pragma nv_diag_default 221

__device__ static void
put_u32le(unsigned char* dst, unsigned int value)
{
  dst[0] = (unsigned char)value;
  dst[1] = (unsigned char)(value >> 8);
  dst[2] = (unsigned char)(value >> 16);
  dst[3] = (unsigned char)(value >> 24);
}

// Like shard aggregation, compute offsets first and gather variable-size
// records second. The scan is segmented by chunk so no host offsets or extra
// multidimensional reorder are needed. Tile the scan for arbitrarily many
// blocks per chunk (including more than one CUDA block's worth of records).
__global__ static void
block_offsets_kernel(const size_t* block_sizes,
                     size_t* block_offsets,
                     unsigned char* encoded,
                     size_t encoded_stride,
                     size_t* encoded_sizes,
                     size_t chunk_bytes,
                     size_t block_bytes,
                     size_t blocks_per_chunk,
                     size_t typesize,
                     int codec_format,
                     int shuffle,
                     int force_copy)
{
  using Scan = cub::BlockScan<size_t, 256>;
  __shared__ Scan::TempStorage scan;
  __shared__ size_t end;
  const size_t chunk = blockIdx.x;
  if (threadIdx.x == 0)
    end = GPU_BLOSC_HEADER_BYTES + blocks_per_chunk * sizeof(unsigned int);
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
        // Equal-size records are Blosc's raw-block marker. A failed/empty
        // payload also has a lossless raw representation in filtered input.
        const size_t payload =
          compressed && compressed < nbytes ? compressed : nbytes;
        record_bytes = sizeof(unsigned int) + payload;
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
    unsigned char* dst = encoded + chunk * encoded_stride;
    const int compressed =
      !force_copy && end < chunk_bytes + GPU_BLOSC_HEADER_BYTES;
    unsigned char flags = (unsigned char)(0x10 | (codec_format << 5));
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
    dst[3] = (unsigned char)typesize;
    put_u32le(dst + 4, (unsigned int)chunk_bytes);
    put_u32le(dst + 8, (unsigned int)block_bytes);
    put_u32le(dst + 12, (unsigned int)size);
    encoded_sizes[chunk] = size;
  }
}

__global__ static void
pack_blocks_kernel(const unsigned char* original,
                   size_t original_stride,
                   const unsigned char* filtered,
                   size_t filtered_stride,
                   const unsigned char* block_data,
                   size_t block_stride,
                   const size_t* block_sizes,
                   const size_t* block_offsets,
                   unsigned char* encoded,
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
  unsigned char* frame = encoded + chunk * encoded_stride;
  const unsigned char* src;
  unsigned char* dst;
  size_t payload;
  if (frame[2] & 0x02) {
    // Whole-chunk fallback bypasses the block table and all filters.
    src = original + chunk * original_stride + block_offset;
    dst = frame + GPU_BLOSC_HEADER_BYTES + block_offset;
    payload = nbytes;
  } else {
    const size_t compressed = block_sizes[i];
    const int raw = compressed == 0 || compressed >= nbytes;
    payload = raw ? nbytes : compressed;
    src = raw ? filtered + chunk * filtered_stride + block_offset
              : block_data + i * block_stride;
    const size_t offset = block_offsets[i];
    if (threadIdx.x == 0) {
      put_u32le(frame + GPU_BLOSC_HEADER_BYTES + block * sizeof(unsigned int),
                (unsigned int)offset);
      put_u32le(frame + offset, (unsigned int)payload);
    }
    dst = frame + offset + sizeof(unsigned int);
  }
  for (size_t j = threadIdx.x; j < payload; j += blockDim.x)
    dst[j] = src[j];
}

extern "C" int
gpu_blosc_pack_async(enum compression_codec codec,
                     enum codec_shuffle shuffle,
                     size_t typesize,
                     size_t chunk_bytes,
                     const void* original,
                     size_t original_stride,
                     const void* filtered,
                     size_t filtered_stride,
                     const void* block_data,
                     size_t block_stride,
                     const size_t* block_sizes,
                     size_t* block_offsets,
                     void* encoded,
                     size_t encoded_stride,
                     size_t* encoded_sizes,
                     size_t batch_size,
                     int force_copy,
                     CUstream stream)
{
  const size_t block_bytes = chunk_bytes < GPU_BLOSC_BLOCK_BYTES
                               ? chunk_bytes
                               : (size_t)GPU_BLOSC_BLOCK_BYTES;
  const size_t blocks_per_chunk = (chunk_bytes + block_bytes - 1) / block_bytes;
  const int format = codec == CODEC_BLOSC_LZ4 ? 1 : 4;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  if (CUDA_LAUNCH(block_offsets_kernel<<<batch_size, 256, 0, cuda_stream>>>(
        block_sizes,
        block_offsets,
        (unsigned char*)encoded,
        encoded_stride,
        encoded_sizes,
        chunk_bytes,
        block_bytes,
        blocks_per_chunk,
        typesize,
        format,
        shuffle,
        force_copy)))
    return 1;
  return CUDA_LAUNCH(
    pack_blocks_kernel<<<batch_size * blocks_per_chunk, 256, 0, cuda_stream>>>(
      (const unsigned char*)original,
      original_stride,
      (const unsigned char*)filtered,
      filtered_stride,
      (const unsigned char*)block_data,
      block_stride,
      block_sizes,
      block_offsets,
      (unsigned char*)encoded,
      encoded_stride,
      chunk_bytes,
      block_bytes,
      blocks_per_chunk));
}
