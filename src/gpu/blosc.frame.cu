#include "gpu/blosc.frame.h"
#include "gpu/prelude.cuda.h"

__device__ static void
put_u32le(unsigned char* dst, unsigned int value)
{
  dst[0] = (unsigned char)value;
  dst[1] = (unsigned char)(value >> 8);
  dst[2] = (unsigned char)(value >> 16);
  dst[3] = (unsigned char)(value >> 24);
}

__global__ static void
finalize_kernel(const unsigned char* original,
                size_t original_stride,
                unsigned char* encoded,
                size_t encoded_stride,
                size_t* sizes,
                size_t chunk_bytes,
                size_t typesize,
                int codec_format,
                int shuffle,
                int force_copy,
                size_t batch_size)
{
  const size_t chunk = blockIdx.x;
  if (chunk >= batch_size)
    return;

  unsigned char* dst = encoded + chunk * encoded_stride;
  __shared__ size_t payload_bytes;
  __shared__ int compressed;
  if (threadIdx.x == 0) {
    payload_bytes = force_copy ? 0 : sizes[chunk];
    compressed =
      !force_copy && chunk_bytes > 8 && payload_bytes < chunk_bytes - 8;
  }
  // All warps must use the payload size before thread 0 replaces sizes[chunk]
  // with the complete frame size. A late read could otherwise select fallback
  // and overwrite a compressed payload near the compression threshold.
  __syncthreads();
  unsigned char flags = (unsigned char)(0x10 | (codec_format << 5));
  if (shuffle == CODEC_SHUFFLE_BYTE)
    flags |= 0x01;
  else if (shuffle == CODEC_SHUFFLE_BIT)
    flags |= 0x04;
  if (!compressed)
    flags |= 0x02;

  if (threadIdx.x == 0) {
    dst[0] = 2; // BLOSC_VERSION_FORMAT
    dst[1] = 1; // LZ4/Zstd payload format version
    dst[2] = flags;
    dst[3] = (unsigned char)typesize;
    put_u32le(dst + 4, (unsigned int)chunk_bytes);
    put_u32le(dst + 8, (unsigned int)chunk_bytes);
    if (compressed) {
      put_u32le(dst + 12,
                (unsigned int)(GPU_BLOSC_PAYLOAD_OFFSET + payload_bytes));
      put_u32le(dst + 16, 20); // absolute bstart
      put_u32le(dst + 20, (unsigned int)payload_bytes);
      sizes[chunk] = GPU_BLOSC_PAYLOAD_OFFSET + payload_bytes;
    } else {
      put_u32le(dst + 12, (unsigned int)(GPU_BLOSC_HEADER_BYTES + chunk_bytes));
      sizes[chunk] = GPU_BLOSC_HEADER_BYTES + chunk_bytes;
    }
  }

  if (!compressed) {
    const unsigned char* src = original + chunk * original_stride;
    for (size_t i = threadIdx.x; i < chunk_bytes; i += blockDim.x)
      dst[GPU_BLOSC_HEADER_BYTES + i] = src[i];
  }
}

extern "C" int
gpu_blosc_finalize_async(enum compression_codec codec,
                         enum codec_shuffle shuffle,
                         size_t typesize,
                         size_t chunk_bytes,
                         const void* original,
                         size_t original_stride,
                         void* encoded,
                         size_t encoded_stride,
                         size_t* encoded_sizes,
                         size_t batch_size,
                         int force_copy,
                         CUstream stream)
{
  const int format = codec == CODEC_BLOSC_LZ4 ? 1 : 4;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  return CUDA_LAUNCH(finalize_kernel<<<batch_size, 256, 0, cuda_stream>>>(
    (const unsigned char*)original,
    original_stride,
    (unsigned char*)encoded,
    encoded_stride,
    encoded_sizes,
    chunk_bytes,
    typesize,
    format,
    shuffle,
    force_copy,
    batch_size));
}
