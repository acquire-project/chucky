#pragma once

#include "types.codec.h"
#include <cuda.h>
#include <stddef.h>

enum
{
  GPU_BLOSC_HEADER_BYTES = 16,
  GPU_BLOSC_BLOCK_PREFIX_BYTES = 8,
  GPU_BLOSC_PAYLOAD_OFFSET =
    GPU_BLOSC_HEADER_BYTES + GPU_BLOSC_BLOCK_PREFIX_BYTES,
  GPU_BLOSC_MIN_COMPRESS_BYTES = 128,
};

#ifdef __cplusplus
extern "C"
{
#endif

  // Turn raw nvCOMP payloads at encoded + GPU_BLOSC_PAYLOAD_OFFSET into
  // complete one-block Blosc chunks. Incompressible inputs become whole-chunk
  // MEMCPYED buffers copied from original without a host synchronization.
  int gpu_blosc_finalize_async(enum compression_codec codec,
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
                               CUstream stream);

#ifdef __cplusplus
}
#endif
