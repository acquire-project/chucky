#pragma once

#include "types.codec.h"
#include <cuda.h>
#include <stddef.h>

enum
{
  GPU_BLOSC_HEADER_BYTES = 16,
  GPU_BLOSC_MIN_COMPRESS_BYTES = 128,
};

#ifdef __cplusplus
extern "C"
{
#endif

  // Encode one Blosc chunk per input chunk. Returns 0 on success.
  int gpu_blosc_pack_async(enum compression_codec codec,
                           enum codec_shuffle shuffle,
                           size_t typesize,
                           size_t chunk_bytes,
                           size_t block_bytes,
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
                           CUstream stream);

#ifdef __cplusplus
}
#endif
