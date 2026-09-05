#pragma once

#include "types.codec.h"

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Apply the existing exact filters independently to blocks of contiguous
  // chunk data, preserving each block's incomplete-element tail.
  int gpu_blosc_filter_blocks_async(enum codec_shuffle shuffle,
                                    const void* src,
                                    size_t src_stride,
                                    void* dst,
                                    size_t dst_stride,
                                    size_t chunk_bytes,
                                    size_t block_bytes,
                                    size_t typesize,
                                    size_t batch_size,
                                    CUstream stream);

#ifdef __cplusplus
}
#endif
