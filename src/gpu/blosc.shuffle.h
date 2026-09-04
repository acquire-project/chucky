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

  // Exact C-Blosc byte-plane mapping. Complete elements are transposed and any
  // incomplete final element remains byte-for-byte at the end of the block.
  int gpu_blosc_shuffle_async(const void* src,
                              size_t src_stride,
                              void* dst,
                              size_t dst_stride,
                              size_t chunk_bytes,
                              size_t typesize,
                              size_t batch_size,
                              CUstream stream);

  // Exact C-Blosc 1.x bitshuffle mapping. When the number of complete
  // elements is not divisible by eight C-Blosc leaves the entire block
  // unchanged. Otherwise complete elements are bit-transposed and any
  // incomplete final element remains byte-for-byte at the end.
  int gpu_blosc_bitshuffle_async(const void* src,
                                 size_t src_stride,
                                 void* dst,
                                 size_t dst_stride,
                                 size_t chunk_bytes,
                                 size_t typesize,
                                 size_t batch_size,
                                 CUstream stream);

#ifdef __cplusplus
}
#endif
