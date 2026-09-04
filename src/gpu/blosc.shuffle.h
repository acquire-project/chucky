#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

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

#ifdef __cplusplus
}
#endif
