#pragma once

#include "gpu/blosc.frame.h"

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Copy or apply the exact C-Blosc filter into independent aligned block
  // slots. Incomplete-element tails retain their original bytes.
  int gpu_blosc_prepare_blocks_async(struct gpu_blosc_frame_layout layout,
                                     struct gpu_blosc_input original,
                                     void* prepared,
                                     size_t block_stride,
                                     size_t batch_size,
                                     CUstream stream);

#ifdef __cplusplus
}
#endif
