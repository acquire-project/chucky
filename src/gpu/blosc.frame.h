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

  struct gpu_blosc_frame_layout
  {
    enum compression_codec codec;
    enum codec_shuffle shuffle;
    size_t typesize;
    size_t chunk_bytes;
    size_t block_bytes;
  };

  struct gpu_blosc_input
  {
    const void* data;
    size_t stride;
  };

  struct gpu_blosc_blocks
  {
    const void* data;
    size_t stride;
    const size_t* sizes;
    size_t* offsets;
  };

  struct gpu_blosc_output
  {
    void* data;
    size_t stride;
    size_t* sizes;
  };

  // Encode one Blosc frame per chunk. Views borrow device buffers; the layout
  // describes active geometry, independent of the buffers' allocation capacity.
  int gpu_blosc_pack_async(struct gpu_blosc_frame_layout layout,
                           struct gpu_blosc_input original,
                           struct gpu_blosc_input filtered,
                           struct gpu_blosc_blocks blocks,
                           struct gpu_blosc_output encoded,
                           size_t batch_size,
                           int force_copy,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
