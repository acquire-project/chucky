#pragma once

#include "defs.limits.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Write src_bytes worth of elements to the chunk positions the lifted shape
  // and strides give them. d_src must be aligned to bpe.
  //
  // A destination position repeats every epoch_elements, so the elements past
  // the first epoch go epoch_bytes further along per epoch crossed. d_dst is
  // the region for the epoch holding the first element, and i_offset is that
  // element's position within its epoch.
  void transpose(CUdeviceptr d_dst_beg,
                 CUdeviceptr d_src_beg,
                 uint64_t src_bytes,
                 uint8_t bpe,
                 uint64_t i_offset,
                 uint64_t epoch_elements,
                 uint64_t epoch_bytes,
                 uint8_t rank,
                 const uint64_t* d_shape,
                 const int64_t* d_strides,
                 CUstream stream);
#ifdef __cplusplus
}
#endif
