#pragma once

#include "defs.limits.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // i_offset counts from the start of the epoch d_dst_beg points at, not from
  // the start of the stream.
  void transpose(CUdeviceptr d_dst_beg,
                 CUdeviceptr d_src_beg,
                 uint64_t src_bytes,
                 uint8_t bpe,
                 uint64_t i_offset,
                 uint64_t epoch_elements,
                 uint64_t region_bytes,
                 uint8_t rank,
                 const uint64_t* d_shape,
                 const int64_t* d_strides,
                 CUstream stream);
#ifdef __cplusplus
}
#endif
