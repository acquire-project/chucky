#pragma once

#include "defs.limits.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void transpose(CUdeviceptr d_dst_beg,
                 CUdeviceptr d_src_beg,
                 uint64_t src_bytes,
                 uint8_t bpe,
                 uint64_t i_offset,
                 uint8_t rank,
                 const uint64_t* d_shape,
                 const int64_t* d_strides,
                 CUstream stream);
#ifdef __cplusplus
}
#endif
