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
  //
  // shape and strides are host arrays; they travel to the device in the launch
  // parameters. The trailing dimensions must multiply out to epoch_elements,
  // and each dimension in front of those must have either a zero stride or an
  // extent of one — that is what lets the kernel treat the leftover of the
  // index decomposition as the epoch number. A shape that does not meet this
  // returns non-zero without launching.
  //
  // Returns 0 on success, non-zero on error.
  int transpose(CUdeviceptr d_dst_beg,
                CUdeviceptr d_src_beg,
                uint64_t src_bytes,
                uint8_t bpe,
                uint64_t i_offset,
                uint64_t epoch_elements,
                uint64_t region_bytes,
                uint8_t rank,
                const uint64_t* shape,
                const int64_t* strides,
                CUstream stream);
#ifdef __cplusplus
}
#endif
