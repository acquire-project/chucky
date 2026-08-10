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
  // shape and strides are host arrays. Some run of trailing extents must
  // multiply out to epoch_elements, and every dimension in front of that run
  // must have a zero stride or a single position.
  //
  // Returns 0 once the scatter is queued, or when there is nothing to scatter.
  // Non-zero means nothing was queued. What the device reports while running
  // surfaces on the stream instead.
  // The layout half of what transpose checks, so a stream can refuse a layout
  // it could never scatter when it opens rather than partway through an
  // append. Returns 0 when the scatter can place the layout.
  int transpose_check_layout(uint64_t epoch_elements,
                             uint8_t rank,
                             const uint64_t* shape,
                             const int64_t* strides);

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
