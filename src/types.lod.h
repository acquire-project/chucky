#pragma once

#include "dtype.h"

#ifdef __cplusplus
extern "C"
{
#endif

  enum lod_reduce_method
  {
    lod_reduce_mean,
    lod_reduce_min,
    lod_reduce_max,
    lod_reduce_median,
    lod_reduce_max_suppressed, // 2nd highest value
    lod_reduce_min_suppressed, // 2nd lowest value
  };

  // Accumulator bytes-per-element for append fold/emit (device memory).
  // Always native type — no widening, to avoid doubling the buffer.
  // Integer mean may lose precision from wrapping; that's acceptable.
  static inline size_t dtype_accum_bpe(enum dtype dt,
                                       enum lod_reduce_method method)
  {
    (void)method;
    return dtype_bpe(dt);
  }

#ifdef __cplusplus
}
#endif
