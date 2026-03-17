#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum lod_dtype
  {
    lod_dtype_u8,
    lod_dtype_u16,
    lod_dtype_u32,
    lod_dtype_u64,
    lod_dtype_i8,
    lod_dtype_i16,
    lod_dtype_i32,
    lod_dtype_i64,
    lod_dtype_f16,
    lod_dtype_f32,
    lod_dtype_f64,
  };

  enum lod_reduce_method
  {
    lod_reduce_mean,
    lod_reduce_min,
    lod_reduce_max,
    lod_reduce_median,
    lod_reduce_max_suppressed, // 2nd highest value
    lod_reduce_min_suppressed, // 2nd lowest value
  };

  static inline size_t lod_dtype_bpe(enum lod_dtype dtype)
  {
    switch (dtype) {
      case lod_dtype_u8:
      case lod_dtype_i8:
        return 1;
      case lod_dtype_u16:
      case lod_dtype_i16:
      case lod_dtype_f16:
        return 2;
      case lod_dtype_u32:
      case lod_dtype_i32:
      case lod_dtype_f32:
        return 4;
      case lod_dtype_u64:
      case lod_dtype_i64:
      case lod_dtype_f64:
        return 8;
    }
    return 0;
  }

  // Accumulator bytes-per-element for dim0 fold/emit (device memory).
  // Always native type — no widening, to avoid doubling the buffer.
  // Integer mean may lose precision from wrapping; that's acceptable.
  static inline size_t lod_accum_bpe(enum lod_dtype dtype,
                                     enum lod_reduce_method method)
  {
    (void)method;
    return lod_dtype_bpe(dtype);
  }

#ifdef __cplusplus
}
#endif
