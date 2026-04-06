#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum dtype
  {
    dtype_u8,
    dtype_u16,
    dtype_u32,
    dtype_u64,
    dtype_i8,
    dtype_i16,
    dtype_i32,
    dtype_i64,
    dtype_f16,
    dtype_f32,
    dtype_f64,
  };

  size_t dtype_bpe(enum dtype dt);

#ifdef __cplusplus
}
#endif
