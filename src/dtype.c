#include "dtype.h"

size_t
dtype_bpe(enum dtype dt)
{
  switch (dt) {
    case dtype_u8:
    case dtype_i8:
      return 1;
    case dtype_u16:
    case dtype_i16:
    case dtype_f16:
      return 2;
    case dtype_u32:
    case dtype_i32:
    case dtype_f32:
      return 4;
    case dtype_u64:
    case dtype_i64:
    case dtype_f64:
      return 8;
  }
  return 0;
}
