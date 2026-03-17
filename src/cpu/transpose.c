#include "transpose.h"

#include "index.ops.h"

#include <stdlib.h>
#include <string.h>

// Compute output element offsets for input indices [beg, beg+n) using the
// vadd algorithm: fill with innermost-stride delta, apply carry corrections
// at dimension boundaries, then prefix-sum into absolute offsets.
static void
scatter_offsets(uint8_t rank,
          const uint64_t* restrict shape,
          const int64_t* restrict strides,
          uint64_t beg,
          uint64_t n,
          int64_t* restrict out)
{
  // Absolute offset for the first element.
  int64_t o = (int64_t)ravel(rank, shape, strides, beg);

  // Fill with innermost stride (constant delta between consecutive elements
  // within the same innermost-dimension run).
  {
    int64_t delta = strides[rank - 1];
    for (uint64_t i = 0; i < n; ++i)
      out[i] = delta;
  }

  // Carry corrections: at each dimension boundary the delta changes by
  //   correction = strides[d-1] - shape[d] * strides[d]
  // The correction is applied at the element where dimension d wraps.
  {
    uint64_t rest = beg;
    uint64_t input_stride = 1;
    uint64_t first_carry = shape[rank - 1] - (beg % shape[rank - 1]);

    for (int d = rank - 1; d > 0; --d) {
      uint64_t e = shape[d];
      rest /= e;
      uint64_t next_input_stride = input_stride * e;
      int64_t correction = strides[d - 1] - (int64_t)e * strides[d];

      for (uint64_t i = first_carry; i < n; i += next_input_stride)
        out[i] += correction;

      if (d > 1) {
        uint64_t r_next = rest % shape[d - 1];
        first_carry += next_input_stride * (shape[d - 1] - r_next - 1);
      }

      input_stride = next_input_stride;
    }
  }

  // Prefix sum: convert deltas to absolute offsets.
  out[0] = o;
  for (uint64_t i = 1; i < n; ++i)
    out[i] += out[i - 1];
}

// Scatter elements from src to dst at computed offsets.
// Dispatch on bpe so the compiler emits native loads/stores.
static void
scatter(void* restrict dst,
        const void* restrict src,
        const int64_t* restrict offsets,
        uint64_t n,
        uint8_t bpe)
{
#define CASE(T)                                                                \
  {                                                                            \
    const T* s = (const T*)src;                                                \
    T* d = (T*)dst;                                                            \
    for (uint64_t i = 0; i < n; ++i)                                           \
      d[offsets[i]] = s[i];                                                    \
  }                                                                            \
  break

  switch (bpe) {
    case 1: CASE(uint8_t);
    case 2: CASE(uint16_t);
    case 4: CASE(uint32_t);
    case 8: CASE(uint64_t);
    default: {
      const char* s = (const char*)src;
      char* d = (char*)dst;
      for (uint64_t i = 0; i < n; ++i)
        memcpy(d + offsets[i] * bpe, s + i * bpe, bpe);
    } break;
  }
#undef CASE
}

int
transpose_cpu(void* dst,
              const void* src,
              uint64_t src_bytes,
              uint8_t bpe,
              uint64_t i_offset,
              uint8_t lifted_rank,
              const uint64_t* lifted_shape,
              const int64_t* lifted_strides)
{
  if (bpe == 0)
    return 0;
  const uint64_t n = src_bytes / bpe;
  if (n == 0)
    return 0;

  int64_t* offsets = (int64_t*)malloc(n * sizeof(int64_t));
  if (!offsets)
    return 1;

  scatter_offsets(lifted_rank, lifted_shape, lifted_strides, i_offset, n, offsets);
  scatter(dst, src, offsets, n, bpe);

  free(offsets);
  return 0;
}
