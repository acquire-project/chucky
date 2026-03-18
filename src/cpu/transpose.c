#include "transpose.h"

#include "defs.limits.h"
#include "index.ops.h"

#include <omp.h>
#include <string.h>

// Fused odometer transpose: zero allocations.
// Each element is scattered using an incremental odometer that tracks
// coordinates and a running output offset on the stack.

#define ODOMETER_LOOP(T)                                                       \
  {                                                                            \
    const T* s = (const T*)my_src;                                             \
    T* d = (T*)dst;                                                            \
    for (uint64_t i = 0; i < my_n; ++i) {                                     \
      d[o] = s[i];                                                             \
      o += inner_stride;                                                       \
      if (++coords[rank - 1] >= shape[rank - 1]) {                            \
        coords[rank - 1] = 0;                                                 \
        for (int dd = rank - 2; dd >= 0; --dd) {                              \
          o += correction[dd];                                                 \
          if (++coords[dd] < shape[dd])                                        \
            break;                                                             \
          coords[dd] = 0;                                                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
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

  const int rank = lifted_rank;
  const uint64_t* shape = lifted_shape;
  const int64_t* strides = lifted_strides;

  // Precompute carry corrections (shared, read-only across threads).
  // When dimension d+1 wraps, the running offset changes by correction[d].
  int64_t correction[MAX_RANK];
  for (int d = 0; d < rank - 1; ++d)
    correction[d] = strides[d] - (int64_t)shape[d + 1] * strides[d + 1];

  const int64_t inner_stride = strides[rank - 1];

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    uint64_t per = n / (uint64_t)nthreads;
    uint64_t rem = n % (uint64_t)nthreads;
    uint64_t my_begin = tid * per + ((uint64_t)tid < rem ? tid : rem);
    uint64_t my_end = my_begin + per + ((uint64_t)tid < rem ? 1 : 0);
    uint64_t my_n = my_end - my_begin;

    if (my_n > 0) {
      // Init odometer: decompose starting index into coordinates (rank-1 is
      // fastest, matching ravel's convention).
      uint64_t base = i_offset + my_begin;
      uint64_t coords[MAX_RANK];
      {
        uint64_t rest = base;
        for (int d = rank - 1; d >= 0; --d) {
          coords[d] = rest % shape[d];
          rest /= shape[d];
        }
      }
      int64_t o = (int64_t)ravel(rank, shape, strides, base);
      const void* my_src = (const char*)src + my_begin * bpe;

      switch (bpe) {
        case 1: ODOMETER_LOOP(uint8_t); break;
        case 2: ODOMETER_LOOP(uint16_t); break;
        case 4: ODOMETER_LOOP(uint32_t); break;
        case 8: ODOMETER_LOOP(uint64_t); break;
        default: {
          const char* s = (const char*)my_src;
          char* d = (char*)dst;
          for (uint64_t i = 0; i < my_n; ++i) {
            memcpy(d + o * bpe, s + i * bpe, bpe);
            o += inner_stride;
            if (++coords[rank - 1] >= shape[rank - 1]) {
              coords[rank - 1] = 0;
              for (int dd = rank - 2; dd >= 0; --dd) {
                o += correction[dd];
                if (++coords[dd] < shape[dd])
                  break;
                coords[dd] = 0;
              }
            }
          }
        } break;
      }
    }
  }

  return 0;
}
