#include "cpu/transpose.h"

#include "defs.limits.h"
#include "util/index.ops.h"

#include <omp.h>

template<typename T>
static void
scatter_loop(T* dst,
             const void* my_src,
             uint64_t my_n,
             int rank,
             const uint64_t* shape,
             const int64_t* correction,
             int64_t inner_stride,
             uint64_t* coords,
             int64_t o)
{
  const T* s = (const T*)my_src;
  for (uint64_t i = 0; i < my_n; ++i) {
    dst[o] = s[i];
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
}

int
transpose_cpu(void* dst,
              const void* src,
              uint64_t src_bytes,
              uint8_t bpe,
              uint64_t i_offset,
              uint8_t lifted_rank,
              const uint64_t* lifted_shape,
              const int64_t* lifted_strides,
              int nthreads)
{
  if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8)
    return 1;
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

#pragma omp parallel if (n > 1024) num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    uint64_t per = n / (uint64_t)nt;
    uint64_t rem = n % (uint64_t)nt;
    uint64_t my_begin = tid * per + ((uint64_t)tid < rem ? tid : rem);
    uint64_t my_end = my_begin + per + ((uint64_t)tid < rem ? 1 : 0);
    uint64_t my_n = my_end - my_begin;

    if (my_n > 0) {
      uint64_t base = i_offset + my_begin;
      uint64_t coords[MAX_RANK];
      int64_t o =
        (int64_t)transposed_offset(rank, shape, strides, base, coords);
      const void* my_src = (const char*)src + my_begin * bpe;

#define CASE(b, T)                                                             \
  case b:                                                                      \
    scatter_loop((T*)dst,                                                      \
                 my_src,                                                       \
                 my_n,                                                         \
                 rank,                                                         \
                 shape,                                                        \
                 correction,                                                   \
                 inner_stride,                                                 \
                 coords,                                                       \
                 o);                                                           \
    break
      switch (bpe) {
        CASE(1, uint8_t);
        CASE(2, uint16_t);
        CASE(4, uint32_t);
        CASE(8, uint64_t);
      }
#undef CASE
    }
  }

  return 0;
}
