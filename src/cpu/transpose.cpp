#include "cpu/transpose.h"

#include "defs.limits.h"
#include "threadpool/threadpool.h"
#include "util/index.ops.h"

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

struct transpose_ctx
{
  void* dst;
  const char* src;
  uint64_t i_offset;
  int rank;
  const uint64_t* shape;
  const int64_t* strides;
  const int64_t* correction;
  int64_t inner_stride;
  uint8_t bpe;
};

static void
transpose_range(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct transpose_ctx* c = (struct transpose_ctx*)vctx;
  uint64_t my_n = end - beg;
  if (my_n == 0)
    return;
  uint64_t base = c->i_offset + beg;
  uint64_t coords[MAX_RANK];
  int64_t o =
    (int64_t)transposed_offset(c->rank, c->shape, c->strides, base, coords);
  const void* my_src = c->src + beg * c->bpe;

#define CASE(b, T)                                                             \
  case b:                                                                      \
    scatter_loop((T*)c->dst,                                                   \
                 my_src,                                                       \
                 my_n,                                                         \
                 c->rank,                                                      \
                 c->shape,                                                     \
                 c->correction,                                                \
                 c->inner_stride,                                              \
                 coords,                                                       \
                 o);                                                           \
    break
  switch (c->bpe) {
    CASE(1, uint8_t);
    CASE(2, uint16_t);
    CASE(4, uint32_t);
    CASE(8, uint64_t);
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
              const int64_t* lifted_strides,
              struct threadpool* pool)
{
  if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8)
    return 1;
  const uint64_t n = src_bytes / bpe;
  if (n == 0)
    return 0;

  const int rank = lifted_rank;
  const uint64_t* shape = lifted_shape;
  const int64_t* strides = lifted_strides;

  int64_t correction[MAX_RANK];
  for (int d = 0; d < rank - 1; ++d)
    correction[d] = strides[d] - (int64_t)shape[d + 1] * strides[d + 1];

  const int64_t inner_stride = strides[rank - 1];

  struct transpose_ctx c = {
    dst,     (const char*)src, i_offset,     rank, shape,
    strides, correction,       inner_stride, bpe,
  };
  threadpool_for_n(pool, n, transpose_range, &c);

  return 0;
}
