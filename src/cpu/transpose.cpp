#include "cpu/transpose.h"

#include "defs.limits.h"
#include "stream/layouts.h"
#include "threadpool/threadpool.h"
#include "util/index.ops.h"

#include <string.h>

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
  uint64_t epoch_elements;
  uint8_t bpe;
};

// The append chunk coordinates have zero strides: the caller selects their
// destination epoch. All remaining coordinates describe one epoch. If their
// strides are row-major, every range within the epoch can be copied directly.
static bool
range_is_contiguous(const struct transpose_ctx* c, uint64_t base, uint64_t n)
{
  uint64_t expected_stride = 1;
  for (int d = c->rank - 1; d >= 0; --d) {
    if (c->strides[d] == 0)
      continue;
    if (c->shape[d] > 1 &&
        (c->strides[d] < 0 || (uint64_t)c->strides[d] != expected_stride))
      return false;
    expected_stride *= c->shape[d];
  }
  const uint64_t epoch_offset = base % c->epoch_elements;
  return expected_stride == c->epoch_elements &&
         n <= c->epoch_elements - epoch_offset;
}

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

  if (range_is_contiguous(c, base, my_n)) {
    memcpy((char*)c->dst + o * c->bpe, my_src, my_n * c->bpe);
    return;
  }

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
              const struct tile_stream_layout* layout,
              struct threadpool* pool)
{
  if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8)
    return 1;
  const uint64_t n = src_bytes / bpe;
  if (n == 0)
    return 0;

  const int rank = layout->lifted_rank;
  const uint64_t* shape = layout->lifted_shape;
  const int64_t* strides = layout->lifted_strides;

  int64_t correction[MAX_RANK];
  for (int d = 0; d < rank - 1; ++d)
    correction[d] = strides[d] - (int64_t)shape[d + 1] * strides[d + 1];

  const int64_t inner_stride = strides[rank - 1];

  struct transpose_ctx c = {
    dst,
    (const char*)src,
    i_offset,
    rank,
    shape,
    strides,
    correction,
    inner_stride,
    layout->epoch_elements,
    bpe,
  };
  // For small appends, dispatching and joining the pool costs more than the
  // scatter. Keep this choice independent of compression parallelism.
  constexpr uint64_t min_parallel_bytes = 64u << 10;
  if (src_bytes < min_parallel_bytes)
    transpose_range(0, n, 0, &c);
  else
    threadpool_for_n(pool, n, transpose_range, &c);

  return 0;
}
