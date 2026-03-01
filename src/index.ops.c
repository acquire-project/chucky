#include "index.ops.h"

void
unravel(int rank,
        const uint64_t* shape,
        uint64_t idx,
        uint64_t* coords)
{
  for (int d = 0; d < rank; ++d) {
    coords[d] = idx % shape[d];
    idx /= shape[d];
  }
}

uint64_t
ravel(int rank,
      const uint64_t* shape,
      const int64_t* strides,
      uint64_t idx)
{
  uint64_t o = 0;
  uint64_t rest = idx;
  for (int d = rank - 1; d >= 0; --d) {
    uint64_t coord = rest % shape[d];
    rest /= shape[d];
    o += coord * (uint64_t)strides[d];
  }
  return o;
}

uint64_t
ravel_i32(int rank,
          const int* restrict shape,
          const int* restrict strides,
          uint64_t idx)
{
  uint64_t o = 0;
  uint64_t rest = idx;
  for (int d = rank - 1; d >= 0; --d) {
    const int r = rest % shape[d];
    o += r * strides[d];
    rest /= shape[d];
  }
  return o;
}

void
compute_strides(int rank, const int* shape, int* strides)
{
  strides[rank - 1] = 1;
  for (int d = rank - 1; d > 0; --d) {
    strides[d - 1] = shape[d] * strides[d];
  }
}

void
permute_i32(int n,
            const int* restrict p,
            const int* restrict in,
            int* restrict out)
{
  for (int i = 0; i < n; ++i) {
    out[i] = in[p[i]];
  }
}

void
inverse_permutation_i32(int n, const int* restrict p, int* restrict inv)
{
  for (int i = 0; i < n; ++i) {
    inv[p[i]] = i;
  }
}
