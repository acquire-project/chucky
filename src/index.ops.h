#pragma once

#include <stdint.h>

// Decompose a flat index into coordinates for a row-major array.
// coords[d] = idx % shape[d], then idx /= shape[d], for d=0..rank-1.
void
unravel(int rank,
        const uint64_t* shape,
        uint64_t idx,
        uint64_t* coords);

// Compute the flat output index for a transposed array.
// Iterates d=rank-1..0: coord = idx % shape[d], o += coord * strides[d].
uint64_t
ravel(int rank,
      const uint64_t* shape,
      const int64_t* strides,
      uint64_t idx);

// Same as ravel but with int shape and int strides.
uint64_t
ravel_i32(int rank,
          const int* restrict shape,
          const int* restrict strides,
          uint64_t idx);

// Row-major strides from shape: strides[rank-1]=1, strides[d-1]=shape[d]*strides[d].
void
compute_strides(int rank, const int* shape, int* strides);

// Permute: out[i] = in[p[i]].
void
permute_i32(int n,
            const int* restrict p,
            const int* restrict in,
            int* restrict out);

// Inverse permutation: inv[p[i]] = i.
void
inverse_permutation_i32(int n,
                        const int* restrict p,
                        int* restrict inv);
