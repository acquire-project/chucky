#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#define restrict __restrict
#endif

  // Decompose a flat index into coordinates for a row-major array.
  // coords[d] = idx % shape[d], then idx /= shape[d], for d=0..rank-1.
  void unravel(int rank, const uint64_t* shape, uint64_t idx, uint64_t* coords);

  // Decompose idx into coords and compute the strided offset in one pass.
  // Iterates d=rank-1..0: coords[d] = idx % shape[d], o += coords[d] *
  // strides[d].
  uint64_t transposed_offset(int rank,
                             const uint64_t* shape,
                             const int64_t* strides,
                             uint64_t idx,
                             uint64_t* coords);

  // Compute the flat output index for a transposed array.
  // Iterates d=rank-1..0: coord = idx % shape[d], o += coord * strides[d].
  uint64_t ravel(int rank,
                 const uint64_t* shape,
                 const int64_t* strides,
                 uint64_t idx);

  // Same as ravel but with int shape and int strides.
  uint64_t ravel_i32(int rank,
                     const int* restrict shape,
                     const int* restrict strides,
                     uint64_t idx);

  // Row-major strides from shape: strides[rank-1]=1,
  // strides[d-1]=shape[d]*strides[d].
  void compute_strides(int rank, const int* shape, int* strides);

  // Permute: out[i] = in[p[i]].
  void permute_i32(int n,
                   const int* restrict p,
                   const int* restrict in,
                   int* restrict out);

  // Inverse permutation: inv[p[i]] = i.
  void inverse_permutation_i32(int n, const int* restrict p, int* restrict inv);

  // Compute lifted strides for chunk decomposition with storage-order
  // permutation. Produces 2*rank strides: lifted_strides[2*i] = chunk stride
  // for dim i, lifted_strides[2*i+1] = element stride for dim i. storage_order
  // may be NULL (identity).
  void compute_lifted_strides(int rank,
                              const uint64_t* chunk_sizes,
                              const uint64_t* chunk_count,
                              const uint8_t* storage_order,
                              int64_t chunk_stride,
                              int64_t* lifted_strides);

  // Maximum element in shape array.
  uint64_t max_shape(int ndim, const uint64_t* shape);

#ifdef __cplusplus
#undef restrict
}
#endif
