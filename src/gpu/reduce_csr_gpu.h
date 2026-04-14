#pragma once

#include <cuda.h>
#include <stdint.h>

struct level_dims;

#ifdef __cplusplus
extern "C"
{
#endif

// Device-side CSR reduce LUT for one level transition (l -> l+1).
// Mirrors the host-side reduce_csr in src/lod/reduce_csr.h, but with device
// pointers. Built directly on GPU via lod_build_csr_gpu (in lod.cu).
struct reduce_csr_gpu
{
  CUdeviceptr starts;  // [dst_segment_size + 1] u64, device
  CUdeviceptr indices; // [src_lod_count] u64, device
  uint64_t dst_segment_size;
  uint64_t src_lod_count;
  uint64_t batch_count;
};

// Allocate device memory for starts/indices sized by src_total/dst_total.
// If either is zero, allocates nothing and returns success; build is a no-op.
// On failure leaves *csr in a state safe to pass to reduce_csr_gpu_free.
// Returns 0 on success, non-zero on failure.
int
reduce_csr_gpu_alloc(struct reduce_csr_gpu* csr,
                     uint64_t src_total,
                     uint64_t dst_total);

// Populate starts/indices on device for the src->dst transition.
// Requires reduce_csr_gpu_alloc to have been called with matching sizes.
// Returns 0 on success, non-zero on failure.
int
reduce_csr_gpu_build(struct reduce_csr_gpu* csr,
                     const struct level_dims* src,
                     const struct level_dims* dst,
                     CUstream stream);

// Null-safe. Frees both device pointers and zeros the struct. Safe on zeroed
// struct or after failed alloc.
void
reduce_csr_gpu_free(struct reduce_csr_gpu* csr);

#ifdef __cplusplus
}
#endif
