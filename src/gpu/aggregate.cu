#include "gpu/aggregate.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#pragma nv_diag_suppress 221
#include <cub/cub.cuh>
#pragma nv_diag_default 221
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Kernel: write_total_k — write the total at d_offsets[C]
// ---------------------------------------------------------------------------
__global__ void
write_total_k(size_t* __restrict__ d_offsets,
              const size_t* __restrict__ d_permuted_sizes,
              uint64_t C)
{
  d_offsets[C] = d_offsets[C - 1] + d_permuted_sizes[C - 1];
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

extern "C" int
aggregate_cub_temp_bytes(uint64_t count, size_t* out_bytes)
{
  if (!out_bytes)
    return 1;
  if (count == 0) {
    *out_bytes = 0;
    return 0;
  }
  *out_bytes = 0;
  (void)cub::DeviceScan::ExclusiveSum(
    nullptr, *out_bytes, (size_t*)nullptr, (size_t*)nullptr, (int)count);
  return 0;
}

extern "C" void
aggregate_slot_destroy(struct aggregate_slot* slot)
{
  if (!slot)
    return;
  cuMemFree((CUdeviceptr)slot->d_permuted_sizes);
  cuMemFree((CUdeviceptr)slot->d_offsets);
  cuMemFree((CUdeviceptr)slot->d_aggregated);
  cuMemFreeHost(slot->h_aggregated);
  cuMemFreeHost(slot->h_offsets);
  cuMemFreeHost(slot->h_permuted_sizes);
  cuMemFree((CUdeviceptr)slot->d_temp);
  memset(slot, 0, sizeof(*slot));
}

// ---------------------------------------------------------------------------
// Batch aggregate kernels (LUT-based)
// ---------------------------------------------------------------------------

// permute_sizes_batch_k:
//   Thread i reads comp size from d_comp_sizes[gather[i]], writes to
//   d_permuted_sizes[perm[i]].
__global__ void
permute_sizes_batch_k(const size_t* __restrict__ d_comp_sizes,
                      size_t* __restrict__ d_permuted_sizes,
                      const uint32_t* __restrict__ d_gather,
                      const uint32_t* __restrict__ d_perm,
                      uint64_t N)
{
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  d_permuted_sizes[d_perm[i]] = d_comp_sizes[d_gather[i]];
}

// gather_batch_k:
//   Block i copies compressed chunk gather[i] to output position perm[i].
__global__ void
gather_batch_k(const void* __restrict__ d_compressed,
               void* __restrict__ d_aggregated,
               const size_t* __restrict__ d_comp_sizes,
               const size_t* __restrict__ d_offsets,
               const uint32_t* __restrict__ d_gather,
               const uint32_t* __restrict__ d_perm,
               size_t max_comp_chunk_bytes)
{
  const uint64_t i = blockIdx.x;
  const uint32_t src_idx = d_gather[i];
  const size_t nbytes = d_comp_sizes[src_idx];
  if (nbytes == 0)
    return;

  const uint8_t* src =
    (const uint8_t*)d_compressed + (uint64_t)src_idx * max_comp_chunk_bytes;
  uint8_t* dst = (uint8_t*)d_aggregated + d_offsets[d_perm[i]];

  for (size_t off = threadIdx.x; off < nbytes; off += blockDim.x)
    dst[off] = src[off];
}

// ---------------------------------------------------------------------------
// Sizing (must mirror aggregate_batch_slot_init exactly —
// tile_stream_gpu_memory_estimate sums these)
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_slot_memory(uint64_t batch_covering_count,
                            size_t device_data_bytes,
                            size_t host_data_bytes,
                            size_t* device_bytes,
                            size_t* host_bytes)
{
  const uint64_t C = batch_covering_count;
  size_t temp = 0;
  if (aggregate_cub_temp_bytes(C, &temp))
    return 1;
  *device_bytes = 2 * (C + 1) * sizeof(size_t) // d_permuted_sizes + d_offsets
                  + device_data_bytes          // d_aggregated
                  + temp;                      // d_temp
  *host_bytes = host_data_bytes                // h_aggregated
                + (C + 1) * sizeof(size_t)     // h_offsets
                + C * sizeof(size_t);          // h_permuted_sizes
  return 0;
}

// ---------------------------------------------------------------------------
// aggregate_batch_slot_init
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_slot_init(struct aggregate_slot* slot,
                          uint64_t batch_covering_count,
                          size_t device_data_bytes,
                          size_t host_data_bytes)
{
  uint64_t C = batch_covering_count;

  CHECK(Error, slot);
  memset(slot, 0, sizeof(*slot));

  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_permuted_sizes,
                (C + 1) * sizeof(size_t)));
  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_offsets, (C + 1) * sizeof(size_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_aggregated, device_data_bytes));
  CU(Error, cuMemHostAlloc(&slot->h_aggregated, host_data_bytes, 0));
  CU(Error,
     cuMemHostAlloc((void**)&slot->h_offsets, (C + 1) * sizeof(size_t), 0));
  CU(Error,
     cuMemHostAlloc((void**)&slot->h_permuted_sizes, C * sizeof(size_t), 0));

  slot->temp_bytes = 0;
  CUDA_CALL_OR(Error,
               cub::DeviceScan::ExclusiveSum(nullptr,
                                             slot->temp_bytes,
                                             slot->d_permuted_sizes,
                                             slot->d_offsets,
                                             (int)C,
                                             (cudaStream_t)0));

  if (slot->temp_bytes > 0)
    CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_temp, slot->temp_bytes));

  slot->device_capacity = device_data_bytes;
  slot->host_capacity = host_data_bytes;

  return 0;

Error:
  aggregate_slot_destroy(slot);
  return 1;
}

// ---------------------------------------------------------------------------
// aggregate_batch_by_shard_async
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_by_shard_async(const void* d_compressed,
                               size_t* d_comp_sizes,
                               const uint32_t* d_batch_gather,
                               const uint32_t* d_batch_perm,
                               uint64_t batch_chunk_count,
                               uint64_t batch_covering_count,
                               size_t max_comp_chunk_bytes,
                               struct aggregate_slot* slot,
                               CUstream stream)
{
  const uint64_t N = batch_chunk_count;
  const uint64_t C = batch_covering_count;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  // Zero permuted_sizes (C+1 entries)
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)slot->d_permuted_sizes,
                     0,
                     (C + 1) * sizeof(size_t),
                     stream));

  // Pass 1: permute sizes using LUTs
  {
    const int block = 256;
    const int grid = (int)((N + block - 1) / block);
    CUDA_LAUNCH_OR(
      Error,
      permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
        d_comp_sizes, slot->d_permuted_sizes, d_batch_gather, d_batch_perm, N));
  }

  // Pass 2: exclusive prefix sum on C elements (tight; no padding inflations).
  {
    size_t temp = slot->temp_bytes;
    // A scan that never ran leaves the last batch's offsets, and the passes
    // below would pack this batch's chunks at those.
    CUDA_CALL_OR(Error,
                 cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                               temp,
                                               slot->d_permuted_sizes,
                                               slot->d_offsets,
                                               (int)C,
                                               cuda_stream));

    CUDA_LAUNCH_OR(Error,
                   write_total_k<<<1, 1, 0, cuda_stream>>>(
                     slot->d_offsets, slot->d_permuted_sizes, C));
  }

  // Pass 3: gather compressed tiles using compact absolute offsets.
  {
    const int block = 256;
    const int grid = (int)N;
    CUDA_LAUNCH_OR(
      Error,
      gather_batch_k<<<grid, block, 0, cuda_stream>>>(d_compressed,
                                                      slot->d_aggregated,
                                                      d_comp_sizes,
                                                      slot->d_offsets,
                                                      d_batch_gather,
                                                      d_batch_perm,
                                                      max_comp_chunk_bytes));
  }

  return 0;

Error:
  return 1;
}

// ---------------------------------------------------------------------------
// aggregate_batch_unified_async
//   Single compact dispatch across all LODs. Unified gather/permutation LUTs
//   encode the shard-major destination order without per-shard device tables.
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_unified_async(const void* d_compressed,
                              size_t* d_comp_sizes,
                              const uint32_t* d_batch_gather,
                              const uint32_t* d_batch_perm,
                              uint64_t total_batch_chunks,
                              uint64_t total_batch_covering,
                              uint8_t nlod,
                              size_t max_comp_chunk_bytes,
                              struct aggregate_slot* slot,
                              CUstream stream)
{
  const uint64_t N = total_batch_chunks;
  const uint64_t C = total_batch_covering;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  // Zero permuted_sizes (C + nlod entries: covering plus one sentinel per LOD)
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)slot->d_permuted_sizes,
                     0,
                     (C + nlod) * sizeof(size_t),
                     stream));

  // Pass 1: permute sizes using unified LUTs (perm targets already shifted
  // by +lv per LOD inside aggregate_batch_luts_unified).
  {
    const int block = 256;
    const int grid = (int)((N + block - 1) / block);
    CUDA_LAUNCH_OR(
      Error,
      permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
        d_comp_sizes, slot->d_permuted_sizes, d_batch_gather, d_batch_perm, N));
  }

  // Pass 2: single exclusive prefix sum across the unified covering range
  // plus one sentinel slot per LOD. The scan writes every LOD's
  // tail-sentinel position; no separate write_total fixup needed.
  {
    size_t temp = slot->temp_bytes;
    CUDA_CALL_OR(Error,
                 cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                               temp,
                                               slot->d_permuted_sizes,
                                               slot->d_offsets,
                                               (int)(C + nlod),
                                               cuda_stream));
  }

  // Pass 3: gather compressed tiles using unified compact offsets.
  {
    const int block = 256;
    const int grid = (int)N;
    CUDA_LAUNCH_OR(
      Error,
      gather_batch_k<<<grid, block, 0, cuda_stream>>>(d_compressed,
                                                      slot->d_aggregated,
                                                      d_comp_sizes,
                                                      slot->d_offsets,
                                                      d_batch_gather,
                                                      d_batch_perm,
                                                      max_comp_chunk_bytes));
  }

  return 0;

Error:
  return 1;
}
