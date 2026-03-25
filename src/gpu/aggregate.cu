#include "gpu/aggregate.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#pragma nv_diag_suppress 221
#include <cub/cub.cuh>
#pragma nv_diag_default 221
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Kernel 1: permute_sizes_k
//   Thread i (0..M-1) computes P[i] via lifted-stride unravel, then writes:
//     d_permuted_sizes[P[i]] = d_comp_sizes[i]
//     d_perm[i] = P[i]
// ---------------------------------------------------------------------------
__global__ void
permute_sizes_k(const size_t* __restrict__ d_comp_sizes,
                size_t* __restrict__ d_permuted_sizes,
                uint32_t* __restrict__ d_perm,
                uint64_t M,
                uint8_t lifted_rank,
                const uint64_t* __restrict__ shape,
                const int64_t* __restrict__ strides)
{
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  // Unravel chunk index i against lifted_shape, dot with lifted_strides
  uint64_t out = 0;
  uint64_t rest = i;
  for (int d = lifted_rank - 1; d >= 0; --d) {
    const uint64_t coord = rest % shape[d];
    rest /= shape[d];
    out += coord * strides[d];
  }

  uint32_t pi = (uint32_t)out;
  d_permuted_sizes[pi] = d_comp_sizes[i];
  d_perm[i] = pi;
}

// ---------------------------------------------------------------------------
// Kernel 2 (tiny): write the total at d_offsets[C]
// ---------------------------------------------------------------------------
__global__ void
write_total_k(size_t* __restrict__ d_offsets,
              const size_t* __restrict__ d_permuted_sizes,
              uint64_t C)
{
  d_offsets[C] = d_offsets[C - 1] + d_permuted_sizes[C - 1];
}

// ---------------------------------------------------------------------------
// Kernel 3: gather_k
//   Block i copies d_comp_sizes[i] bytes from
//     d_compressed + i * max_comp_chunk_bytes
//   to
//     d_aggregated + d_offsets[d_perm[i]]
// ---------------------------------------------------------------------------
__global__ void
gather_k(const void* __restrict__ d_compressed,
         void* __restrict__ d_aggregated,
         const size_t* __restrict__ d_comp_sizes,
         const size_t* __restrict__ d_offsets,
         const uint32_t* __restrict__ d_perm,
         size_t max_comp_chunk_bytes)
{
  const uint64_t i = blockIdx.x;
  const size_t nbytes = d_comp_sizes[i];
  if (nbytes == 0)
    return;

  const uint8_t* src = (const uint8_t*)d_compressed + i * max_comp_chunk_bytes;
  uint8_t* dst = (uint8_t*)d_aggregated + d_offsets[d_perm[i]];

  for (size_t off = threadIdx.x; off < nbytes; off += blockDim.x)
    dst[off] = src[off];
}

// ---------------------------------------------------------------------------
// Kernel: pad_shard_sizes_k
//   Each thread handles one shard: sums its chunk sizes, computes padding to
//   reach the next page boundary, and adds it to the last chunk's size.
//   This ensures shard-boundary offsets are page-aligned after prefix sum.
// ---------------------------------------------------------------------------
__global__ void
pad_shard_sizes_k(size_t* __restrict__ d_permuted_sizes,
                  uint64_t cps_inner,
                  uint64_t num_shards,
                  size_t page_size)
{
  uint64_t s = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_shards)
    return;

  uint64_t base = s * cps_inner;
  size_t total = 0;
  for (uint64_t i = 0; i < cps_inner; i++)
    total += d_permuted_sizes[base + i];

  size_t aligned = ((total + page_size - 1) / page_size) * page_size;
  size_t padding = aligned - total;
  if (padding > 0)
    d_permuted_sizes[base + cps_inner - 1] += padding;
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
  cub::DeviceScan::ExclusiveSum(
    nullptr, *out_bytes, (size_t*)nullptr, (size_t*)nullptr, (int)count);
  return 0;
}

extern "C" int
aggregate_layout_upload(struct aggregate_layout* layout)
{
  if (layout->lifted_rank == 0)
    return 0;

  const size_t shape_bytes = layout->lifted_rank * sizeof(uint64_t);
  const size_t strides_bytes = layout->lifted_rank * sizeof(int64_t);
  CU(Error, cuMemAlloc((CUdeviceptr*)&layout->d_lifted_shape, shape_bytes));
  CU(Error, cuMemAlloc((CUdeviceptr*)&layout->d_lifted_strides, strides_bytes));
  CU(Error,
     cuMemcpyHtoD(
       (CUdeviceptr)layout->d_lifted_shape, layout->lifted_shape, shape_bytes));
  CU(Error,
     cuMemcpyHtoD((CUdeviceptr)layout->d_lifted_strides,
                  layout->lifted_strides,
                  strides_bytes));
  return 0;

Error:
  return 1;
}

extern "C" void
aggregate_layout_destroy(struct aggregate_layout* layout)
{
  if (!layout)
    return;
  cuMemFree((CUdeviceptr)layout->d_lifted_shape);
  cuMemFree((CUdeviceptr)layout->d_lifted_strides);
  memset(layout, 0, sizeof(*layout));
}

extern "C" void
aggregate_slot_destroy(struct aggregate_slot* slot)
{
  if (!slot)
    return;
  if (slot->ready)
    cuEventDestroy(slot->ready);
  cuMemFree((CUdeviceptr)slot->d_permuted_sizes);
  cuMemFree((CUdeviceptr)slot->d_offsets);
  cuMemFree((CUdeviceptr)slot->d_perm);
  cuMemFree((CUdeviceptr)slot->d_aggregated);
  cuMemFreeHost(slot->h_aggregated);
  cuMemFreeHost(slot->h_offsets);
  cuMemFreeHost(slot->h_permuted_sizes);
  cuMemFree((CUdeviceptr)slot->d_temp);
  memset(slot, 0, sizeof(*slot));
}

extern "C" int
aggregate_by_shard_async(const struct aggregate_layout* layout,
                         void* d_compressed,
                         size_t* d_comp_sizes,
                         struct aggregate_slot* slot,
                         CUstream stream)
{
  const uint64_t M = layout->chunks_per_epoch;
  const uint64_t C = layout->covering_count;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  // Zero permuted_sizes (C+1 entries so the prefix-sum total slot is clean)
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)slot->d_permuted_sizes,
                     0,
                     (C + 1) * sizeof(size_t),
                     stream));

  // Pass 1: permute sizes
  {
    const int block = 256;
    const int grid = (int)((M + block - 1) / block);
    permute_sizes_k<<<grid, block, 0, cuda_stream>>>(d_comp_sizes,
                                                     slot->d_permuted_sizes,
                                                     slot->d_perm,
                                                     M,
                                                     layout->lifted_rank,
                                                     layout->d_lifted_shape,
                                                     layout->d_lifted_strides);
  }

  // D2H real (pre-padding) permuted sizes for shard index
  CU(Error,
     cuMemcpyDtoHAsync(slot->h_permuted_sizes,
                       (CUdeviceptr)slot->d_permuted_sizes,
                       C * sizeof(size_t),
                       stream));

  // Pass 1.5: pad shard sizes for page alignment
  if (layout->page_size > 0) {
    uint64_t num_shards = C / layout->cps_inner;
    const int block = 256;
    const int grid = (int)((num_shards + block - 1) / block);
    pad_shard_sizes_k<<<grid, block, 0, cuda_stream>>>(
      slot->d_permuted_sizes, layout->cps_inner, num_shards, layout->page_size);
  }

  // Pass 2: exclusive prefix sum on C elements
  {
    size_t temp = slot->temp_bytes;
    cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                  temp,
                                  slot->d_permuted_sizes,
                                  slot->d_offsets,
                                  (int)C,
                                  cuda_stream);

    // Write total at d_offsets[C]
    write_total_k<<<1, 1, 0, cuda_stream>>>(
      slot->d_offsets, slot->d_permuted_sizes, C);
  }

  // Pass 3: gather compressed tiles in shard order
  {
    const int block = 256;
    const int grid = (int)M;
    gather_k<<<grid, block, 0, cuda_stream>>>(d_compressed,
                                              slot->d_aggregated,
                                              d_comp_sizes,
                                              slot->d_offsets,
                                              slot->d_perm,
                                              layout->max_comp_chunk_bytes);
  }

  return 0;

Error:
  return 1;
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
// aggregate_batch_slot_init
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_slot_init(struct aggregate_slot* slot,
                          uint64_t batch_chunk_count,
                          uint64_t batch_covering_count,
                          size_t comp_pool_bytes)
{
  uint64_t C = batch_covering_count;
  uint64_t M = batch_chunk_count;

  CHECK(Error, slot);
  memset(slot, 0, sizeof(*slot));

  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_permuted_sizes,
                (C + 1) * sizeof(size_t)));
  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_offsets, (C + 1) * sizeof(size_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_perm, M * sizeof(uint32_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_aggregated, comp_pool_bytes));
  CU(Error, cuMemHostAlloc(&slot->h_aggregated, comp_pool_bytes, 0));
  CU(Error,
     cuMemHostAlloc((void**)&slot->h_offsets, (C + 1) * sizeof(size_t), 0));
  CU(Error,
     cuMemHostAlloc((void**)&slot->h_permuted_sizes, C * sizeof(size_t), 0));

  slot->temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                slot->temp_bytes,
                                slot->d_permuted_sizes,
                                slot->d_offsets,
                                (int)C,
                                (cudaStream_t)0);

  if (slot->temp_bytes > 0)
    CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_temp, slot->temp_bytes));

  CU(Error, cuEventCreate(&slot->ready, CU_EVENT_DEFAULT));

  return 0;

Error:
  aggregate_slot_destroy(slot);
  return 1;
}

// ---------------------------------------------------------------------------
// aggregate_batch_by_shard_async
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_by_shard_async(void* d_compressed,
                               size_t* d_comp_sizes,
                               const uint32_t* d_batch_gather,
                               const uint32_t* d_batch_perm,
                               uint64_t batch_chunk_count,
                               uint64_t batch_covering_count,
                               size_t max_comp_chunk_bytes,
                               const struct aggregate_layout* layout,
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
    permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
      d_comp_sizes, slot->d_permuted_sizes, d_batch_gather, d_batch_perm, N);
  }

  // D2H real (pre-padding) permuted sizes for shard index
  CU(Error,
     cuMemcpyDtoHAsync(slot->h_permuted_sizes,
                       (CUdeviceptr)slot->d_permuted_sizes,
                       C * sizeof(size_t),
                       stream));

  // Pass 1.5: pad shard sizes for page alignment.
  // Pad at shard-group boundaries (cps_inner * batch_count entries per group)
  // so all epochs for one shard are contiguous and can be written in one call.
  if (layout->page_size > 0 && layout->cps_inner > 0) {
    uint64_t num_shards = layout->covering_count / layout->cps_inner;
    uint64_t tps_group = C / num_shards; // cps_inner * batch_count
    const int block = 256;
    const int grid = (int)((num_shards + block - 1) / block);
    pad_shard_sizes_k<<<grid, block, 0, cuda_stream>>>(
      slot->d_permuted_sizes, tps_group, num_shards, layout->page_size);
  }

  // Pass 2: exclusive prefix sum on C elements
  {
    size_t temp = slot->temp_bytes;
    cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                  temp,
                                  slot->d_permuted_sizes,
                                  slot->d_offsets,
                                  (int)C,
                                  cuda_stream);

    write_total_k<<<1, 1, 0, cuda_stream>>>(
      slot->d_offsets, slot->d_permuted_sizes, C);
  }

  // Pass 3: gather compressed tiles using LUTs
  {
    const int block = 256;
    const int grid = (int)N;
    gather_batch_k<<<grid, block, 0, cuda_stream>>>(d_compressed,
                                                    slot->d_aggregated,
                                                    d_comp_sizes,
                                                    slot->d_offsets,
                                                    d_batch_gather,
                                                    d_batch_perm,
                                                    max_comp_chunk_bytes);
  }

  return 0;

Error:
  return 1;
}
