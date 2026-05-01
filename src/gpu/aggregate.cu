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
// Kernel: compute_bias_k
//   Thread s computes the per-shard offset bias used to land chunks at
//     d_aggregated[shard_base[s] + tail_bytes_prev[s] + within_shard_offset]
//   given that the exclusive prefix sum produced tight (unpadded) cumulative
//   offsets. shard_base[s] = s * shard_capacity is page-aligned by
//   construction. bias[s] = s * shard_capacity + tail_bytes_prev[s] -
//   d_offsets[s * tps_group]
// ---------------------------------------------------------------------------
__global__ void
compute_bias_k(size_t* __restrict__ d_bias,
               const size_t* __restrict__ d_offsets,
               const size_t* __restrict__ d_tail_bytes_prev,
               uint64_t tps_group,
               uint64_t num_shards,
               size_t shard_capacity)
{
  uint64_t s = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_shards)
    return;
  d_bias[s] =
    s * shard_capacity + d_tail_bytes_prev[s] - d_offsets[s * tps_group];
}

// ---------------------------------------------------------------------------
// Kernel: apply_bias_k
//   Thread j adds d_bias[j / tps_group] to d_offsets[j], shifting all chunks
//   in shard s into their final positions in d_aggregated.
// ---------------------------------------------------------------------------
__global__ void
apply_bias_k(size_t* __restrict__ d_offsets,
             const size_t* __restrict__ d_bias,
             uint64_t tps_group,
             uint64_t C)
{
  uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= C)
    return;
  d_offsets[j] += d_bias[j / tps_group];
}

// ---------------------------------------------------------------------------
// Kernel: copy_leading_tail_k
//   Block s copies tail_bytes_prev[s] bytes from d_tail_carry[s * page_size]
//   into d_aggregated[s * shard_capacity], staging the prior batch's ragged
//   tail at the start of this shard's region. No-op when tail length is 0.
// ---------------------------------------------------------------------------
__global__ void
copy_leading_tail_k(void* __restrict__ d_aggregated,
                    const void* __restrict__ d_tail_carry,
                    const size_t* __restrict__ d_tail_bytes_prev,
                    size_t shard_capacity,
                    size_t page_size)
{
  const uint64_t s = blockIdx.x;
  const size_t nbytes = d_tail_bytes_prev[s];
  if (nbytes == 0)
    return;
  const uint8_t* src = (const uint8_t*)d_tail_carry + s * page_size;
  uint8_t* dst = (uint8_t*)d_aggregated + s * shard_capacity;
  for (size_t off = threadIdx.x; off < nbytes; off += blockDim.x)
    dst[off] = src[off];
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
  cuMemFree((CUdeviceptr)slot->d_bias);
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
// aggregate_batch_slot_init
// ---------------------------------------------------------------------------

extern "C" int
aggregate_batch_slot_init(struct aggregate_slot* slot,
                          uint64_t batch_chunk_count,
                          uint64_t batch_covering_count,
                          uint64_t num_shards,
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
  if (num_shards > 0) {
    CU(Error,
       cuMemAlloc((CUdeviceptr*)&slot->d_bias, num_shards * sizeof(size_t)));
  }
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
                               size_t* d_tail_bytes,
                               CUdeviceptr d_tail_carry,
                               CUstream stream)
{
  const uint64_t N = batch_chunk_count;
  const uint64_t C = batch_covering_count;
  const uint64_t num_shards = layout->num_shards;
  const size_t page_size = layout->page_size;
  const size_t shard_capacity = layout->shard_capacity;
  const uint64_t tps_group = num_shards > 0 ? C / num_shards : 0;
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

  // D2H real permuted sizes (host uses these for delivery sizing + next-kick
  // tail bookkeeping).
  CU(Error,
     cuMemcpyDtoHAsync(slot->h_permuted_sizes,
                       (CUdeviceptr)slot->d_permuted_sizes,
                       C * sizeof(size_t),
                       stream));

  // Pass 2: exclusive prefix sum on C elements (tight; no padding inflations).
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

  if (page_size > 0 && num_shards > 0) {
    // Pass 3a: per-shard bias (relocates each shard's chunks into its
    // shard_capacity-sized region with leading-tail headroom).
    {
      const int block = 128;
      const int grid = (int)((num_shards + block - 1) / block);
      compute_bias_k<<<grid, block, 0, cuda_stream>>>(slot->d_bias,
                                                      slot->d_offsets,
                                                      d_tail_bytes,
                                                      tps_group,
                                                      num_shards,
                                                      shard_capacity);
    }
    // Pass 3b: apply bias to each chunk's offset.
    {
      const int block = 256;
      const int grid = (int)((C + block - 1) / block);
      apply_bias_k<<<grid, block, 0, cuda_stream>>>(
        slot->d_offsets, slot->d_bias, tps_group, C);
    }
    // Pass 3c: stage prior batch's ragged tail at the head of each shard's
    // region so chunks pack just past it.
    {
      const int block = 256;
      copy_leading_tail_k<<<(int)num_shards, block, 0, cuda_stream>>>(
        slot->d_aggregated,
        (const void*)d_tail_carry,
        d_tail_bytes,
        shard_capacity,
        page_size);
    }
  }

  // Pass 4: gather compressed tiles using LUTs.
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

  // The next batch's compute_bias_k / copy_leading_tail_k consume
  // d_tail_bytes / d_tail_carry. Both are uploaded by the host after
  // delivery (see flush.d2h_deliver.c) so the values reflect per-shard-
  // generation tails — the GPU has no view of where shard generations
  // begin and end within a batch and so cannot compute them itself.

  return 0;

Error:
  return 1;
}
