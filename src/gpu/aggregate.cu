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
// Kernel: add_shard_bias_k
//   For each chunk j in shard s, add bias_s = s * shard_capacity +
//   tail_bytes_prev[s] - d_offsets[s * tps_group] to d_offsets[j]. After
//   the exclusive prefix sum, this lands shard s's first chunk at offset
//   s*shard_capacity + tail_in (just past the leading tail copy) and packs
//   subsequent chunks tightly. Multiple generations within one batch are
//   contiguous; intra-batch fresh-gen runs land mid-shard and intentionally
//   take the bounce path in delivery.
//
//   bias_s is read once into shared memory before any thread writes back to
//   d_offsets — otherwise thread 0's write to d_offsets[base] would clobber
//   the prefix-sum value other threads still need to compute their bias.
// ---------------------------------------------------------------------------
__global__ void
add_shard_bias_k(size_t* __restrict__ d_offsets,
                 const size_t* __restrict__ d_tail_bytes_prev,
                 uint64_t tps_group,
                 uint64_t num_shards,
                 size_t shard_capacity)
{
  const uint64_t s = blockIdx.x;
  if (s >= num_shards)
    return;
  const uint64_t base = s * tps_group;
  __shared__ size_t bias_s;
  if (threadIdx.x == 0)
    bias_s = s * shard_capacity + d_tail_bytes_prev[s] - d_offsets[base];
  __syncthreads();
  for (uint64_t k = threadIdx.x; k < tps_group; k += blockDim.x)
    d_offsets[base + k] += bias_s;
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
                            size_t comp_pool_bytes,
                            size_t* device_bytes,
                            size_t* host_bytes)
{
  const uint64_t C = batch_covering_count;
  size_t temp = 0;
  if (aggregate_cub_temp_bytes(C, &temp))
    return 1;
  *device_bytes = 2 * (C + 1) * sizeof(size_t) // d_permuted_sizes + d_offsets
                  + comp_pool_bytes            // d_aggregated
                  + temp;                      // d_temp
  *host_bytes = comp_pool_bytes                // h_aggregated
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
                          size_t comp_pool_bytes)
{
  uint64_t C = batch_covering_count;

  CHECK(Error, slot);
  memset(slot, 0, sizeof(*slot));

  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_permuted_sizes,
                (C + 1) * sizeof(size_t)));
  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_offsets, (C + 1) * sizeof(size_t)));
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

  return 0;

Error:
  aggregate_slot_destroy(slot);
  return 1;
}

// ---------------------------------------------------------------------------
// Unified kernels (across all LODs)
// ---------------------------------------------------------------------------

// add_shard_bias_unified_k:
//   Block per shard. Reads per-shard parameters (base index, run length,
//   destination base byte offset) from device tables. Computes
//     bias_s = d_shard_base_offsets[s] + d_tail_bytes_prev[s]
//              - d_offsets[d_shard_offsets_base[s]]
//   once into shared memory, then adds it across the shard's run in
//   d_offsets. Same structure as add_shard_bias_k; per-shard parameters
//   replace the uniform tps_group / s*shard_capacity values.
__global__ void
add_shard_bias_unified_k(size_t* __restrict__ d_offsets,
                         const size_t* __restrict__ d_tail_bytes_prev,
                         const size_t* __restrict__ d_shard_base_offsets,
                         const uint64_t* __restrict__ d_shard_tps_group,
                         const uint64_t* __restrict__ d_shard_offsets_base,
                         uint64_t num_shards)
{
  const uint64_t s = blockIdx.x;
  if (s >= num_shards)
    return;
  const uint64_t base = d_shard_offsets_base[s];
  const uint64_t tps_group = d_shard_tps_group[s];
  __shared__ size_t bias_s;
  if (threadIdx.x == 0)
    bias_s = d_shard_base_offsets[s] + d_tail_bytes_prev[s] - d_offsets[base];
  __syncthreads();
  for (uint64_t k = threadIdx.x; k < tps_group; k += blockDim.x)
    d_offsets[base + k] += bias_s;
}

// copy_leading_tail_unified_k:
//   Block per shard. Copies d_tail_bytes_prev[s] bytes from
//   d_tail_carry + s*page_size into d_aggregated + d_shard_base_offsets[s].
//   Same structure as copy_leading_tail_k; page_size remains uniform across
//   shards today, while the destination base is now per-shard.
__global__ void
copy_leading_tail_unified_k(void* __restrict__ d_aggregated,
                            const void* __restrict__ d_tail_carry,
                            const size_t* __restrict__ d_tail_bytes_prev,
                            const size_t* __restrict__ d_shard_base_offsets,
                            size_t page_size)
{
  const uint64_t s = blockIdx.x;
  const size_t nbytes = d_tail_bytes_prev[s];
  if (nbytes == 0)
    return;
  const uint8_t* src = (const uint8_t*)d_tail_carry + s * page_size;
  uint8_t* dst = (uint8_t*)d_aggregated + d_shard_base_offsets[s];
  for (size_t off = threadIdx.x; off < nbytes; off += blockDim.x)
    dst[off] = src[off];
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
    CHECK_SILENT(
      Error,
      CUDA_LAUNCH(permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
        d_comp_sizes,
        slot->d_permuted_sizes,
        d_batch_gather,
        d_batch_perm,
        N)) == 0);
  }

  // Pass 2: exclusive prefix sum on C elements (tight; no padding inflations).
  {
    size_t temp = slot->temp_bytes;
    cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                  temp,
                                  slot->d_permuted_sizes,
                                  slot->d_offsets,
                                  (int)C,
                                  cuda_stream);

    CHECK_SILENT(Error,
                 CUDA_LAUNCH(write_total_k<<<1, 1, 0, cuda_stream>>>(
                   slot->d_offsets, slot->d_permuted_sizes, C)) == 0);
  }

  if (page_size > 0 && num_shards > 0) {
    // Pass 3a: per-shard bias. Anchors each shard's first chunk at
    // s*shard_capacity + tail_in (just past the leading tail) and packs
    // chunks tightly. Multiple generations within one batch are contiguous;
    // intra-batch fresh-gen runs land mid-shard and take the bounce path.
    {
      const int block = 256;
      CHECK_SILENT(
        Error,
        CUDA_LAUNCH(
          add_shard_bias_k<<<(int)num_shards, block, 0, cuda_stream>>>(
            slot->d_offsets,
            d_tail_bytes,
            tps_group,
            num_shards,
            shard_capacity)) == 0);
    }
    // Pass 3b: stage prior batch's ragged tail at the head of each shard's
    // region so chunks pack just past it.
    {
      const int block = 256;
      CHECK_SILENT(
        Error,
        CUDA_LAUNCH(
          copy_leading_tail_k<<<(int)num_shards, block, 0, cuda_stream>>>(
            slot->d_aggregated,
            (const void*)d_tail_carry,
            d_tail_bytes,
            shard_capacity,
            page_size)) == 0);
    }
  }

  // Pass 4: gather compressed tiles using LUTs.
  {
    const int block = 256;
    const int grid = (int)N;
    CHECK_SILENT(Error,
                 CUDA_LAUNCH(gather_batch_k<<<grid, block, 0, cuda_stream>>>(
                   d_compressed,
                   slot->d_aggregated,
                   d_comp_sizes,
                   slot->d_offsets,
                   d_batch_gather,
                   d_batch_perm,
                   max_comp_chunk_bytes)) == 0);
  }

  // d_tail_bytes / d_tail_carry are uploaded by host post-delivery
  // (flush.d2h_deliver.c).

  return 0;

Error:
  return 1;
}

// ---------------------------------------------------------------------------
// aggregate_batch_unified_async
//   Single dispatch across all LODs. The kernel launches are identical in
//   structure to aggregate_batch_by_shard_async, but per-LOD/per-shard
//   parameters come from device-side tables built host-side at kick time.
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
                              const size_t* d_shard_base_offsets,
                              const size_t* d_shard_capacity,
                              const uint64_t* d_shard_tps_group,
                              const uint64_t* d_shard_offsets_base,
                              const size_t* d_tail_bytes,
                              CUdeviceptr d_tail_carry,
                              size_t page_size,
                              uint64_t total_shards,
                              CUstream stream)
{
  (void)d_shard_capacity; // kept for symmetry/asserts; gather is offset-driven
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
    CHECK_SILENT(
      Error,
      CUDA_LAUNCH(permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
        d_comp_sizes,
        slot->d_permuted_sizes,
        d_batch_gather,
        d_batch_perm,
        N)) == 0);
  }

  // Pass 2: single exclusive prefix sum across the unified covering range
  // plus one sentinel slot per LOD. The scan writes every LOD's
  // tail-sentinel position; no separate write_total fixup needed.
  {
    size_t temp = slot->temp_bytes;
    cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                  temp,
                                  slot->d_permuted_sizes,
                                  slot->d_offsets,
                                  (int)(C + nlod),
                                  cuda_stream);
  }

  if (page_size > 0 && total_shards > 0) {
    // Pass 3a: per-shard bias. Anchors each shard's first chunk at its
    // base byte offset + tail_in, packs subsequent chunks tightly.
    {
      const int block = 256;
      CHECK_SILENT(Error,
                   CUDA_LAUNCH(add_shard_bias_unified_k<<<(int)total_shards,
                                                          block,
                                                          0,
                                                          cuda_stream>>>(
                     slot->d_offsets,
                     d_tail_bytes,
                     d_shard_base_offsets,
                     d_shard_tps_group,
                     d_shard_offsets_base,
                     total_shards)) == 0);
    }
    // Pass 3b: stage prior batch's ragged tail at the head of each shard's
    // region so chunks pack just past it.
    {
      const int block = 256;
      CHECK_SILENT(Error,
                   CUDA_LAUNCH(copy_leading_tail_unified_k<<<(int)total_shards,
                                                             block,
                                                             0,
                                                             cuda_stream>>>(
                     slot->d_aggregated,
                     (const void*)d_tail_carry,
                     d_tail_bytes,
                     d_shard_base_offsets,
                     page_size)) == 0);
    }
  }

  // Pass 4: gather compressed tiles using unified LUTs.
  {
    const int block = 256;
    const int grid = (int)N;
    CHECK_SILENT(Error,
                 CUDA_LAUNCH(gather_batch_k<<<grid, block, 0, cuda_stream>>>(
                   d_compressed,
                   slot->d_aggregated,
                   d_comp_sizes,
                   slot->d_offsets,
                   d_batch_gather,
                   d_batch_perm,
                   max_comp_chunk_bytes)) == 0);
  }

  // d_tail_bytes / d_tail_carry are uploaded by host post-delivery
  // (flush.d2h_deliver.c).

  return 0;

Error:
  return 1;
}
