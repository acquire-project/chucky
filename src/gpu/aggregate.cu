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
  if (slot->host_func_done)
    cuEventDestroy(slot->host_func_done);
  cuMemFree((CUdeviceptr)slot->d_permuted_sizes);
  cuMemFree((CUdeviceptr)slot->d_offsets);
  cuMemFree((CUdeviceptr)slot->d_aggregated);
  cuMemFreeHost(slot->h_aggregated);
  cuMemFreeHost(slot->h_offsets);
  cuMemFreeHost(slot->h_permuted_sizes);
  cuMemFreeHost((void*)slot->h_batch_actual_bytes);
  cuMemFree((CUdeviceptr)slot->d_shard_sum_bytes);
  if (slot->h_shard_sum_bytes)
    cuMemFreeHost((void*)slot->h_shard_sum_bytes);
  if (slot->h_shard_base_offsets_dense)
    cuMemFreeHost((void*)slot->h_shard_base_offsets_dense);
  cuMemFree((CUdeviceptr)slot->d_shard_base_offsets_dense);
  if (slot->h_tail_bytes_next)
    cuMemFreeHost((void*)slot->h_tail_bytes_next);
  cuMemFree((CUdeviceptr)slot->d_temp);
  free(slot->slot_batches);
  memset(slot, 0, sizeof(*slot));
}

// cuLaunchHostFunc body. Runs on the driver thread after the per-shard
// sums D2H has landed. Computes dense per-shard base offsets into
// h_shard_base_offsets_dense for a subsequent H2D upload to feed the
// bias/tail kernels. May not call any CUDA APIs.
//
// dense[s] = slot_cursor + sum_{i<s} (h_tail_bytes_view[i] +
// h_shard_sum_bytes[i])
//
// Prep-1 addition (purely additive at B=1):
//   - h_tail_bytes_next[s] = (h_tail_bytes_view[s] + h_shard_sum_bytes[s]) %
//     page_size — future-state tail bytes for the next in-slot batch. At B=1
//     the orchestrator's post-delivery H2D from h_tail_bytes still overwrites
//     d_tail_bytes; this write is unconsumed but verifiable against host's
//     post-delivery h_tail_bytes for correctness.
//
// slot_cursor is NOT updated by this callback. Prep-2 will factor the slot
// lifecycle so the callback can advance it without breaking B=1 (today the
// next kick reads slot_cursor before resetting it).
//
// Phase 2/2b: batches_per_slot is always 1 and slot_cursor stays 0, so the
// computed values match what s*shard_capacity would have produced modulo
// tail+compression — when h_shard_sum_bytes[s] + tail[s] < shard_capacity,
// dense packing puts shards closer together than uniform packing did.
static void CUDART_CB
aggregate_post_batch_cb(void* userData)
{
  // CUDA guarantees stream operations sequenced before this callback have
  // completed, so the preceding D2Hs to h_batch_actual_bytes /
  // h_shard_sum_bytes are visible without an explicit fence.
  struct aggregate_slot* slot = (struct aggregate_slot*)userData;
  if (!slot || slot->batches_per_slot == 0)
    return;

  const uint64_t n = slot->total_shards_in;
  if (n == 0 || !slot->h_shard_base_offsets_dense || !slot->h_shard_sum_bytes)
    return;
  const size_t* tail = slot->h_tail_bytes_view;
  const size_t page_size = slot->page_size;
  size_t cum = slot->slot_cursor;
  for (uint64_t s = 0; s < n; ++s) {
    ((size_t*)slot->h_shard_base_offsets_dense)[s] = cum;
    const size_t t = tail ? tail[s] : 0;
    const size_t total = t + (size_t)slot->h_shard_sum_bytes[s];
    cum += total;
    if (slot->h_tail_bytes_next)
      ((size_t*)slot->h_tail_bytes_next)[s] =
        (page_size > 0) ? (total % page_size) : 0;
  }
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
                          uint64_t slot_chunk_cap,
                          size_t comp_pool_bytes,
                          uint32_t batches_per_slot_cap,
                          uint64_t max_total_shards)
{
  uint64_t C = slot_chunk_cap;
  (void)batch_chunk_count;

  CHECK(Error, slot);
  CHECK(Error, batches_per_slot_cap >= 1);
  memset(slot, 0, sizeof(*slot));

  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_permuted_sizes, C * sizeof(size_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_offsets, C * sizeof(size_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_aggregated, comp_pool_bytes));
  CU(Error, cuMemHostAlloc(&slot->h_aggregated, comp_pool_bytes, 0));
  CU(Error, cuMemHostAlloc((void**)&slot->h_offsets, C * sizeof(size_t), 0));
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

  slot->batches_per_slot_cap = batches_per_slot_cap;
  slot->slot_batches = (struct batch_slice_entry*)calloc(
    batches_per_slot_cap, sizeof(struct batch_slice_entry));
  CHECK(Error, slot->slot_batches);

  CU(Error,
     cuMemHostAlloc((void**)&slot->h_batch_actual_bytes, sizeof(size_t), 0));
  *slot->h_batch_actual_bytes = 0;

  slot->shard_sum_capacity = max_total_shards;
  if (max_total_shards > 0) {
    CU(Error,
       cuMemAlloc((CUdeviceptr*)&slot->d_shard_sum_bytes,
                  max_total_shards * sizeof(size_t)));
    CU(Error,
       cuMemHostAlloc((void**)&slot->h_shard_sum_bytes,
                      max_total_shards * sizeof(size_t),
                      0));
    CU(Error,
       cuMemHostAlloc((void**)&slot->h_shard_base_offsets_dense,
                      max_total_shards * sizeof(size_t),
                      0));
    CU(Error,
       cuMemAlloc((CUdeviceptr*)&slot->d_shard_base_offsets_dense,
                  max_total_shards * sizeof(size_t)));
    CU(Error,
       cuMemHostAlloc((void**)&slot->h_tail_bytes_next,
                      max_total_shards * sizeof(size_t),
                      0));
    for (uint64_t i = 0; i < max_total_shards; ++i) {
      ((size_t*)slot->h_shard_sum_bytes)[i] = 0;
      ((size_t*)slot->h_shard_base_offsets_dense)[i] = 0;
      ((size_t*)slot->h_tail_bytes_next)[i] = 0;
    }
  }

  CU(Error, cuEventCreate(&slot->ready, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&slot->host_func_done, CU_EVENT_DEFAULT));

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

// compute_shard_sum_bytes_k:
//   Block per shard. Sums d_permuted_sizes over the shard's run
//   [base .. base + tps_group) and writes the per-shard byte total to
//   d_shard_sum_bytes[s]. Called after permute_sizes_batch_k; independent
//   of the prefix scan. The result is consumed by the host callback to
//   compute dense shard_base_offsets for the next batch in the slot.
__global__ void
compute_shard_sum_bytes_k(const size_t* __restrict__ d_permuted_sizes,
                          const uint64_t* __restrict__ d_shard_offsets_base,
                          const uint64_t* __restrict__ d_shard_tps_group,
                          size_t* __restrict__ d_shard_sum_bytes,
                          uint64_t num_shards)
{
  const uint64_t s = blockIdx.x;
  if (s >= num_shards)
    return;
  const uint64_t base = d_shard_offsets_base[s];
  const uint64_t n = d_shard_tps_group[s];
  __shared__ size_t sdata[256];
  size_t local = 0;
  for (uint64_t k = threadIdx.x; k < n; k += blockDim.x)
    local += d_permuted_sizes[base + k];
  sdata[threadIdx.x] = local;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    d_shard_sum_bytes[s] = sdata[0];
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

// rollforward_tail_unified_k:
//   Block per shard. Updates d_tail_bytes in place with the new trailing-tail
//   for the just-aggregated batch and copies those bytes into
//   d_tail_carry + s*page_size. Each shard's region in d_aggregated spans
//   [base, base + tail_in + sum_bytes) where base = d_shard_base_offsets[s],
//   tail_in = prior d_tail_bytes[s], sum_bytes = d_shard_sum_bytes[s]. The
//   new tail = total % page_size; the bytes are the last new_tail bytes of
//   that region.
//
//   Read d_tail_bytes[s] BEFORE the threadIdx.x==0 write; __syncthreads
//   guards the in-block read-then-write. Across blocks each block touches
//   only its own shard, so no inter-block race.
//
//   Used between batches in a slot (macro-aggregation): the next batch's
//   bias/copy_leading_tail kernels read the values this kernel just wrote.
//   For batches_per_slot==1 the host post-delivery H2D into d_tail_bytes /
//   d_tail_carry overwrites the GPU state — still correct, just redundant.
//
//   Correctness assumes no shard finalizes mid-batch: the GPU sees only the
//   batch's per-shard byte total, while host-side delivery zeros the tail on
//   any finalizing run inside the batch. The orchestrator's fit check must
//   guarantee no mid-slot finalization when macro-agg is active.
__global__ void
rollforward_tail_unified_k(const void* __restrict__ d_aggregated,
                           const size_t* __restrict__ d_shard_base_offsets,
                           const size_t* __restrict__ d_shard_sum_bytes,
                           size_t page_size,
                           size_t* __restrict__ d_tail_bytes,
                           void* __restrict__ d_tail_carry,
                           uint64_t num_shards)
{
  const uint64_t s = blockIdx.x;
  if (s >= num_shards)
    return;
  const size_t prev_tail = d_tail_bytes[s];
  const size_t total = prev_tail + d_shard_sum_bytes[s];
  const size_t new_tail = page_size > 0 ? (total % page_size) : 0;
  __syncthreads();
  if (threadIdx.x == 0)
    d_tail_bytes[s] = new_tail;
  if (new_tail == 0)
    return;
  const size_t src_off = d_shard_base_offsets[s] + total - new_tail;
  const uint8_t* src = (const uint8_t*)d_aggregated + src_off;
  uint8_t* dst = (uint8_t*)d_tail_carry + s * page_size;
  for (size_t i = threadIdx.x; i < new_tail; i += blockDim.x)
    dst[i] = src[i];
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
    permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
      d_comp_sizes, slot->d_permuted_sizes, d_batch_gather, d_batch_perm, N);
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

    write_total_k<<<1, 1, 0, cuda_stream>>>(
      slot->d_offsets, slot->d_permuted_sizes, C);
  }

  if (page_size > 0 && num_shards > 0) {
    // Pass 3a: per-shard bias. Anchors each shard's first chunk at
    // s*shard_capacity + tail_in (just past the leading tail) and packs
    // chunks tightly. Multiple generations within one batch are contiguous;
    // intra-batch fresh-gen runs land mid-shard and take the bounce path.
    {
      const int block = 256;
      add_shard_bias_k<<<(int)num_shards, block, 0, cuda_stream>>>(
        slot->d_offsets, d_tail_bytes, tps_group, num_shards, shard_capacity);
    }
    // Pass 3b: stage prior batch's ragged tail at the head of each shard's
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
                              size_t slot_data_base,
                              uint64_t slot_desc_base,
                              const size_t* d_shard_base_offsets,
                              const size_t* d_shard_capacity,
                              const uint64_t* d_shard_tps_group,
                              const uint64_t* d_shard_offsets_base,
                              size_t* d_tail_bytes,
                              CUdeviceptr d_tail_carry,
                              size_t page_size,
                              uint64_t total_shards,
                              CUstream stream)
{
  (void)d_shard_capacity; // kept for symmetry/asserts; gather is offset-driven
  const uint64_t N = total_batch_chunks;
  const uint64_t C = total_batch_covering;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  // Biased descriptor views: each batch in a slot scans its own slice of
  // d_offsets / d_permuted_sizes so prior-batch values don't clobber the
  // prefix-scan output.
  //
  // d_aggregated is NOT biased: the bias kernel applies slot-relative dense
  // offsets (computed by the host callback as slot_cursor + cum), so gather
  // and copy_leading_tail write into d_aggregated at slot-relative positions
  // directly. slot_data_base is consumed by the dense computation, not by a
  // pointer bias.
  (void)slot_data_base;
  size_t* d_perm_sizes_b = slot->d_permuted_sizes + slot_desc_base;
  size_t* d_offsets_b = slot->d_offsets + slot_desc_base;

  // Zero permuted_sizes for this batch's slice (C + nlod entries).
  CU(Error,
     cuMemsetD8Async(
       (CUdeviceptr)d_perm_sizes_b, 0, (C + nlod) * sizeof(size_t), stream));

  // Pass 1: permute sizes using unified LUTs (perm targets already shifted
  // by +lv per LOD inside aggregate_batch_luts_unified; targets are
  // batch-local within d_perm_sizes_b).
  {
    const int block = 256;
    const int grid = (int)((N + block - 1) / block);
    permute_sizes_batch_k<<<grid, block, 0, cuda_stream>>>(
      d_comp_sizes, d_perm_sizes_b, d_batch_gather, d_batch_perm, N);
  }

  // Pass 2: exclusive prefix sum over this batch's slice only (each batch's
  // scan is independent of prior batches' sentinels).
  {
    size_t temp = slot->temp_bytes;
    cub::DeviceScan::ExclusiveSum(slot->d_temp,
                                  temp,
                                  d_perm_sizes_b,
                                  d_offsets_b,
                                  (int)(C + nlod),
                                  cuda_stream);
  }

  // Pass 2b: per-shard compressed-bytes sums for macro-agg bookkeeping.
  // Independent of bias/gather. Result is D2H'd to host, host callback
  // computes dense offsets, H2D stages them to device — all sequenced on
  // this stream so the bias kernel below sees fresh dense offsets.
  if (total_shards > 0 && slot->d_shard_sum_bytes) {
    const int block = 256;
    compute_shard_sum_bytes_k<<<(int)total_shards, block, 0, cuda_stream>>>(
      d_perm_sizes_b,
      d_shard_offsets_base,
      d_shard_tps_group,
      slot->d_shard_sum_bytes,
      total_shards);
  }

  // Small D2H of prefix-sum end (sentinel at C+nlod-1 holds the batch's
  // total compressed bytes — bias kernel does not touch sentinel
  // positions so this value is stable post-scan).
  CU(Error,
     cuMemcpyDtoHAsync((void*)slot->h_batch_actual_bytes,
                       (CUdeviceptr)(d_offsets_b + (C + nlod - 1)),
                       sizeof(size_t),
                       stream));

  // Per-shard sums D2H — consumed by host callback to build dense offsets.
  if (total_shards > 0 && slot->d_shard_sum_bytes && slot->h_shard_sum_bytes) {
    CU(Error,
       cuMemcpyDtoHAsync((void*)slot->h_shard_sum_bytes,
                         (CUdeviceptr)slot->d_shard_sum_bytes,
                         total_shards * sizeof(size_t),
                         stream));
  }

  // Host callback writes dense per-shard base offsets into pinned mem.
  CU(Error, cuLaunchHostFunc(stream, aggregate_post_batch_cb, slot));
  CU(Error, cuEventRecord(slot->host_func_done, stream));

  // Stage dense offsets H2D for downstream bias/tail kernels (2f flips
  // the kernels to read from d_shard_base_offsets_dense; for now this
  // buffer is just populated and not yet consumed).
  if (total_shards > 0 && slot->h_shard_base_offsets_dense &&
      slot->d_shard_base_offsets_dense) {
    CU(Error,
       cuMemcpyHtoDAsync((CUdeviceptr)slot->d_shard_base_offsets_dense,
                         (const void*)slot->h_shard_base_offsets_dense,
                         total_shards * sizeof(size_t),
                         stream));
  }

  if (page_size > 0 && total_shards > 0) {
    // Pass 3a: per-shard bias. Anchors each shard's first chunk at its
    // batch-local base byte offset + tail_in.
    {
      const int block = 256;
      add_shard_bias_unified_k<<<(int)total_shards, block, 0, cuda_stream>>>(
        d_offsets_b,
        d_tail_bytes,
        d_shard_base_offsets,
        d_shard_tps_group,
        d_shard_offsets_base,
        total_shards);
    }
    // Pass 3b: stage prior batch's ragged tail at the head of each shard's
    // region in this batch's slice.
    {
      const int block = 256;
      copy_leading_tail_unified_k<<<(int)total_shards, block, 0, cuda_stream>>>(
        slot->d_aggregated,
        (const void*)d_tail_carry,
        d_tail_bytes,
        d_shard_base_offsets,
        page_size);
    }
  }

  // Pass 4: gather compressed tiles. d_offsets values are slot-relative
  // post-bias, so d_aggregated must be unbiased.
  {
    const int block = 256;
    const int grid = (int)N;
    gather_batch_k<<<grid, block, 0, cuda_stream>>>(d_compressed,
                                                    slot->d_aggregated,
                                                    d_comp_sizes,
                                                    d_offsets_b,
                                                    d_batch_gather,
                                                    d_batch_perm,
                                                    max_comp_chunk_bytes);
  }

  // Pass 5: tail rollforward. Updates d_tail_bytes and d_tail_carry to
  // reflect this batch's trailing sub-page bytes so the next batch's
  // bias/copy_leading_tail kernels see the correct leading-tail. For
  // batches_per_slot==1 the host post-delivery H2D will overwrite these
  // values; for B>1 this kernel is the only source of truth between
  // in-slot batches. Reads from d_aggregated unbiased because the dense
  // shard_base_offsets are slot-relative.
  if (page_size > 0 && total_shards > 0 && slot->d_shard_sum_bytes) {
    const int block = 256;
    rollforward_tail_unified_k<<<(int)total_shards, block, 0, cuda_stream>>>(
      slot->d_aggregated,
      d_shard_base_offsets,
      slot->d_shard_sum_bytes,
      page_size,
      (size_t*)d_tail_bytes,
      (void*)d_tail_carry,
      total_shards);
  }

  return 0;

Error:
  return 1;
}
