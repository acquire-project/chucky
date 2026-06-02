#include "gpu/aggregate.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#pragma nv_diag_suppress 221
#include <cub/cub.cuh>
#pragma nv_diag_default 221
#include <stdlib.h>
#include <string.h>

__global__ void
write_total_k(size_t* __restrict__ d_offsets,
              const size_t* __restrict__ d_permuted_sizes,
              uint64_t C)
{
  d_offsets[C] = d_offsets[C - 1] + d_permuted_sizes[C - 1];
}

// bias_s is read into shared mem before any thread writes d_offsets:
// thread 0's write to d_offsets[base] would otherwise clobber the
// prefix-sum value other threads still need to compute their bias.
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
output_slot_close_reset(struct aggregate_slot* slot)
{
  if (!slot)
    return;
  slot->slot_cursor = 0;
  slot->slot_desc_cursor = 0;
  slot->batches_per_slot = 0;
  memset(slot->stacked_n_active, 0, sizeof(slot->stacked_n_active));
  if (slot->d_runtime)
    cuMemsetD8(
      (CUdeviceptr)slot->d_runtime, 0, sizeof(struct slot_runtime_state));
}

extern "C" int
slot_can_fit(const struct aggregate_slot* slot, size_t next_batch_bytes)
{
  if (!slot)
    return 0;
  if (slot->batches_per_slot >= slot->batches_per_slot_cap)
    return 0;
  if (slot->slot_capacity_bytes == 0)
    return 1; // not-yet-initialized slot: don't block
  return (slot->slot_cursor + next_batch_bytes) <= slot->slot_capacity_bytes;
}

extern "C" int
no_shard_finalizes(uint8_t nlod,
                   const uint64_t* epoch_in_shard,
                   const uint32_t* n_active_next,
                   const uint64_t* cps_append)
{
  if (!epoch_in_shard || !n_active_next || !cps_append)
    return 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    if (n_active_next[lv] == 0)
      continue;
    if (epoch_in_shard[lv] + (uint64_t)n_active_next[lv] >= cps_append[lv])
      return 0;
  }
  return 1;
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
  cuMemFree((CUdeviceptr)slot->d_shard_sum_bytes);
  if (slot->h_shard_base_offsets_dense)
    cuMemFreeHost((void*)slot->h_shard_base_offsets_dense);
  cuMemFree((CUdeviceptr)slot->d_shard_base_offsets_dense);
  cuMemFree((CUdeviceptr)slot->d_temp);
  cuMemFree((CUdeviceptr)slot->d_runtime);
  free(slot->slot_batches);
  memset(slot, 0, sizeof(*slot));
}

static void CUDART_CB
aggregate_post_batch_write_cb(void* userData)
{
  struct aggregate_write_cb_args* args =
    (struct aggregate_write_cb_args*)userData;
  volatile struct aggregate_write_desc* r = args->h_write_desc;
  const int target_idx = r->target_slot_idx;
  struct aggregate_slot* target = args->slots[target_idx];
  const uint32_t bi = r->batch_idx_in_slot;
  const uint8_t nlod = args->nlod;

  struct batch_slice_entry* be = &target->slot_batches[bi];
  be->data_base_offset = r->data_base_offset;
  be->desc_base_offset = r->desc_base_offset;
  be->nlod = nlod;
  memcpy(be->per_lod_lods,
         args->layout.lods,
         (size_t)nlod * sizeof(struct lod_segment));

  target->slot_cursor = r->data_base_offset + r->actual_bytes;
  target->slot_desc_cursor =
    r->desc_base_offset + args->layout.total_batch_covering + (uint64_t)nlod;
  target->batches_per_slot = bi + 1;

  // bi==0 means this is the first batch in the target slot (fresh or after
  // swap-reset on device). Mirror that reset on the host accumulator.
  if (bi == 0)
    memset(target->stacked_n_active, 0, sizeof(target->stacked_n_active));
  for (uint8_t lv = 0; lv < nlod; ++lv)
    target->stacked_n_active[lv] += args->per_lod_n_active[lv];
}

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

__global__ void
write_desc_from_reservation_k(
  struct aggregate_write_desc* desc,
  struct slot_dev_ptrs target,
  const struct aggregate_append_measurement* __restrict__ d_measurement,
  struct aggregate_slot_reservation reservation)
{
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  const struct aggregate_append_measurement measurement = *d_measurement;
  desc->data_base_offset = reservation.data_base;
  desc->desc_base_offset = reservation.desc_base;
  desc->actual_bytes = measurement.data_bytes;
  desc->batch_idx_in_slot = reservation.batch_index;
  desc->target_slot_idx = reservation.slot;
  desc->close_prior_slot_idx = reservation.close_slot;
  desc->target_d_aggregated = target.d_aggregated;
  desc->target_d_offsets = target.d_offsets;
  desc->target_d_permuted_sizes = target.d_permuted_sizes;
  desc->target_d_shard_base_offsets_dense = target.d_shard_base_offsets_dense;
  desc->measurement = measurement;

  if (target.d_runtime) {
    target.d_runtime->cursor = reservation.data_base + measurement.data_bytes;
    target.d_runtime->desc_cursor =
      reservation.desc_base + measurement.desc_entries;
    target.d_runtime->batches_per_slot = reservation.batch_index + 1;
  }
}

extern "C" int
write_desc_from_reservation_launch(
  struct aggregate_write_desc* desc,
  struct slot_dev_ptrs target,
  const struct aggregate_append_measurement* d_measurement,
  const struct aggregate_slot_reservation* reservation,
  CUstream stream)
{
  if (!reservation)
    return 1;
  write_desc_from_reservation_k<<<1, 1, 0, (cudaStream_t)stream>>>(
    desc, target, d_measurement, *reservation);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

__global__ void
aggregate_measurement_k(struct aggregate_append_measurement* d_measurement,
                        const size_t* __restrict__ d_actual_bytes_ptr,
                        const size_t* __restrict__ d_tail_sum_bytes_ptr,
                        int include_tail_sum,
                        uint64_t desc_entries,
                        int closes_after_append,
                        int tail_rollforward_blocked)
{
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  size_t data_bytes = *d_actual_bytes_ptr;
  if (include_tail_sum)
    data_bytes += *d_tail_sum_bytes_ptr;
  *d_measurement = (struct aggregate_append_measurement){
    .data_bytes = data_bytes,
    .desc_entries = desc_entries,
    .closes_after_append = closes_after_append ? 1 : 0,
    .tail_rollforward_blocked = tail_rollforward_blocked ? 1 : 0,
  };
}

extern "C" int
aggregate_measurement_launch(struct aggregate_append_measurement* d_measurement,
                             const size_t* d_actual_bytes_ptr,
                             const size_t* d_tail_sum_bytes_ptr,
                             int include_tail_sum,
                             uint64_t desc_entries,
                             int closes_after_append,
                             int tail_rollforward_blocked,
                             CUstream stream)
{
  aggregate_measurement_k<<<1, 1, 0, (cudaStream_t)stream>>>(
    d_measurement,
    d_actual_bytes_ptr,
    d_tail_sum_bytes_ptr,
    include_tail_sum,
    desc_entries,
    closes_after_append,
    tail_rollforward_blocked);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

// Serial: total_shards is small (<= a few hundred); a parallel scan would
// cost more in launch + sync than the per-shard work saved.
__global__ void
dense_offsets_k(struct aggregate_write_desc* aggregate_write_desc,
                const size_t* __restrict__ d_shard_sum_bytes,
                const size_t* __restrict__ d_tail_bytes,
                uint64_t total_shards)
{
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  size_t* dense_slice =
    aggregate_write_desc->target_d_shard_base_offsets_dense +
    (uint64_t)aggregate_write_desc->batch_idx_in_slot * total_shards;
  size_t cum = aggregate_write_desc->data_base_offset;
  for (uint64_t s = 0; s < total_shards; ++s) {
    dense_slice[s] = cum;
    cum += d_tail_bytes[s] + d_shard_sum_bytes[s];
  }
}

extern "C" int
dense_offsets_launch(struct aggregate_write_desc* aggregate_write_desc,
                     const size_t* d_shard_sum_bytes,
                     const size_t* d_tail_bytes,
                     uint64_t total_shards,
                     CUstream stream)
{
  dense_offsets_k<<<1, 1, 0, (cudaStream_t)stream>>>(
    aggregate_write_desc, d_shard_sum_bytes, d_tail_bytes, total_shards);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

// Copy [0..count) from per-stage temps into target slot at desc_base_offset.
// Offsets are biased by data_base_offset so batch 2+ chunks land at the
// correct slot-absolute positions without overwriting prior batches.
// (add_shard_bias_unified_k's formula is invariant under this shift; only
// kicks at page_size==0 rely on this bias alone.)
__global__ void
copy_to_slot_k(
  const struct aggregate_write_desc* __restrict__ aggregate_write_desc,
  const size_t* __restrict__ d_temp_offsets,
  const size_t* __restrict__ d_temp_perm_sizes,
  uint64_t count)
{
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  __shared__ size_t* dst_offsets;
  __shared__ size_t* dst_perm_sizes;
  __shared__ size_t data_base;
  if (threadIdx.x == 0) {
    dst_offsets = aggregate_write_desc->target_d_offsets +
                  aggregate_write_desc->desc_base_offset;
    dst_perm_sizes = aggregate_write_desc->target_d_permuted_sizes +
                     aggregate_write_desc->desc_base_offset;
    data_base = aggregate_write_desc->data_base_offset;
  }
  __syncthreads();
  dst_offsets[i] = d_temp_offsets[i] + data_base;
  dst_perm_sizes[i] = d_temp_perm_sizes[i];
}

extern "C" int
copy_to_slot_launch(const struct aggregate_write_desc* aggregate_write_desc,
                    const size_t* d_temp_offsets,
                    const size_t* d_temp_perm_sizes,
                    uint64_t count,
                    CUstream stream)
{
  const int block = 256;
  const int grid = (int)((count + block - 1) / block);
  copy_to_slot_k<<<grid, block, 0, (cudaStream_t)stream>>>(
    aggregate_write_desc, d_temp_offsets, d_temp_perm_sizes, count);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

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

// Descriptor variant: reads target pointers from the host-ledger write
// descriptor (the legacy
// aggregate_batch_by_shard_async still uses gather_batch_k).
__global__ void
gather_batch_write_desc_k(
  const struct aggregate_write_desc* __restrict__ aggregate_write_desc,
  const void* __restrict__ d_compressed,
  const size_t* __restrict__ d_comp_sizes,
  const uint32_t* __restrict__ d_gather,
  const uint32_t* __restrict__ d_perm,
  size_t max_comp_chunk_bytes)
{
  const uint64_t i = blockIdx.x;
  const uint32_t src_idx = d_gather[i];
  const size_t nbytes = d_comp_sizes[src_idx];
  if (nbytes == 0)
    return;
  __shared__ void* d_aggregated;
  __shared__ const size_t* d_offsets;
  if (threadIdx.x == 0) {
    d_aggregated = aggregate_write_desc->target_d_aggregated;
    d_offsets = aggregate_write_desc->target_d_offsets +
                aggregate_write_desc->desc_base_offset;
  }
  __syncthreads();

  const uint8_t* src =
    (const uint8_t*)d_compressed + (uint64_t)src_idx * max_comp_chunk_bytes;
  uint8_t* dst = (uint8_t*)d_aggregated + d_offsets[d_perm[i]];

  for (size_t off = threadIdx.x; off < nbytes; off += blockDim.x)
    dst[off] = src[off];
}

extern "C" int
aggregate_batch_slot_init(struct aggregate_slot* slot,
                          uint64_t slot_chunk_cap,
                          size_t comp_pool_bytes,
                          uint32_t batches_per_slot_cap,
                          uint64_t max_total_shards)
{
  uint64_t C = slot_chunk_cap;

  CHECK(Error, slot);
  CHECK(Error, batches_per_slot_cap >= 1);
  memset(slot, 0, sizeof(*slot));

  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_permuted_sizes, C * sizeof(size_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_offsets, C * sizeof(size_t)));
  CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_aggregated, comp_pool_bytes));
  slot->slot_capacity_bytes = comp_pool_bytes;
  CU(Error, cuMemHostAlloc(&slot->h_aggregated, comp_pool_bytes, 0));
  CU(Error, cuMemHostAlloc((void**)&slot->h_offsets, C * sizeof(size_t), 0));
  CU(Error,
     cuMemHostAlloc((void**)&slot->h_permuted_sizes, C * sizeof(size_t), 0));
  slot->slot_desc_capacity = C;

  slot->temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                slot->temp_bytes,
                                slot->d_permuted_sizes,
                                slot->d_offsets,
                                (int)C,
                                (cudaStream_t)0);

  slot->batches_per_slot_cap = batches_per_slot_cap;
  slot->slot_batches = (struct batch_slice_entry*)calloc(
    batches_per_slot_cap, sizeof(struct batch_slice_entry));
  CHECK(Error, slot->slot_batches);

  if (max_total_shards > 0) {
    const size_t dense_h_count =
      (size_t)batches_per_slot_cap * max_total_shards;
    CU(Error,
       cuMemAlloc((CUdeviceptr*)&slot->d_shard_sum_bytes,
                  max_total_shards * sizeof(size_t)));
    CU(Error,
       cuMemHostAlloc((void**)&slot->h_shard_base_offsets_dense,
                      dense_h_count * sizeof(size_t),
                      0));
    CU(Error,
       cuMemAlloc((CUdeviceptr*)&slot->d_shard_base_offsets_dense,
                  dense_h_count * sizeof(size_t)));
    for (size_t i = 0; i < dense_h_count; ++i)
      ((size_t*)slot->h_shard_base_offsets_dense)[i] = 0;

    size_t reduce_temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr,
                           reduce_temp_bytes,
                           (const size_t*)nullptr,
                           (size_t*)nullptr,
                           (int)max_total_shards,
                           (cudaStream_t)0);
    if (reduce_temp_bytes > slot->temp_bytes)
      slot->temp_bytes = reduce_temp_bytes;
  }

  if (slot->temp_bytes > 0)
    CU(Error, cuMemAlloc((CUdeviceptr*)&slot->d_temp, slot->temp_bytes));

  CU(Error, cuEventCreate(&slot->ready, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&slot->host_func_done, CU_EVENT_DEFAULT));

  CU(Error,
     cuMemAlloc((CUdeviceptr*)&slot->d_runtime,
                sizeof(struct slot_runtime_state)));
  CU(Error,
     cuMemsetD8(
       (CUdeviceptr)slot->d_runtime, 0, sizeof(struct slot_runtime_state)));

  return 0;

Error:
  aggregate_slot_destroy(slot);
  return 1;
}

__global__ void
add_shard_bias_unified_k(
  const struct aggregate_write_desc* __restrict__ aggregate_write_desc,
  const size_t* __restrict__ d_tail_bytes_prev,
  const uint64_t* __restrict__ d_shard_tps_group,
  const uint64_t* __restrict__ d_shard_offsets_base,
  uint64_t num_shards)
{
  const uint64_t s = blockIdx.x;
  if (s >= num_shards)
    return;
  __shared__ size_t* d_offsets;
  __shared__ const size_t* d_shard_base_offsets;
  if (threadIdx.x == 0) {
    d_offsets = aggregate_write_desc->target_d_offsets +
                aggregate_write_desc->desc_base_offset;
    d_shard_base_offsets =
      aggregate_write_desc->target_d_shard_base_offsets_dense +
      (uint64_t)aggregate_write_desc->batch_idx_in_slot * num_shards;
  }
  __syncthreads();
  const uint64_t base = d_shard_offsets_base[s];
  const uint64_t tps_group = d_shard_tps_group[s];
  __shared__ size_t bias_s;
  if (threadIdx.x == 0)
    bias_s = d_shard_base_offsets[s] + d_tail_bytes_prev[s] - d_offsets[base];
  __syncthreads();
  for (uint64_t k = threadIdx.x; k < tps_group; k += blockDim.x)
    d_offsets[base + k] += bias_s;
}

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

static int
tail_sum_launch(void* d_temp,
                size_t temp_bytes,
                const size_t* d_tail_bytes,
                size_t* d_tail_sum_bytes,
                uint64_t num_shards,
                CUstream stream)
{
  return cub::DeviceReduce::Sum(d_temp,
                                temp_bytes,
                                d_tail_bytes,
                                d_tail_sum_bytes,
                                (int)num_shards,
                                (cudaStream_t)stream) == cudaSuccess
           ? 0
           : 1;
}

__global__ void
copy_leading_tail_unified_k(
  const struct aggregate_write_desc* __restrict__ aggregate_write_desc,
  const void* __restrict__ d_tail_carry,
  const size_t* __restrict__ d_tail_bytes_prev,
  uint64_t num_shards,
  size_t page_size)
{
  const uint64_t s = blockIdx.x;
  const size_t nbytes = d_tail_bytes_prev[s];
  if (nbytes == 0)
    return;
  __shared__ void* d_aggregated;
  __shared__ const size_t* d_shard_base_offsets;
  if (threadIdx.x == 0) {
    d_aggregated = aggregate_write_desc->target_d_aggregated;
    d_shard_base_offsets =
      aggregate_write_desc->target_d_shard_base_offsets_dense +
      (uint64_t)aggregate_write_desc->batch_idx_in_slot * num_shards;
  }
  __syncthreads();
  const uint8_t* src = (const uint8_t*)d_tail_carry + s * page_size;
  uint8_t* dst = (uint8_t*)d_aggregated + d_shard_base_offsets[s];
  for (size_t off = threadIdx.x; off < nbytes; off += blockDim.x)
    dst[off] = src[off];
}

// Read d_tail_bytes[s] before the thread-0 write; __syncthreads guards the
// in-block read-then-write. Correctness assumes no shard finalizes mid-slot
// (the host-side fit gate must enforce this when macro-agg is active).
__global__ void
rollforward_tail_unified_k(
  const struct aggregate_write_desc* __restrict__ aggregate_write_desc,
  const size_t* __restrict__ d_shard_sum_bytes,
  size_t page_size,
  size_t* __restrict__ d_tail_bytes,
  void* __restrict__ d_tail_carry,
  uint64_t num_shards)
{
  const uint64_t s = blockIdx.x;
  if (s >= num_shards)
    return;
  __shared__ const void* d_aggregated;
  __shared__ const size_t* d_shard_base_offsets;
  if (threadIdx.x == 0) {
    d_aggregated = aggregate_write_desc->target_d_aggregated;
    d_shard_base_offsets =
      aggregate_write_desc->target_d_shard_base_offsets_dense +
      (uint64_t)aggregate_write_desc->batch_idx_in_slot * num_shards;
  }
  __syncthreads();
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

extern "C" int
aggregate_batch_measure_unified_async(
  const void* d_compressed,
  size_t* d_comp_sizes,
  const uint32_t* d_batch_gather,
  const uint32_t* d_batch_perm,
  uint64_t total_batch_chunks,
  uint64_t total_batch_covering,
  uint8_t nlod,
  size_t max_comp_chunk_bytes,
  struct aggregate_slot* slot,
  const size_t* d_shard_capacity,
  const uint64_t* d_shard_tps_group,
  const uint64_t* d_shard_offsets_base,
  size_t* d_tail_bytes,
  CUdeviceptr d_tail_carry,
  size_t page_size,
  uint64_t total_shards,
  int would_finalize_stay,
  int would_finalize_alone,
  struct aggregate_append_measurement* d_measurement,
  volatile struct aggregate_append_measurement* h_measurement,
  CUevent measurement_ready,
  size_t* d_tail_sum_bytes,
  size_t* d_temp_offsets,
  size_t* d_temp_perm_sizes,
  CUstream stream)
{
  (void)d_compressed;
  (void)d_shard_capacity;
  (void)max_comp_chunk_bytes;
  (void)d_tail_carry;
  const uint64_t N = total_batch_chunks;
  const uint64_t C = total_batch_covering;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  size_t* d_perm_sizes_b = d_temp_perm_sizes;
  size_t* d_offsets_b = d_temp_offsets;
  size_t* d_actual_bytes_ptr = d_offsets_b + (C + nlod - 1);
  int include_tail_sum = 0;

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

  if (total_shards > 0 && slot->d_shard_sum_bytes) {
    const int block = 256;
    compute_shard_sum_bytes_k<<<(int)total_shards, block, 0, cuda_stream>>>(
      d_perm_sizes_b,
      d_shard_offsets_base,
      d_shard_tps_group,
      slot->d_shard_sum_bytes,
      total_shards);
  }
  if (page_size > 0 && total_shards > 0 && slot->d_shard_sum_bytes) {
    CHECK(Error, d_tail_sum_bytes);
    CHECK(Error,
          tail_sum_launch(slot->d_temp,
                          slot->temp_bytes,
                          d_tail_bytes,
                          d_tail_sum_bytes,
                          total_shards,
                          stream) == 0);
    include_tail_sum = 1;
  }

  CHECK(Error, d_measurement);
  CHECK(Error,
        aggregate_measurement_launch(d_measurement,
                                     d_actual_bytes_ptr,
                                     d_tail_sum_bytes,
                                     include_tail_sum,
                                     C + (uint64_t)nlod,
                                     would_finalize_alone,
                                     would_finalize_stay,
                                     stream) == 0);
  if (h_measurement) {
    CU(Error,
       cuMemcpyDtoHAsync((void*)h_measurement,
                         (CUdeviceptr)d_measurement,
                         sizeof(struct aggregate_append_measurement),
                         stream));
  }
  if (measurement_ready)
    CU(Error, cuEventRecord(measurement_ready, stream));

  return 0;

Error:
  return 1;
}

extern "C" int
aggregate_batch_write_reserved_unified_async(
  const void* d_compressed,
  size_t* d_comp_sizes,
  const uint32_t* d_batch_gather,
  const uint32_t* d_batch_perm,
  uint64_t total_batch_chunks,
  uint64_t total_batch_covering,
  uint8_t nlod,
  size_t max_comp_chunk_bytes,
  struct aggregate_slot* scratch_slot,
  struct aggregate_slot* target_slot,
  const struct aggregate_slot_reservation* reservation,
  const uint64_t* d_shard_tps_group,
  const uint64_t* d_shard_offsets_base,
  size_t* d_tail_bytes,
  CUdeviceptr d_tail_carry,
  size_t page_size,
  uint64_t total_shards,
  struct aggregate_write_desc* aggregate_write_desc,
  struct aggregate_append_measurement* d_measurement,
  size_t* d_temp_offsets,
  size_t* d_temp_perm_sizes,
  struct aggregate_write_cb_args* cb_args,
  CUstream stream)
{
  const uint64_t N = total_batch_chunks;
  const uint64_t C = total_batch_covering;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  struct slot_dev_ptrs target = {
    .d_aggregated = target_slot->d_aggregated,
    .d_offsets = target_slot->d_offsets,
    .d_permuted_sizes = target_slot->d_permuted_sizes,
    .d_shard_base_offsets_dense = target_slot->d_shard_base_offsets_dense,
    .d_runtime = target_slot->d_runtime,
  };

  CHECK(Error,
        write_desc_from_reservation_launch(
          aggregate_write_desc, target, d_measurement, reservation, stream) ==
          0);

  if (total_shards > 0 && scratch_slot->d_shard_sum_bytes) {
    CHECK(Error,
          dense_offsets_launch(aggregate_write_desc,
                               scratch_slot->d_shard_sum_bytes,
                               d_tail_bytes,
                               total_shards,
                               stream) == 0);
    if (target_slot->h_shard_base_offsets_dense) {
      // Copy all cap slices: dense_offsets_k writes to
      // slice[batch_idx_in_slot]; delivery reads each batch's slice. Only the
      // current batch's slice is freshly written, but the unchanged slices must
      // remain valid host-side.
      CU(Error,
         cuMemcpyDtoHAsync((void*)target_slot->h_shard_base_offsets_dense,
                           (CUdeviceptr)target_slot->d_shard_base_offsets_dense,
                           (size_t)target_slot->batches_per_slot_cap *
                             total_shards * sizeof(size_t),
                           stream));
    }
  }

  CHECK(Error,
        copy_to_slot_launch(aggregate_write_desc,
                            d_temp_offsets,
                            d_temp_perm_sizes,
                            C + (uint64_t)nlod,
                            stream) == 0);

  if (page_size > 0 && total_shards > 0) {
    {
      const int block = 256;
      add_shard_bias_unified_k<<<(int)total_shards, block, 0, cuda_stream>>>(
        aggregate_write_desc,
        d_tail_bytes,
        d_shard_tps_group,
        d_shard_offsets_base,
        total_shards);
    }
    {
      const int block = 256;
      copy_leading_tail_unified_k<<<(int)total_shards, block, 0, cuda_stream>>>(
        aggregate_write_desc,
        (const void*)d_tail_carry,
        d_tail_bytes,
        total_shards,
        page_size);
    }
  }

  {
    const int block = 256;
    const int grid = (int)N;
    gather_batch_write_desc_k<<<grid, block, 0, cuda_stream>>>(
      aggregate_write_desc,
      d_compressed,
      d_comp_sizes,
      d_batch_gather,
      d_batch_perm,
      max_comp_chunk_bytes);
  }

  if (page_size > 0 && total_shards > 0 && scratch_slot->d_shard_sum_bytes) {
    const int block = 256;
    rollforward_tail_unified_k<<<(int)total_shards, block, 0, cuda_stream>>>(
      aggregate_write_desc,
      scratch_slot->d_shard_sum_bytes,
      page_size,
      (size_t*)d_tail_bytes,
      (void*)d_tail_carry,
      total_shards);
  }

  CU(Error,
     cuMemcpyDtoHAsync((void*)cb_args->h_write_desc,
                       (CUdeviceptr)aggregate_write_desc,
                       sizeof(struct aggregate_write_desc),
                       stream));
  CU(Error, cuLaunchHostFunc(stream, aggregate_post_batch_write_cb, cb_args));
  CU(Error, cuEventRecord(scratch_slot->host_func_done, stream));

  return 0;

Error:
  return 1;
}
