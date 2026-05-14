// Contract test for the GPU-side aggregate_result.
//
// Drives aggregate_batch_by_shard_async with synthetic device-side inputs,
// then runs the shared verifier from aggregate_contract.h. Mirrors the CPU
// contract test (test_aggregate_contract_cpu.c). The contract is:
//
//   page_size > 0  : per-shard regions of size shard_capacity, leading tail
//                    at the head, chunks pack tightly afterward.
//   page_size == 0 : single tightly-packed prefix sum starting at 0.

#include "aggregate_contract.h"

#include "gpu/aggregate.h"
#include "gpu/prelude.cuda.h"
#include "lod/lod_plan.h"
#include "stream/types.aggregate.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// One LOD; same geometry as the CPU contract test for symmetry.
//   t = unbounded, chunk=1, cps=2  (cps_append=2)
//   y = 4,         chunk=2, cps=2
//   x = 8,         chunk=2, cps=2  (=> 4 x-chunks, 2 along x => 2 inner shards)
struct geom
{
  uint8_t rank;
  uint8_t n_append;
  uint64_t chunk_count[3];
  uint64_t chunks_per_shard[3];
  uint64_t M;
  size_t max_comp;
  uint32_t batch_count;
  uint64_t cps_append;
};

static struct geom
make_geom(void)
{
  return (struct geom){
    .rank = 3,
    .n_append = 1,
    .chunk_count = { 0, 2, 4 },
    .chunks_per_shard = { 0, 2, 2 },
    .M = 8,
    .max_comp = 64,
    .batch_count = 1,
    .cps_append = 2,
  };
}

struct gpu_run
{
  struct aggregate_layout layout;
  struct aggregate_slot slot;
  CUstream stream;
  CUdeviceptr d_compressed;
  size_t* d_comp_sizes;
  uint32_t* d_gather;
  uint32_t* d_perm;
  size_t* d_tail_bytes;
  CUdeviceptr d_tail_carry;
  size_t* h_offsets;
  // Outputs filled at the end of each scenario:
  struct aggregate_result result;
};

static void
gpu_run_destroy(struct gpu_run* r)
{
  if (r->h_offsets)
    free(r->h_offsets);
  cu_mem_free((CUdeviceptr)r->d_comp_sizes);
  cu_mem_free((CUdeviceptr)r->d_gather);
  cu_mem_free((CUdeviceptr)r->d_perm);
  cu_mem_free((CUdeviceptr)r->d_tail_bytes);
  cu_mem_free(r->d_tail_carry);
  cu_mem_free(r->d_compressed);
  if (r->slot.h_aggregated || r->slot.d_aggregated)
    aggregate_slot_destroy(&r->slot);
  cu_stream_destroy(r->stream);
  aggregate_layout_destroy(&r->layout);
  memset(r, 0, sizeof(*r));
}

// Set up the layout/slot/buffers and execute one aggregate kick. The result
// is left in r->result with offsets D2H'd into r->h_offsets and chunk_sizes
// already on the host (h_permuted_sizes).
static int
run_gpu_aggregate(struct gpu_run* r,
                  const struct geom* g,
                  size_t page_size,
                  const size_t* tail_in /* [num_shards] or NULL */)
{
  memset(r, 0, sizeof(*r));

  // Declared up front so the Fail label's free()s are well-defined on the
  // early CHECK paths before the alloc lines.
  size_t* h_sizes = NULL;
  uint8_t* h_input = NULL;
  uint32_t* h_gather = NULL;
  uint32_t* h_perm = NULL;
  uint32_t* pool_epochs = NULL;

  CHECK(Fail,
        aggregate_layout_compute(&r->layout,
                                 g->rank,
                                 g->n_append,
                                 g->chunk_count,
                                 g->chunks_per_shard,
                                 g->M,
                                 g->max_comp,
                                 g->batch_count,
                                 page_size,
                                 g->cps_append) == 0);
  CHECK(Fail, aggregate_layout_upload(&r->layout) == 0);

  CU(Fail, cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

  const uint64_t M = r->layout.chunks_per_epoch;
  const uint64_t C = r->layout.covering_count;
  const uint64_t N = (uint64_t)g->batch_count * M;
  const uint64_t batch_C = (uint64_t)g->batch_count * C;
  const uint64_t num_shards = r->layout.num_shards;
  const size_t comp_pool_bytes = N * g->max_comp;

  CHECK(Fail,
        aggregate_batch_slot_init(
          &r->slot, N, batch_C + 1, comp_pool_bytes, 1) == 0);

  // Synthetic compressed pool: chunk i has size (10 + i%7), filled with
  // value (i+1)&0xff. Same shape as the CPU test.
  h_sizes = (size_t*)calloc(N, sizeof(size_t));
  h_input = (uint8_t*)calloc(N, g->max_comp);
  CHECK(Fail, h_sizes && h_input);
  for (uint64_t i = 0; i < N; ++i) {
    h_sizes[i] = 10 + (i % 7);
    memset(h_input + i * g->max_comp, (int)((i + 1) & 0xff), h_sizes[i]);
  }
  CU(Fail, cuMemAlloc(&r->d_compressed, comp_pool_bytes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&r->d_comp_sizes, N * sizeof(size_t)));
  CU(Fail, cuMemcpyHtoD(r->d_compressed, h_input, comp_pool_bytes));
  CU(Fail,
     cuMemcpyHtoD((CUdeviceptr)r->d_comp_sizes, h_sizes, N * sizeof(size_t)));
  free(h_input);

  // Build gather / perm LUTs for batch_count epochs at pool epoch 0..K-1.
  h_gather = (uint32_t*)calloc(N, sizeof(uint32_t));
  h_perm = (uint32_t*)calloc(N, sizeof(uint32_t));
  pool_epochs = (uint32_t*)calloc(g->batch_count, sizeof(uint32_t));
  CHECK(Fail, h_gather && h_perm && pool_epochs);
  for (uint32_t a = 0; a < g->batch_count; ++a)
    pool_epochs[a] = a;
  struct level_geometry levels = { .nlod = 1, .total_chunks = M };
  levels.level[0].chunk_count = M;
  levels.level[0].chunk_offset = 0;
  aggregate_batch_luts(&r->layout,
                       &levels,
                       /*lv=*/0,
                       g->batch_count,
                       pool_epochs,
                       h_gather,
                       h_perm);
  free(pool_epochs);

  CU(Fail, cuMemAlloc((CUdeviceptr*)&r->d_gather, N * sizeof(uint32_t)));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&r->d_perm, N * sizeof(uint32_t)));
  CU(Fail,
     cuMemcpyHtoD((CUdeviceptr)r->d_gather, h_gather, N * sizeof(uint32_t)));
  CU(Fail, cuMemcpyHtoD((CUdeviceptr)r->d_perm, h_perm, N * sizeof(uint32_t)));
  free(h_gather);
  free(h_perm);

  // Tail-carry state. Allocated even when page_size==0 (passed as harmless
  // nulls below) — but to stay defensive we condition on page_size>0.
  if (page_size > 0 && num_shards > 0) {
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&r->d_tail_bytes, num_shards * sizeof(size_t)));
    CU(Fail, cuMemAlloc(&r->d_tail_carry, num_shards * page_size));
    CU(
      Fail,
      cuMemsetD8((CUdeviceptr)r->d_tail_bytes, 0, num_shards * sizeof(size_t)));
    CU(Fail, cuMemsetD8(r->d_tail_carry, 0, num_shards * page_size));
    if (tail_in) {
      CU(Fail,
         cuMemcpyHtoD(
           (CUdeviceptr)r->d_tail_bytes, tail_in, num_shards * sizeof(size_t)));
      // d_tail_carry contents don't affect the offsets contract; leave zero.
    }
  }

  free(h_sizes);
  h_sizes = NULL;

  // Kick the per-shard aggregate.
  CHECK(Fail,
        aggregate_batch_by_shard_async((const void*)r->d_compressed,
                                       r->d_comp_sizes,
                                       r->d_gather,
                                       r->d_perm,
                                       N,
                                       batch_C,
                                       g->max_comp,
                                       &r->layout,
                                       &r->slot,
                                       r->d_tail_bytes,
                                       r->d_tail_carry,
                                       r->stream) == 0);

  CU(Fail, cuStreamSynchronize(r->stream));

  // Pull offsets and permuted sizes back to host. Allocate a separate copy
  // for offsets so the slot's h_offsets is decoupled from the contract
  // result struct (lifetime-wise, both are fine here, but the pattern
  // matches how delivery sees it). h_permuted_sizes is now copied here
  // (production does the D2H in d2h_deliver_kick on d2h_stream).
  r->h_offsets = (size_t*)calloc(batch_C + 1, sizeof(size_t));
  CHECK(Fail, r->h_offsets);
  CU(Fail,
     cuMemcpyDtoH(r->h_offsets,
                  (CUdeviceptr)r->slot.d_offsets,
                  (batch_C + 1) * sizeof(size_t)));
  CU(Fail,
     cuMemcpyDtoH(r->slot.h_permuted_sizes,
                  (CUdeviceptr)r->slot.d_permuted_sizes,
                  batch_C * sizeof(size_t)));

  r->result.data = r->slot.h_aggregated;
  r->result.offsets = r->h_offsets;
  r->result.chunk_sizes = r->slot.h_permuted_sizes;
  return 0;

Fail:
  free(h_sizes);
  free(h_input);
  free(h_gather);
  free(h_perm);
  free(pool_epochs);
  return 1;
}

static int
test_carryover_no_tail(void)
{
  log_info("=== test_carryover_no_tail ===");
  struct geom g = make_geom();
  struct gpu_run r;
  CHECK(Fail, run_gpu_aggregate(&r, &g, 4096, NULL) == 0);
  CHECK(Fail,
        verify_aggregate_result_carryover(
          &r.result, &r.layout, NULL, g.batch_count) == 0);
  gpu_run_destroy(&r);
  log_info("  PASS");
  return 0;
Fail:
  gpu_run_destroy(&r);
  log_error("  FAIL");
  return 1;
}

static int
test_carryover_with_tail(void)
{
  log_info("=== test_carryover_with_tail ===");
  struct geom g = make_geom();
  // Match the CPU test's tail values for cross-side parity.
  const size_t tail_in[2] = { 17, 113 };
  struct gpu_run r;
  CHECK(Fail, run_gpu_aggregate(&r, &g, 4096, tail_in) == 0);
  CHECK(Fail,
        verify_aggregate_result_carryover(
          &r.result, &r.layout, tail_in, g.batch_count) == 0);
  gpu_run_destroy(&r);
  log_info("  PASS");
  return 0;
Fail:
  gpu_run_destroy(&r);
  log_error("  FAIL");
  return 1;
}

static int
test_contiguous(void)
{
  log_info("=== test_contiguous ===");
  struct geom g = make_geom();
  struct gpu_run r;
  CHECK(Fail, run_gpu_aggregate(&r, &g, 0, NULL) == 0);
  CHECK(Fail,
        verify_aggregate_result_contiguous(
          &r.result, &r.layout, g.batch_count) == 0);
  gpu_run_destroy(&r);
  log_info("  PASS");
  return 0;
Fail:
  gpu_run_destroy(&r);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;
  CUcontext ctx = NULL;
  CUdevice dev;
  if (cuInit(0) != CUDA_SUCCESS)
    return 1;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS)
    return 1;
  if (cu_ctx_create(&ctx, 0, dev) != CUDA_SUCCESS)
    return 1;

  int rc = 0;
  rc |= test_carryover_no_tail();
  rc |= test_carryover_with_tail();
  rc |= test_contiguous();

  cuCtxDestroy(ctx);
  return rc;
}
