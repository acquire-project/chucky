#include "aggregate.h"
#include "index.ops.util.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "test_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Test 1: even case — rank=3, tc=(4,6,8), tps=(_,3,4)
//   M = 6*8 = 48, sc = (2,2), C = 2*3 * 2*4 = 48 (C == M)
// ---------------------------------------------------------------------------
static int
test_aggregate_even(void)
{
  printf("--- test_aggregate_even ---\n");
  int ok = 0;

  const uint8_t rank = 3;
  const uint64_t chunk_count[3] = { 4, 6, 8 };
  const uint64_t chunks_per_shard[3] = { 0, 3, 4 };
  const uint64_t M = chunk_count[1] * chunk_count[2]; // 48
  const size_t max_comp = 64;

  struct aggregate_layout layout;
  memset(&layout, 0, sizeof(layout));
  struct aggregate_slot slot;
  memset(&slot, 0, sizeof(slot));
  CUstream stream = NULL;
  CUdeviceptr d_compressed = 0;
  CUdeviceptr d_comp_sizes = 0;
  size_t* h_sizes = NULL;
  uint8_t* h_input = NULL;
  uint8_t* h_agg = NULL;

  CHECK(Fail,
        aggregate_layout_compute(
          &layout, rank, chunk_count, chunks_per_shard, M, max_comp, 0) == 0);
  CHECK(Fail, aggregate_layout_upload(&layout) == 0);

  const uint64_t C = layout.covering_count;
  printf("  M=%llu C=%llu lifted_rank=%u\n",
         (unsigned long long)M,
         (unsigned long long)C,
         layout.lifted_rank);
  CHECK(Fail, C == 48);

  const size_t pool_bytes = M * max_comp;
  CHECK(Fail, aggregate_batch_slot_init(&slot, M, C, pool_bytes) == 0);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // Build fake compressed input on host, then upload
  h_sizes = (size_t*)calloc(M, sizeof(size_t));
  h_input = (uint8_t*)calloc(pool_bytes, 1);
  CHECK(Fail, h_sizes && h_input);

  for (uint64_t i = 0; i < M; ++i) {
    h_sizes[i] = 10 + (i % 7); // varying sizes: 10..16
    uint8_t val = (uint8_t)(i + 1);
    memset(h_input + i * max_comp, val, h_sizes[i]);
  }

  CU(Fail, cuMemAlloc(&d_compressed, pool_bytes));
  CU(Fail, cuMemAlloc(&d_comp_sizes, M * sizeof(size_t)));
  CU(Fail, cuMemcpyHtoD(d_compressed, h_input, pool_bytes));
  CU(Fail, cuMemcpyHtoD(d_comp_sizes, h_sizes, M * sizeof(size_t)));

  CHECK(Fail,
        aggregate_by_shard_async(
          &layout, (void*)d_compressed, (size_t*)d_comp_sizes, &slot, stream) ==
          0);
  CU(Fail, cuStreamSynchronize(stream));

  // D2H aggregated buffer and offsets
  h_agg = (uint8_t*)malloc(pool_bytes);
  CHECK(Fail, h_agg);

  CU(Fail, cuMemcpyDtoH(h_agg, (CUdeviceptr)slot.d_aggregated, pool_bytes));
  CU(Fail,
     cuMemcpyDtoH(
       slot.h_offsets, (CUdeviceptr)slot.d_offsets, (C + 1) * sizeof(size_t)));

  // Verify: compute CPU reference permutation and check
  {
    // Build expected prefix sum in shard-major order
    size_t* permuted_sizes = (size_t*)calloc(C, sizeof(size_t));
    CHECK(Fail, permuted_sizes);

    for (uint64_t i = 0; i < M; ++i) {
      uint32_t pi = cpu_perm(
        i, layout.lifted_rank, layout.lifted_shape, layout.lifted_strides);
      CHECK(Fail, pi < C);
      permuted_sizes[pi] = h_sizes[i];
    }

    // Expected exclusive prefix sum
    size_t expected_total = 0;
    for (uint64_t j = 0; j < C; ++j) {
      CHECK(Fail, slot.h_offsets[j] == expected_total);
      expected_total += permuted_sizes[j];
    }
    CHECK(Fail, slot.h_offsets[C] == expected_total);
    printf("  total bytes: %zu\n", expected_total);

    // Verify byte contents
    for (uint64_t i = 0; i < M; ++i) {
      uint32_t pi = cpu_perm(
        i, layout.lifted_rank, layout.lifted_shape, layout.lifted_strides);
      size_t off = slot.h_offsets[pi];
      uint8_t expected_val = (uint8_t)(i + 1);
      for (size_t b = 0; b < h_sizes[i]; ++b) {
        if (h_agg[off + b] != expected_val) {
          fprintf(stderr,
                  "  MISMATCH chunk %llu P[i]=%u byte %zu: got %u want %u\n",
                  (unsigned long long)i,
                  pi,
                  b,
                  (unsigned)h_agg[off + b],
                  (unsigned)expected_val);
          free(permuted_sizes);
          goto Fail;
        }
      }
    }

    free(permuted_sizes);
  }

  printf("  PASS\n");
  ok = 1;

Fail:
  free(h_agg);
  free(h_input);
  free(h_sizes);
  cuMemFree(d_compressed);
  cuMemFree(d_comp_sizes);
  cuStreamDestroy(stream);
  aggregate_slot_destroy(&slot);
  aggregate_layout_destroy(&layout);
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: uneven case — rank=2, tc=(1,7), tps=(_,3)
//   M = 7, sc = (3), C = 3*3 = 9 > M
//   Two padding slots have size 0
// ---------------------------------------------------------------------------
static int
test_aggregate_uneven(void)
{
  printf("--- test_aggregate_uneven ---\n");
  int ok = 0;

  const uint8_t rank = 2;
  const uint64_t chunk_count[2] = { 1, 7 };
  const uint64_t chunks_per_shard[2] = { 0, 3 };
  const uint64_t M = chunk_count[1]; // 7
  const size_t max_comp = 32;

  struct aggregate_layout layout;
  memset(&layout, 0, sizeof(layout));
  struct aggregate_slot slot;
  memset(&slot, 0, sizeof(slot));
  CUstream stream = NULL;
  CUdeviceptr d_compressed = 0;
  CUdeviceptr d_comp_sizes = 0;
  size_t* h_sizes = NULL;
  uint8_t* h_input = NULL;
  uint8_t* h_agg = NULL;

  CHECK(Fail,
        aggregate_layout_compute(
          &layout, rank, chunk_count, chunks_per_shard, M, max_comp, 0) == 0);
  CHECK(Fail, aggregate_layout_upload(&layout) == 0);

  const uint64_t C = layout.covering_count;
  printf("  M=%llu C=%llu lifted_rank=%u\n",
         (unsigned long long)M,
         (unsigned long long)C,
         layout.lifted_rank);
  CHECK(Fail, C == 9);
  CHECK(Fail, C > M);

  const size_t pool_bytes = M * max_comp;
  CHECK(Fail, aggregate_batch_slot_init(&slot, M, C, pool_bytes) == 0);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  h_sizes = (size_t*)calloc(M, sizeof(size_t));
  h_input = (uint8_t*)calloc(pool_bytes, 1);
  CHECK(Fail, h_sizes && h_input);

  for (uint64_t i = 0; i < M; ++i) {
    h_sizes[i] = 5 + (i % 4); // 5..8
    uint8_t val = (uint8_t)(i + 1);
    memset(h_input + i * max_comp, val, h_sizes[i]);
  }

  CU(Fail, cuMemAlloc(&d_compressed, pool_bytes));
  CU(Fail, cuMemAlloc(&d_comp_sizes, M * sizeof(size_t)));
  CU(Fail, cuMemcpyHtoD(d_compressed, h_input, pool_bytes));
  CU(Fail, cuMemcpyHtoD(d_comp_sizes, h_sizes, M * sizeof(size_t)));

  CHECK(Fail,
        aggregate_by_shard_async(
          &layout, (void*)d_compressed, (size_t*)d_comp_sizes, &slot, stream) ==
          0);
  CU(Fail, cuStreamSynchronize(stream));

  h_agg = (uint8_t*)malloc(pool_bytes);
  CHECK(Fail, h_agg);

  CU(Fail, cuMemcpyDtoH(h_agg, (CUdeviceptr)slot.d_aggregated, pool_bytes));
  CU(Fail,
     cuMemcpyDtoH(
       slot.h_offsets, (CUdeviceptr)slot.d_offsets, (C + 1) * sizeof(size_t)));

  // Verify
  {
    size_t* permuted_sizes = (size_t*)calloc(C, sizeof(size_t));
    CHECK(Fail, permuted_sizes);

    for (uint64_t i = 0; i < M; ++i) {
      uint32_t pi = cpu_perm(
        i, layout.lifted_rank, layout.lifted_shape, layout.lifted_strides);
      CHECK(Fail, pi < C);
      permuted_sizes[pi] = h_sizes[i];
    }

    // Expected exclusive prefix sum — some entries are 0 (padding)
    size_t expected_total = 0;
    int zero_count = 0;
    for (uint64_t j = 0; j < C; ++j) {
      CHECK(Fail, slot.h_offsets[j] == expected_total);
      if (permuted_sizes[j] == 0)
        ++zero_count;
      expected_total += permuted_sizes[j];
    }
    CHECK(Fail, slot.h_offsets[C] == expected_total);
    CHECK(Fail, zero_count == (int)(C - M));
    printf(
      "  total bytes: %zu, padding slots: %d\n", expected_total, zero_count);

    // Verify byte contents
    for (uint64_t i = 0; i < M; ++i) {
      uint32_t pi = cpu_perm(
        i, layout.lifted_rank, layout.lifted_shape, layout.lifted_strides);
      size_t off = slot.h_offsets[pi];
      uint8_t expected_val = (uint8_t)(i + 1);
      for (size_t b = 0; b < h_sizes[i]; ++b) {
        if (h_agg[off + b] != expected_val) {
          fprintf(stderr,
                  "  MISMATCH chunk %llu P[i]=%u byte %zu: got %u want %u\n",
                  (unsigned long long)i,
                  pi,
                  b,
                  (unsigned)h_agg[off + b],
                  (unsigned)expected_val);
          free(permuted_sizes);
          goto Fail;
        }
      }
    }

    free(permuted_sizes);
  }

  printf("  PASS\n");
  ok = 1;

Fail:
  free(h_agg);
  free(h_input);
  free(h_sizes);
  cuMemFree(d_compressed);
  cuMemFree(d_comp_sizes);
  cuStreamDestroy(stream);
  aggregate_slot_destroy(&slot);
  aggregate_layout_destroy(&layout);
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "aggregate_even", test_aggregate_even },
              { "aggregate_uneven", test_aggregate_uneven }, )
