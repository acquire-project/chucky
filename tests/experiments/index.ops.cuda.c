#include "index.ops.util.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"
#include "gpu/transpose.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

static void
setup_transpose_strides(int rank,
                        const int* shape,
                        const int* permutation,
                        int* transposed_shape,
                        int* transposed_strides)
{
  int output_strides[MAX_RANK] = { 0 };
  int inv_p[MAX_RANK] = { 0 };

  permute_i32(rank, permutation, shape, transposed_shape);
  compute_strides(rank, transposed_shape, output_strides);
  inverse_permutation_i32(rank, permutation, inv_p);
  permute_i32(rank, inv_p, output_strides, transposed_strides);
}

static int
test_transpose_hit_count_cpu(void)
{
  // CPU test: verify each output position is hit exactly once
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  const int p[] = { 0, 2, 4, 1, 3, 5 };
  int transposed_shape[6] = { 0 };
  int transposed_strides[6] = { 0 };
  int input_strides[6] = { 0 };

  setup_transpose_strides(rank, shape, p, transposed_shape, transposed_strides);
  compute_strides(rank, shape, input_strides);

  printf("%30s", "Input shape: ");
  println_vi32(rank, shape);
  printf("%30s", "Input strides (row-major): ");
  println_vi32(rank, input_strides);
  printf("%30s", "Transposed shape: ");
  println_vi32(rank, transposed_shape);
  printf("%30s", "Transposed strides: ");
  println_vi32(rank, transposed_strides);

  const int n = transposed_strides[0] * shape[0];
  printf("Total elements: %d\n", n);

  uint16_t* hit_count = (uint16_t*)calloc(n, sizeof(uint16_t));
  CHECK(Fail, hit_count);

  // Count how many times each output position is hit
  for (int i = 0; i < n; ++i) {
    uint64_t input_idx = i;
    uint64_t out_offset = 0;
    uint64_t rest = input_idx;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out_offset += coord * transposed_strides[d];
    }

    hit_count[out_offset]++;
  }

  // Verify each position hit exactly once
  int ecode = 0;
  int collision_count = 0;
  int zero_count = 0;
  for (int i = 0; i < n; ++i) {
    if (hit_count[i] != 1) {
      // printf("Position %d hit %u times\n", i, hit_count[i]);
      if (hit_count[i] == 0)
        zero_count++;
      else
        collision_count++;
      ecode = 1;
    }
  }
  if (ecode) {
    printf("FAIL: %d positions with collisions, %d positions never hit\n",
           collision_count,
           zero_count);
    printf("First 20 hit counts: ");
    for (int i = 0; i < (n < 20 ? n : 20); ++i) {
      printf("%u ", hit_count[i]);
    }
    printf("\n");
  } else {
    printf("OK: All positions hit exactly once\n");
  }

Finalize:
  free(hit_count);
  return ecode;
Fail:
  printf("FAIL\n");
  ecode = 1;
  goto Finalize;
}

static uint16_t*
make_expected_u16(int rank,
                  const int* shape,
                  const int* transposed_strides,
                  uint64_t offset,
                  uint64_t count)
{
  // Generate expected output for transpose_u16
  // Input array is filled with sequential values: 0, 1, 2, 3, ...
  uint16_t* expected = (uint16_t*)malloc(count * sizeof(uint16_t));
  if (!expected)
    return NULL;

  for (uint64_t i = 0; i < count; ++i) {
    uint64_t input_idx = offset + i;

    // Convert input index to output offset
    uint64_t out_offset = 0;
    uint64_t rest = input_idx;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out_offset += coord * transposed_strides[d];
    }

    // The value at this input position is just the input index
    expected[out_offset] = (uint16_t)(input_idx % 65536);
  }

  return expected;
}

static int
test_transpose_u16_basic(void)
{
  // Setup a tiling of a 640x480x3 array into chunks of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  const int p[] = { 0, 2, 4, 1, 3, 5 };
  int transposed_shape[6] = { 0 };
  int transposed_strides[6] = { 0 };
  int input_strides[6] = { 0 };

  setup_transpose_strides(rank, shape, p, transposed_shape, transposed_strides);
  compute_strides(rank, shape, input_strides);

  printf("%30s", "Input shape: ");
  println_vi32(rank, shape);
  printf("%30s", "Input strides (row-major): ");
  println_vi32(rank, input_strides);
  printf("%30s", "Transposed shape: ");
  println_vi32(rank, transposed_shape);
  printf("%30s", "Transposed strides: ");
  println_vi32(rank, transposed_strides);

  CUdeviceptr d_src = 0, d_dst = 0, d_shape = 0, d_strides = 0;
  uint16_t *src = 0, *actual = 0, *expected = 0;

  const int n = transposed_strides[0] * shape[0];
  printf("Total elements: %d\n", n);

  // Allocate host memory
  CHECK(Fail, src = (uint16_t*)malloc(n * sizeof(uint16_t)));
  CHECK(Fail, actual = (uint16_t*)malloc(n * sizeof(uint16_t)));

  // Fill source with sequential values
  for (int i = 0; i < n; ++i) {
    src[i] = (uint16_t)i;
  }

  // Generate expected output
  CHECK(Fail,
        expected = make_expected_u16(rank, shape, transposed_strides, 0, n));

  // Allocate device memory. Pad d_src to 4KB so the kernel can do
  // unconditional vectorized loads.
  const uint64_t src_alloc = (n * sizeof(uint16_t) + 4095) & ~(uint64_t)4095;
  CU(Fail, cuMemAlloc(&d_src, src_alloc));
  CU(Fail, cuMemsetD8(d_src, 0, src_alloc));
  CU(Fail, cuMemAlloc(&d_dst, n * sizeof(uint16_t)));

  // Copy source to device
  CUstream stream = 0;
  CU(Fail, cuMemcpyHtoDAsync(d_src, src, n * sizeof(uint16_t), stream));

  // Convert shape and strides to uint64_t/int64_t for the API
  uint64_t shape64[MAX_RANK] = { 0 };
  int64_t strides64[MAX_RANK] = { 0 };
  for (int i = 0; i < rank; ++i) {
    shape64[i] = shape[i];
    strides64[i] = transposed_strides[i];
  }

  // Copy shape and strides to device
  CU(Fail, cuMemAlloc(&d_shape, rank * sizeof(uint64_t)));
  CU(Fail, cuMemAlloc(&d_strides, rank * sizeof(int64_t)));
  CU(Fail, cuMemcpyHtoD(d_shape, shape64, rank * sizeof(uint64_t)));
  CU(Fail, cuMemcpyHtoD(d_strides, strides64, rank * sizeof(int64_t)));

  // Call the CUDA transpose kernel
  transpose(d_dst,
            d_src,
            n * sizeof(uint16_t),
            sizeof(uint16_t),
            0, // i_offset (start at input index 0)
            rank,
            (const uint64_t*)d_shape,
            (const int64_t*)d_strides,
            stream);

  // Copy results back to host
  CU(Fail, cuMemcpyDtoHAsync(actual, d_dst, n * sizeof(uint16_t), stream));
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuCtxSynchronize());

  // Compare results
  int ecode = 0;
  for (int i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      if (ecode == 0) {
        printf("FAIL: basic u16 transpose - mismatch at output index %d: "
               "expected %u, got %u\n",
               i,
               expected[i],
               actual[i]);
      }
      ecode = 1;
    }
  }

  if (ecode) {
    printf("Expected (first 20): ");
    for (int i = 0; i < (n < 20 ? n : 20); ++i) {
      printf("%u ", expected[i]);
    }
    printf("\n");
    printf("Actual (first 20): ");
    for (int i = 0; i < (n < 20 ? n : 20); ++i) {
      printf("%u ", actual[i]);
    }
    printf("\n");
  } else {
    printf("OK\n");
  }

Finalize:
  cuMemFree(d_shape);
  cuMemFree(d_strides);
  cuMemFree(d_src);
  cuMemFree(d_dst);
  free(src);
  free(actual);
  free(expected);

  return ecode;
Fail:
  printf("FAIL\n");
  ecode = 1;
  goto Finalize;
}

static int
test_transpose_u16_with_offset(void)
{
  // Setup a tiling of a 640x480x3 array into chunks of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  const int p[] = { 0, 2, 4, 1, 3, 5 };
  int transposed_shape[6] = { 0 };
  int transposed_strides[6] = { 0 };

  setup_transpose_strides(rank, shape, p, transposed_shape, transposed_strides);

  CUdeviceptr d_src = 0, d_dst = 0, d_shape = 0, d_strides = 0;
  uint16_t *src = 0, *actual = 0, *expected = 0;

  const int n = transposed_strides[0] * shape[0];

  // Test with a non-zero offset
  const uint64_t i_offset = 1000;
  const uint64_t count = 10000;

  // Allocate host memory
  CHECK(Fail, src = (uint16_t*)malloc(count * sizeof(uint16_t)));
  CHECK(Fail, actual = (uint16_t*)malloc(n * sizeof(uint16_t)));

  // Fill source with sequential values starting from i_offset
  for (uint64_t i = 0; i < count; ++i) {
    src[i] = (uint16_t)((i_offset + i) % 65536);
  }

  // Generate expected output (full array initialized to zero)
  CHECK(Fail, expected = (uint16_t*)calloc(n, sizeof(uint16_t)));

  // Fill expected output based on transpose logic
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t input_idx = i_offset + i;

    // Convert input index to output offset
    uint64_t out_offset = 0;
    uint64_t rest = input_idx;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out_offset += coord * transposed_strides[d];
    }

    expected[out_offset] = src[i];
  }

  // Allocate device memory. Pad d_src to 4KB so the kernel can do
  // unconditional vectorized loads.
  const uint64_t src_alloc =
    (count * sizeof(uint16_t) + 4095) & ~(uint64_t)4095;
  CU(Fail, cuMemAlloc(&d_src, src_alloc));
  CU(Fail, cuMemsetD8(d_src, 0, src_alloc));
  CU(Fail, cuMemAlloc(&d_dst, n * sizeof(uint16_t)));

  // Initialize destination to zero
  CUstream stream = 0;
  CU(Fail, cuMemsetD16Async(d_dst, 0, n, stream));

  // Copy source to device
  CU(Fail, cuMemcpyHtoDAsync(d_src, src, count * sizeof(uint16_t), stream));

  // Convert shape and strides to uint64_t/int64_t for the API
  uint64_t shape64[MAX_RANK] = { 0 };
  int64_t strides64[MAX_RANK] = { 0 };
  for (int i = 0; i < rank; ++i) {
    shape64[i] = shape[i];
    strides64[i] = transposed_strides[i];
  }

  // Copy shape and strides to device
  CU(Fail, cuMemAlloc(&d_shape, rank * sizeof(uint64_t)));
  CU(Fail, cuMemAlloc(&d_strides, rank * sizeof(int64_t)));
  CU(Fail, cuMemcpyHtoD(d_shape, shape64, rank * sizeof(uint64_t)));
  CU(Fail, cuMemcpyHtoD(d_strides, strides64, rank * sizeof(int64_t)));

  // Call the CUDA transpose kernel with offset
  transpose(d_dst,
            d_src,
            count * sizeof(uint16_t),
            sizeof(uint16_t),
            i_offset,
            rank,
            (const uint64_t*)d_shape,
            (const int64_t*)d_strides,
            stream);

  // Copy results back to host
  CU(Fail, cuMemcpyDtoHAsync(actual, d_dst, n * sizeof(uint16_t), stream));
  CU(Fail, cuStreamSynchronize(stream));

  // Compare results
  int ecode = 0;
  int mismatch_count = 0;
  for (int i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      if (mismatch_count < 10) {
        printf("FAIL: offset u16 transpose - mismatch at output index %d: "
               "expected %u, got %u\n",
               i,
               expected[i],
               actual[i]);
      }
      mismatch_count++;
      ecode = 1;
    }
  }

  if (ecode) {
    printf("Total mismatches: %d out of %d\n", mismatch_count, n);
  } else {
    printf("OK\n");
  }

Finalize:
  cuMemFree(d_shape);
  cuMemFree(d_strides);
  cuMemFree(d_src);
  cuMemFree(d_dst);
  free(src);
  free(actual);
  free(expected);

  return ecode;
Fail:
  printf("FAIL\n");
  ecode = 1;
  goto Finalize;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  printf("=== Test 0: CPU hit count verification ===\n");
  ecode |= test_transpose_hit_count_cpu();

  printf("\n=== Test 1: Basic u16 transpose (full array) ===\n");
  ecode |= test_transpose_u16_basic();

  printf("\n=== Test 2: U16 transpose with offset ===\n");
  ecode |= test_transpose_u16_with_offset();

  cuCtxDestroy(ctx);
  return ecode;
Fail:
  cuCtxDestroy(ctx);
  return 1;
}
