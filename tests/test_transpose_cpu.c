#include "cpu/transpose.h"
#include "defs.limits.h"
#include "index.ops.util.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// Run CPU transpose and verify against ravel() reference.
static int
run_test(const char* name,
         int rank,
         const uint64_t* dim_sizes,
         const uint64_t* chunk_sizes,
         const uint8_t* storage_order,
         uint8_t bpe)
{
  log_info("=== %s ===", name);

  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];
  uint64_t chunk_elements, chunk_stride, chunks_per_epoch, epoch_elements;

  build_lifted_layout(rank,
                      dim_sizes,
                      chunk_sizes,
                      storage_order,
                      &lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      &chunk_elements,
                      &chunk_stride,
                      &chunks_per_epoch,
                      &epoch_elements);

  uint64_t pool_elements = chunks_per_epoch * chunk_stride;
  size_t src_bytes = epoch_elements * bpe;
  size_t dst_bytes = pool_elements * bpe;

  log_info("  rank=%d lifted_rank=%d chunk_elements=%lu chunk_stride=%lu "
           "chunks_per_epoch=%lu epoch_elements=%lu",
           rank,
           lifted_rank,
           (unsigned long)chunk_elements,
           (unsigned long)chunk_stride,
           (unsigned long)chunks_per_epoch,
           (unsigned long)epoch_elements);

  void* src = malloc(src_bytes);
  void* dst = calloc(1, dst_bytes);
  CHECK(Fail, src && dst);

  // Fill source with sequential values
  if (bpe == 1) {
    uint8_t* p = (uint8_t*)src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = (uint8_t)(i & 0xFF);
  } else if (bpe == 2) {
    uint16_t* p = (uint16_t*)src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = (uint16_t)(i & 0xFFFF);
  } else if (bpe == 4) {
    uint32_t* p = (uint32_t*)src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = (uint32_t)i;
  } else {
    uint64_t* p = (uint64_t*)src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = i;
  }

  CHECK(
    Fail,
    transpose_cpu(
      dst, src, src_bytes, bpe, 0, lifted_rank, lifted_shape, lifted_strides, 0) ==
      0);

  // Verify against ravel() reference
  int errors = 0;
  for (uint64_t i = 0; i < epoch_elements; ++i) {
    uint64_t expected_off = ravel(lifted_rank, lifted_shape, lifted_strides, i);

    if (bpe == 1) {
      uint8_t sv = ((uint8_t*)src)[i];
      uint8_t dv = ((uint8_t*)dst)[expected_off];
      if (dv != sv) {
        if (errors < 5)
          log_error("  elem %lu: expected dst[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)expected_off,
                    sv,
                    dv);
        errors++;
      }
    } else if (bpe == 2) {
      uint16_t sv = ((uint16_t*)src)[i];
      uint16_t dv = ((uint16_t*)dst)[expected_off];
      if (dv != sv) {
        if (errors < 5)
          log_error("  elem %lu: expected dst[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)expected_off,
                    sv,
                    dv);
        errors++;
      }
    } else if (bpe == 4) {
      uint32_t sv = ((uint32_t*)src)[i];
      uint32_t dv = ((uint32_t*)dst)[expected_off];
      if (dv != sv) {
        if (errors < 5)
          log_error("  elem %lu: expected dst[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)expected_off,
                    sv,
                    dv);
        errors++;
      }
    } else {
      uint64_t sv = ((uint64_t*)src)[i];
      uint64_t dv = ((uint64_t*)dst)[expected_off];
      if (dv != sv) {
        if (errors < 5)
          log_error("  elem %lu: expected dst[%lu]=%llu, got %llu",
                    (unsigned long)i,
                    (unsigned long)expected_off,
                    (unsigned long long)sv,
                    (unsigned long long)dv);
        errors++;
      }
    }
  }

  if (errors > 0) {
    log_error("  %d mismatches (of %lu elements)",
              errors,
              (unsigned long)epoch_elements);
    goto Fail;
  }

  free(src);
  free(dst);
  log_info("  PASS");
  return 0;

Fail:
  free(src);
  free(dst);
  log_error("  FAIL");
  return 1;
}

// Test with i_offset > 0 (multi-call accumulation).
static int
run_offset_test(const char* name,
                int rank,
                const uint64_t* dim_sizes,
                const uint64_t* chunk_sizes,
                uint8_t bpe)
{
  log_info("=== %s ===", name);

  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];
  uint64_t chunk_elements, chunk_stride, chunks_per_epoch, epoch_elements;

  build_lifted_layout(rank,
                      dim_sizes,
                      chunk_sizes,
                      NULL,
                      &lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      &chunk_elements,
                      &chunk_stride,
                      &chunks_per_epoch,
                      &epoch_elements);

  uint64_t pool_elements = chunks_per_epoch * chunk_stride;
  size_t src_bytes = epoch_elements * bpe;
  size_t dst_bytes = pool_elements * bpe;

  // Fill full source
  void* full_src = malloc(src_bytes);
  void* dst_full = calloc(1, dst_bytes);
  void* dst_split = calloc(1, dst_bytes);
  CHECK(Fail, full_src && dst_full && dst_split);

  {
    uint16_t* p = (uint16_t*)full_src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = (uint16_t)(i & 0xFFFF);
  }

  // Reference: single call with offset=0
  CHECK(Fail,
        transpose_cpu(dst_full,
                      full_src,
                      src_bytes,
                      bpe,
                      0,
                      lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      0) == 0);

  // Split: two calls with offset
  uint64_t split = epoch_elements / 3;
  size_t split_bytes = split * bpe;

  CHECK(Fail,
        transpose_cpu(dst_split,
                      full_src,
                      split_bytes,
                      bpe,
                      0,
                      lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      0) == 0);
  CHECK(Fail,
        transpose_cpu(dst_split,
                      (const char*)full_src + split_bytes,
                      src_bytes - split_bytes,
                      bpe,
                      split,
                      lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      0) == 0);

  // Compare
  if (memcmp(dst_full, dst_split, dst_bytes) != 0) {
    log_error("  split result differs from single-call result");
    goto Fail;
  }

  free(full_src);
  free(dst_full);
  free(dst_split);
  log_info("  PASS");
  return 0;

Fail:
  free(full_src);
  free(dst_full);
  free(dst_split);
  log_error("  FAIL");
  return 1;
}

static int
test_2d(void)
{
  uint64_t dim[] = { 4, 6 };
  uint64_t chunk[] = { 2, 3 };
  return run_test("cpu_transpose_2d", 2, dim, chunk, NULL, 2);
}

static int
test_3d(void)
{
  uint64_t dim[] = { 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 3 };
  return run_test("cpu_transpose_3d", 3, dim, chunk, NULL, 2);
}

static int
test_identity(void)
{
  uint64_t dim[] = { 6, 4 };
  uint64_t chunk[] = { 6, 4 };
  return run_test("cpu_transpose_identity", 2, dim, chunk, NULL, 2);
}

static int
test_bpe1(void)
{
  uint64_t dim[] = { 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 3 };
  return run_test("cpu_transpose_bpe1", 3, dim, chunk, NULL, 1);
}

static int
test_bpe4(void)
{
  uint64_t dim[] = { 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 3 };
  return run_test("cpu_transpose_bpe4", 3, dim, chunk, NULL, 4);
}

static int
test_bpe8(void)
{
  uint64_t dim[] = { 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 3 };
  return run_test("cpu_transpose_bpe8", 3, dim, chunk, NULL, 8);
}

static int
test_3d_storage_order(void)
{
  uint64_t dim[] = { 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 3 };
  uint8_t order[] = { 0, 2, 1 };
  return run_test("cpu_transpose_3d_storage_order", 3, dim, chunk, order, 2);
}

static int
test_4d_storage_order(void)
{
  uint64_t dim[] = { 2, 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 2, 3 };
  uint8_t order[] = { 0, 3, 1, 2 };
  return run_test("cpu_transpose_4d_storage_order", 4, dim, chunk, order, 2);
}

static int
test_offset(void)
{
  uint64_t dim[] = { 4, 4, 6 };
  uint64_t chunk[] = { 2, 2, 3 };
  return run_offset_test("cpu_transpose_offset", 3, dim, chunk, 2);
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;

  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "2d", test_2d },
    { "3d", test_3d },
    { "identity", test_identity },
    { "bpe1", test_bpe1 },
    { "bpe4", test_bpe4 },
    { "bpe8", test_bpe8 },
    { "3d_storage_order", test_3d_storage_order },
    { "4d_storage_order", test_4d_storage_order },
    { "offset", test_offset },
  };

  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    if (tests[i].fn()) {
      log_error("  FAIL: %s", tests[i].name);
      rc = 1;
    }
  }

  return rc;
}
