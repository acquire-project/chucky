#include "gpu/prelude.cuda.h"
#include "gpu/transpose.h"
#include "index.ops.util.h"
#include "util/prelude.h"
#include <stdlib.h>
#include <string.h>

#include "test_runner.h"

static void
fill_elements(uint8_t* dst, uint64_t n, uint8_t bpe)
{
  for (uint64_t i = 0; i < n; ++i)
    for (uint8_t b = 0; b < bpe; ++b)
      dst[i * bpe + b] = (uint8_t)((i * 131 + b * 17 + 1) & 0xFF);
}

// Run transpose kernel and verify against CPU ravel() reference.
// dim_sizes/chunk_sizes: per-dimension sizes.
// bpe: bytes per element.
// element_offset: where the source sits in its buffer, so the kernel can be
// given a start that is not word aligned.
// Returns 0 on success.
static int
run_transpose_test(const char* name,
                   int rank,
                   const uint64_t* dim_sizes,
                   const uint64_t* chunk_sizes,
                   const uint8_t* storage_order,
                   uint8_t bpe,
                   uint64_t element_offset)
{
  log_info(
    "=== %s (bpe=%u offset=%lu) ===", name, bpe, (unsigned long)element_offset);

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

  void* h_src = NULL;
  void* h_dst = NULL;
  CUdeviceptr d_src = 0, d_dst = 0, d_shape = 0, d_strides = 0;
  CUstream stream = 0;
  int ok = 0;

  const size_t src_alloc =
    element_offset * bpe + src_bytes + TRANSPOSE_SOURCE_PAD_BYTES;

  h_src = malloc(src_bytes);
  h_dst = calloc(1, dst_bytes);
  CHECK(Fail, h_src && h_dst);
  fill_elements((uint8_t*)h_src, epoch_elements, bpe);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_src, src_alloc));
  CU(Fail, cuMemAlloc(&d_dst, dst_bytes));
  CU(Fail, cuMemsetD8(d_src, 0, src_alloc));
  CU(Fail, cuMemsetD8(d_dst, 0, dst_bytes));
  CU(Fail, cuMemAlloc(&d_shape, lifted_rank * sizeof(uint64_t)));
  CU(Fail, cuMemAlloc(&d_strides, lifted_rank * sizeof(int64_t)));

  CU(Fail, cuMemcpyHtoD(d_src + element_offset * bpe, h_src, src_bytes));
  CU(Fail, cuMemcpyHtoD(d_shape, lifted_shape, lifted_rank * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(d_strides, lifted_strides, lifted_rank * sizeof(int64_t)));

  transpose(d_dst,
            d_src + element_offset * bpe,
            src_bytes,
            bpe,
            0,
            lifted_rank,
            (const uint64_t*)d_shape,
            (const int64_t*)d_strides,
            stream);
  CU(Fail, cuStreamSynchronize(stream));

  CU(Fail, cuMemcpyDtoH(h_dst, d_dst, dst_bytes));

  // Verify against CPU ravel reference
  int errors = 0;
  for (uint64_t i = 0; i < epoch_elements; ++i) {
    uint64_t expected_off = ravel(lifted_rank, lifted_shape, lifted_strides, i);
    if (memcmp((uint8_t*)h_dst + expected_off * bpe,
               (uint8_t*)h_src + i * bpe,
               bpe) != 0) {
      if (errors < 5)
        log_error("  elem %lu: dst[%lu] does not match source",
                  (unsigned long)i,
                  (unsigned long)expected_off);
      errors++;
    }
  }

  if (errors > 0) {
    log_error("  %d mismatches (of %lu elements)",
              errors,
              (unsigned long)epoch_elements);
    goto Fail;
  }

  ok = 1;

Fail:
  free(h_src);
  free(h_dst);
  cuMemFree(d_src);
  cuMemFree(d_dst);
  cuMemFree(d_shape);
  cuMemFree(d_strides);
  cuStreamDestroy(stream);

  if (ok) {
    log_info("  PASS");
    return 0;
  }
  log_error("  FAIL");
  return 1;
}

static int
test_transpose_2d(void)
{
  // 2D: 4×6 with chunk 2×3
  uint64_t dim_sizes[] = { 4, 6 };
  uint64_t chunk_sizes[] = { 2, 3 };
  return run_transpose_test(
    "test_transpose_2d", 2, dim_sizes, chunk_sizes, NULL, 2, 0);
}

static int
test_transpose_3d(void)
{
  // 3D: matching test_stream's shape (4, 4, 6) chunk (2, 2, 3)
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_3d", 3, dim_sizes, chunk_sizes, NULL, 2, 0);
}

static int
test_transpose_identity(void)
{
  // 2D: 6×4 with chunk 6×4 — single chunk, identity layout
  uint64_t dim_sizes[] = { 6, 4 };
  uint64_t chunk_sizes[] = { 6, 4 };
  return run_transpose_test(
    "test_transpose_identity", 2, dim_sizes, chunk_sizes, NULL, 2, 0);
}

static int
test_transpose_bpe4(void)
{
  // 3D with 4-byte elements (u32)
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_bpe4", 3, dim_sizes, chunk_sizes, NULL, 4, 0);
}

static int
test_transpose_3d_storage_order(void)
{
  // 3D with storage_order={0,2,1}: storage dims are [z,x,y]
  // Acquisition: z=4, y=4, x=6, chunks 2,2,3
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  uint8_t storage_order[] = { 0, 2, 1 };
  return run_transpose_test("test_transpose_3d_storage_order",
                            3,
                            dim_sizes,
                            chunk_sizes,
                            storage_order,
                            2,
                            0);
}

static int
test_transpose_4d_storage_order(void)
{
  // 4D with storage_order={0,3,1,2}: stress test
  // Acquisition: t=2, z=4, y=4, x=6, chunks 2,2,2,3
  uint64_t dim_sizes[] = { 2, 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 2, 3 };
  uint8_t storage_order[] = { 0, 3, 1, 2 };
  return run_transpose_test("test_transpose_4d_storage_order",
                            4,
                            dim_sizes,
                            chunk_sizes,
                            storage_order,
                            2,
                            0);
}

// A scatter split across epochs hands the kernel a start partway into the
// staging buffer, so it has to cope with one that is not word aligned.
static int
test_transpose_unaligned_source(void)
{
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  int err = 0;
  for (uint8_t bpe = 1; bpe <= 2; bpe = (uint8_t)(bpe * 2))
    for (uint64_t off = 0; off < TRANSPOSE_SOURCE_PAD_BYTES / bpe; ++off)
      err |= run_transpose_test("test_transpose_unaligned_source",
                                3,
                                dim_sizes,
                                chunk_sizes,
                                NULL,
                                bpe,
                                off);
  return err;
}

static int
test_transpose_bpe8(void)
{
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_bpe8", 3, dim_sizes, chunk_sizes, NULL, 8, 0);
}

RUN_GPU_TESTS({ "transpose_2d", test_transpose_2d },
              { "transpose_3d", test_transpose_3d },
              { "transpose_identity", test_transpose_identity },
              { "transpose_bpe4", test_transpose_bpe4 },
              { "transpose_3d_storage_order", test_transpose_3d_storage_order },
              { "transpose_4d_storage_order", test_transpose_4d_storage_order },
              { "transpose_bpe8", test_transpose_bpe8 },
              { "transpose_unaligned_source",
                test_transpose_unaligned_source }, )
