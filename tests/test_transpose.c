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
// n_epochs: epochs worth of source to hand the kernel in one call.
// in_epoch_offset: how far into its epoch the first element sits.
// Returns 0 on success.
static int
run_transpose_test(const char* name,
                   int rank,
                   uint8_t n_append,
                   const uint64_t* dim_sizes,
                   const uint64_t* chunk_sizes,
                   const uint8_t* storage_order,
                   uint8_t bpe,
                   uint32_t n_epochs,
                   uint64_t in_epoch_offset)
{
  log_info("=== %s (bpe=%u epochs=%u from=%lu) ===",
           name,
           bpe,
           n_epochs,
           (unsigned long)in_epoch_offset);

  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];
  uint64_t chunk_elements, chunk_stride, chunks_per_epoch, epoch_elements;

  build_lifted_layout(rank,
                      n_append,
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
  const uint64_t region_elements =
    chunks_per_epoch * chunk_stride + TEST_REGION_PAD_ELEMENTS;
  const uint64_t src_elements = n_epochs * epoch_elements;
  const size_t src_bytes = src_elements * bpe;
  // A start partway into an epoch pushes the last elements into one more region
  // than the epochs they cover.
  const uint32_t regions = n_epochs + 1;
  const size_t dst_bytes = regions * region_elements * bpe;

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
  CUdeviceptr d_src = 0, d_dst = 0;
  CUstream stream = 0;
  int ok = 0;

  h_src = malloc(src_bytes);
  h_dst = calloc(1, dst_bytes);
  CHECK(Fail, h_src && h_dst);
  fill_elements((uint8_t*)h_src, src_elements, bpe);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_src, src_bytes));
  CU(Fail, cuMemAlloc(&d_dst, dst_bytes));
  // On the kernel's own stream, so the scatter cannot start before the source
  // lands or race the clear.
  CU(Fail, cuMemsetD8Async(d_dst, 0, dst_bytes, stream));
  CU(Fail, cuMemcpyHtoDAsync(d_src, h_src, src_bytes, stream));

  CHECK(Fail,
        transpose(d_dst,
                  d_src,
                  src_bytes,
                  bpe,
                  in_epoch_offset,
                  epoch_elements,
                  region_elements * bpe,
                  lifted_rank,
                  lifted_shape,
                  lifted_strides,
                  stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));

  CU(Fail, cuMemcpyDtoH(h_dst, d_dst, dst_bytes));

  // Verify against CPU ravel reference
  int errors = 0;
  for (uint64_t i = 0; i < src_elements; ++i) {
    const uint64_t expected_off = expected_scatter_offset(lifted_rank,
                                                          lifted_shape,
                                                          lifted_strides,
                                                          epoch_elements,
                                                          region_elements,
                                                          in_epoch_offset + i);
    if (memcmp((uint8_t*)h_dst + expected_off * bpe,
               (uint8_t*)h_src + i * bpe,
               bpe) != 0) {
      if (errors < 5)
        log_error("  elem %lu: dst[%lu] does not match source",
                  (unsigned long)(in_epoch_offset + i),
                  (unsigned long)expected_off);
      errors++;
    }
  }

  if (errors > 0) {
    log_error(
      "  %d mismatches (of %lu elements)", errors, (unsigned long)src_elements);
    goto Fail;
  }

  ok = 1;

Fail:
  free(h_src);
  free(h_dst);
  cuMemFree(d_src);
  cuMemFree(d_dst);
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
    "test_transpose_2d", 2, 1, dim_sizes, chunk_sizes, NULL, 2, 1, 0);
}

static int
test_transpose_3d(void)
{
  // 3D: matching test_stream's shape (4, 4, 6) chunk (2, 2, 3)
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_3d", 3, 1, dim_sizes, chunk_sizes, NULL, 2, 1, 0);
}

static int
test_transpose_identity(void)
{
  // 2D: 6×4 with chunk 6×4 — single chunk, identity layout
  uint64_t dim_sizes[] = { 6, 4 };
  uint64_t chunk_sizes[] = { 6, 4 };
  return run_transpose_test(
    "test_transpose_identity", 2, 1, dim_sizes, chunk_sizes, NULL, 2, 1, 0);
}

static int
test_transpose_bpe4(void)
{
  // 3D with 4-byte elements (u32)
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_bpe4", 3, 1, dim_sizes, chunk_sizes, NULL, 4, 1, 0);
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
                            1,
                            dim_sizes,
                            chunk_sizes,
                            storage_order,
                            2,
                            1,
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
                            1,
                            dim_sizes,
                            chunk_sizes,
                            storage_order,
                            2,
                            1,
                            0);
}

// One call covering several epochs, each landing in its own destination region.
// The shape spans several blocks, so blocks on either side of an epoch boundary
// have to find the boundary themselves.
static int
test_transpose_many_epochs(void)
{
  uint64_t dim_sizes[] = { 4, 64, 96 };
  uint64_t chunk_sizes[] = { 2, 16, 24 };
  int err = 0;
  for (uint8_t bpe = 1; bpe <= 8; bpe = (uint8_t)(bpe * 2))
    for (uint64_t from = 0; from < 2; ++from)
      err |= run_transpose_test("test_transpose_many_epochs",
                                3,
                                1,
                                dim_sizes,
                                chunk_sizes,
                                NULL,
                                bpe,
                                3,
                                from);
  return err;
}

static int
test_transpose_bpe8(void)
{
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t chunk_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_bpe8", 3, 1, dim_sizes, chunk_sizes, NULL, 8, 1, 0);
}

// Two append dimensions collapse, so the decomposition has to stop short of a
// dimension sitting in the middle of the shape rather than at the front.
static int
test_transpose_two_append_dims(void)
{
  uint64_t dim_sizes[] = { 4, 3, 64, 96 };
  uint64_t chunk_sizes[] = { 1, 1, 16, 24 };
  return run_transpose_test("test_transpose_two_append_dims",
                            4,
                            2,
                            dim_sizes,
                            chunk_sizes,
                            NULL,
                            2,
                            2,
                            0);
}

// A shape whose trailing dimensions cannot make up an epoch is refused, rather
// than scattered somewhere wrong.
static int
test_transpose_rejects_mismatched_epoch(void)
{
  log_info("=== test_transpose_rejects_mismatched_epoch ===");

  const uint64_t shape[] = { 2, 4, 6 };
  const int64_t strides[] = { 0, 6, 1 };
  CUdeviceptr d_src = 0, d_dst = 0;
  CUstream stream = 0;
  int ok = 0;

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_src, 48 * sizeof(uint16_t)));
  CU(Fail, cuMemAlloc(&d_dst, 48 * sizeof(uint16_t)));

  // 25 is not a product of any run of trailing extents.
  ok = transpose(d_dst,
                 d_src,
                 48 * sizeof(uint16_t),
                 2,
                 0,
                 25,
                 24 * sizeof(uint16_t),
                 3,
                 shape,
                 strides,
                 stream) != 0;

Fail:
  cuMemFree(d_src);
  cuMemFree(d_dst);
  cuStreamDestroy(stream);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "transpose_2d", test_transpose_2d },
              { "transpose_3d", test_transpose_3d },
              { "transpose_identity", test_transpose_identity },
              { "transpose_bpe4", test_transpose_bpe4 },
              { "transpose_3d_storage_order", test_transpose_3d_storage_order },
              { "transpose_4d_storage_order", test_transpose_4d_storage_order },
              { "transpose_bpe8", test_transpose_bpe8 },
              { "transpose_many_epochs", test_transpose_many_epochs },
              { "transpose_two_append_dims", test_transpose_two_append_dims },
              { "transpose_rejects_mismatched_epoch",
                test_transpose_rejects_mismatched_epoch }, )
