#include "index.ops.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "transpose.h"
#include <stdlib.h>
#include <string.h>

// Build lifted shape and strides for a tile decomposition, same algorithm as
// stream.c's init_layout(). The epoch dimension (dim 0) stride is set to 0
// so all epochs collapse into the same pool.
static void
build_lifted_layout(int rank,
                    const uint64_t* dim_sizes,
                    const uint64_t* tile_sizes,
                    const uint8_t* storage_order, // NULL = identity
                    uint8_t bpe,
                    uint8_t* out_lifted_rank,
                    uint64_t* lifted_shape,
                    int64_t* lifted_strides,
                    uint64_t* out_tile_elements,
                    uint64_t* out_tile_stride,
                    uint64_t* out_tiles_per_epoch)
{
  *out_lifted_rank = (uint8_t)(2 * rank);
  uint64_t tile_elements = 1;
  uint64_t tile_count[MAX_RANK];

  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dim_sizes[i], tile_sizes[i]);
    lifted_shape[2 * i] = tile_count[i];
    lifted_shape[2 * i + 1] = tile_sizes[i];
    tile_elements *= tile_sizes[i];
  }

  size_t tile_bytes = tile_elements * bpe;
  // No codec alignment needed for direct transpose tests — use tile_elements.
  uint64_t tile_stride = tile_bytes / bpe;
  (void)tile_bytes;

  compute_lifted_strides(
    rank, tile_sizes, tile_count, storage_order, (int64_t)tile_stride,
    lifted_strides);

  *out_tiles_per_epoch = (uint64_t)lifted_strides[0] / tile_stride;
  lifted_strides[0] = 0; // collapse epoch dim

  *out_tile_elements = tile_elements;
  *out_tile_stride = tile_stride;
}

// Run transpose kernel and verify against CPU ravel() reference.
// dim_sizes/tile_sizes: per-dimension sizes.
// bpe: bytes per element (2 or 4).
// Returns 0 on success.
static int
run_transpose_test(const char* name,
                   int rank,
                   const uint64_t* dim_sizes,
                   const uint64_t* tile_sizes,
                   const uint8_t* storage_order,
                   uint8_t bpe)
{
  log_info("=== %s ===", name);

  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];
  uint64_t tile_elements, tile_stride, tiles_per_epoch;

  build_lifted_layout(rank,
                      dim_sizes,
                      tile_sizes,
                      storage_order,
                      bpe,
                      &lifted_rank,
                      lifted_shape,
                      lifted_strides,
                      &tile_elements,
                      &tile_stride,
                      &tiles_per_epoch);

  uint64_t epoch_elements = tiles_per_epoch * tile_elements;
  uint64_t pool_elements = tiles_per_epoch * tile_stride;
  size_t src_bytes = epoch_elements * bpe;
  size_t dst_bytes = pool_elements * bpe;

  log_info("  rank=%d lifted_rank=%d tile_elements=%lu tile_stride=%lu "
           "tiles_per_epoch=%lu epoch_elements=%lu",
           rank,
           lifted_rank,
           (unsigned long)tile_elements,
           (unsigned long)tile_stride,
           (unsigned long)tiles_per_epoch,
           (unsigned long)epoch_elements);

  void* h_src = NULL;
  void* h_dst = NULL;
  CUdeviceptr d_src = 0, d_dst = 0, d_shape = 0, d_strides = 0;
  CUstream stream = 0;
  int ok = 0;

  h_src = malloc(src_bytes);
  h_dst = calloc(1, dst_bytes);
  CHECK(Fail, h_src && h_dst);

  // Fill source with sequential values
  if (bpe == 2) {
    uint16_t* p = (uint16_t*)h_src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = (uint16_t)(i & 0xFFFF);
  } else {
    uint32_t* p = (uint32_t*)h_src;
    for (uint64_t i = 0; i < epoch_elements; ++i)
      p[i] = (uint32_t)i;
  }

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_src, src_bytes));
  CU(Fail, cuMemAlloc(&d_dst, dst_bytes));
  CU(Fail, cuMemsetD8(d_dst, 0, dst_bytes));
  CU(Fail, cuMemAlloc(&d_shape, lifted_rank * sizeof(uint64_t)));
  CU(Fail, cuMemAlloc(&d_strides, lifted_rank * sizeof(int64_t)));

  CU(Fail, cuMemcpyHtoD(d_src, h_src, src_bytes));
  CU(Fail, cuMemcpyHtoD(d_shape, lifted_shape, lifted_rank * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(d_strides, lifted_strides, lifted_rank * sizeof(int64_t)));

  transpose(d_dst,
            d_src,
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

    if (bpe == 2) {
      uint16_t src_val = ((uint16_t*)h_src)[i];
      uint16_t dst_val = ((uint16_t*)h_dst)[expected_off];
      if (dst_val != src_val) {
        if (errors < 5)
          log_error("  elem %lu: expected dst[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)expected_off,
                    src_val,
                    dst_val);
        errors++;
      }
    } else {
      uint32_t src_val = ((uint32_t*)h_src)[i];
      uint32_t dst_val = ((uint32_t*)h_dst)[expected_off];
      if (dst_val != src_val) {
        if (errors < 5)
          log_error("  elem %lu: expected dst[%lu]=%u, got %u",
                    (unsigned long)i,
                    (unsigned long)expected_off,
                    src_val,
                    dst_val);
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
  // 2D: 4×6 with tile 2×3
  uint64_t dim_sizes[] = { 4, 6 };
  uint64_t tile_sizes[] = { 2, 3 };
  return run_transpose_test(
    "test_transpose_2d", 2, dim_sizes, tile_sizes, NULL, 2);
}

static int
test_transpose_3d(void)
{
  // 3D: matching test_stream's shape (4, 4, 6) tile (2, 2, 3)
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t tile_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_3d", 3, dim_sizes, tile_sizes, NULL, 2);
}

static int
test_transpose_identity(void)
{
  // 2D: 6×4 with tile 6×4 — single tile, identity layout
  uint64_t dim_sizes[] = { 6, 4 };
  uint64_t tile_sizes[] = { 6, 4 };
  return run_transpose_test(
    "test_transpose_identity", 2, dim_sizes, tile_sizes, NULL, 2);
}

static int
test_transpose_bpe4(void)
{
  // 3D with 4-byte elements (u32)
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t tile_sizes[] = { 2, 2, 3 };
  return run_transpose_test(
    "test_transpose_bpe4", 3, dim_sizes, tile_sizes, NULL, 4);
}

static int
test_transpose_3d_storage_order(void)
{
  // 3D with storage_order={0,2,1}: storage dims are [z,x,y]
  // Acquisition: z=4, y=4, x=6, tiles 2,2,3
  uint64_t dim_sizes[] = { 4, 4, 6 };
  uint64_t tile_sizes[] = { 2, 2, 3 };
  uint8_t storage_order[] = { 0, 2, 1 };
  return run_transpose_test("test_transpose_3d_storage_order",
                            3,
                            dim_sizes,
                            tile_sizes,
                            storage_order,
                            2);
}

static int
test_transpose_4d_storage_order(void)
{
  // 4D with storage_order={0,3,1,2}: stress test
  // Acquisition: t=2, z=4, y=4, x=6, tiles 2,2,2,3
  uint64_t dim_sizes[] = { 2, 4, 4, 6 };
  uint64_t tile_sizes[] = { 2, 2, 2, 3 };
  uint8_t storage_order[] = { 0, 3, 1, 2 };
  return run_transpose_test("test_transpose_4d_storage_order",
                            4,
                            dim_sizes,
                            tile_sizes,
                            storage_order,
                            2);
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

  ecode |= test_transpose_2d();
  ecode |= test_transpose_3d();
  ecode |= test_transpose_identity();
  ecode |= test_transpose_bpe4();
  ecode |= test_transpose_3d_storage_order();
  ecode |= test_transpose_4d_storage_order();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
