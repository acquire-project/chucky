#include "compress.h"
#include "prelude.h"
#include "prelude.cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// Deterministic source data
// Hash-based values: any element is reconstructable from its global index.
static uint16_t
source_value_at(size_t gi, size_t total)
{
  (void)total;
  // Mix of compressible and incompressible:
  //  First 1/3: linear ramp (very compressible)
  //  Middle 1/3: constant 42 (maximally compressible)
  //  Last 1/3: pseudo-random (incompressible)
  if (gi < total / 3)
    return (uint16_t)(gi);
  if (gi < 2 * total / 3)
    return 42;
  return (uint16_t)(gi ^ (gi >> 16));
}

static int
test_compress_roundtrip(void)
{
  log_info("=== test_compress_roundtrip ===");

  const size_t tile_bytes = 1048576; // 1 MiB
  const size_t num_tiles = 96;
  const size_t pool_bytes = num_tiles * tile_bytes;

  struct codec c = { 0 };
  uint16_t* h_data = NULL;
  uint8_t* h_compressed = NULL;
  size_t* h_comp_sizes = NULL;
  uint8_t* decomp_buf = NULL;
  void* d_data = NULL;
  void* d_compressed = NULL;
  CUstream stream = 0;
  int ok = 0;

  CHECK(Fail, codec_init(&c, CODEC_ZSTD, tile_bytes, num_tiles));

  const size_t comp_pool = num_tiles * c.max_output_size;

  log_info("  tile_bytes=%zu num_tiles=%zu max_comp=%zu",
           tile_bytes,
           num_tiles,
           c.max_output_size);

  h_data = (uint16_t*)malloc(pool_bytes);

  CHECK(Fail, h_data);
  h_compressed = (uint8_t*)malloc(comp_pool);
  CHECK(Fail, h_compressed);
  h_comp_sizes = (size_t*)malloc(num_tiles * sizeof(size_t));
  CHECK(Fail, h_comp_sizes);
  decomp_buf = (uint8_t*)malloc(tile_bytes);
  CHECK(Fail, decomp_buf);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_data, pool_bytes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_compressed, comp_pool));

  // Run TWO rounds with different data
  for (int round = 0; round < 2; ++round) {
    // Fill host data with hash seeded by round
    const size_t elems = pool_bytes / sizeof(uint16_t);
    const size_t gi_offset = (size_t)round * elems;
    for (size_t i = 0; i < elems; ++i)
      h_data[i] = source_value_at(gi_offset + i, 2 * elems);

    // H2D
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)d_data, h_data, pool_bytes));

    // Compress
    CHECK(Fail,
          codec_compress(&c, d_data, tile_bytes, d_compressed, 0, stream));

    // Wait for compress to finish, then D2H
    CU(Fail, cuStreamSynchronize(stream));
    CU(Fail, cuMemcpyDtoH(h_compressed, (CUdeviceptr)d_compressed, comp_pool));
    CU(Fail,
       cuMemcpyDtoH(
         h_comp_sizes, (CUdeviceptr)c.d_comp_sizes, num_tiles * sizeof(size_t)));

    // Verify: decompress each tile and compare
    int round_errors = 0;
    for (size_t t = 0; t < num_tiles; ++t) {
      const uint8_t* comp_tile = h_compressed + t * c.max_output_size;
      size_t result =
        ZSTD_decompress(decomp_buf, tile_bytes, comp_tile, h_comp_sizes[t]);
      if (ZSTD_isError(result)) {
        log_error("  round %d tile %zu: ZSTD_decompress failed: %s",
                  round,
                  t,
                  ZSTD_getErrorName(result));
        round_errors++;
        continue;
      }
      if (result != tile_bytes) {
        log_error("  round %d tile %zu: size mismatch: expected %zu got %zu",
                  round,
                  t,
                  tile_bytes,
                  result);
        round_errors++;
        continue;
      }

      const uint16_t* actual = (const uint16_t*)decomp_buf;
      const uint16_t* expected = h_data + t * (tile_bytes / sizeof(uint16_t));
      int mismatch = 0;
      for (size_t e = 0; e < tile_bytes / sizeof(uint16_t); ++e) {
        if (actual[e] != expected[e]) {
          if (mismatch == 0)
            log_error("  round %d tile %zu elem %zu: expected %u got %u",
                      round,
                      t,
                      e,
                      expected[e],
                      actual[e]);
          mismatch++;
        }
      }
      if (mismatch > 0) {
        log_error("  round %d tile %zu: %d mismatches", round, t, mismatch);
        round_errors++;
      }
    }

    if (round_errors > 0) {
      log_error("  round %d: %d tile errors", round, round_errors);
      goto Fail;
    }
    log_info("  round %d: OK", round);
  }

  ok = 1;

Fail:
  free(h_data);
  free(h_compressed);
  free(h_comp_sizes);
  free(decomp_buf);
  cuMemFree((CUdeviceptr)d_data);
  cuMemFree((CUdeviceptr)d_compressed);
  cuStreamDestroy(stream);
  codec_free(&c);

  if (ok) {
    log_info("  PASS");
    return 0;
  }
  log_error("  FAIL");
  return 1;
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

  ecode |= test_compress_roundtrip();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
