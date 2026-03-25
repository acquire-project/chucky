#include "gpu/compress.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"
#include <nvcomp/lz4.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#include "test_runner.h"

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

  const size_t chunk_bytes = 1048576; // 1 MiB
  const size_t n_chunks = 96;
  const size_t pool_bytes = n_chunks * chunk_bytes;

  struct codec c = { 0 };
  uint16_t* h_data = NULL;
  uint8_t* h_compressed = NULL;
  size_t* h_comp_sizes = NULL;
  uint8_t* decomp_buf = NULL;
  void* d_data = NULL;
  void* d_compressed = NULL;
  CUstream stream = 0;
  int ok = 0;

  CHECK(Fail, codec_init(&c, CODEC_ZSTD, chunk_bytes, n_chunks) == 0);

  const size_t comp_pool = n_chunks * c.max_output_size;

  log_info("  chunk_bytes=%zu n_chunks=%zu max_comp=%zu",
           chunk_bytes,
           n_chunks,
           c.max_output_size);

  h_data = (uint16_t*)malloc(pool_bytes);

  CHECK(Fail, h_data);
  h_compressed = (uint8_t*)malloc(comp_pool);
  CHECK(Fail, h_compressed);
  h_comp_sizes = (size_t*)malloc(n_chunks * sizeof(size_t));
  CHECK(Fail, h_comp_sizes);
  decomp_buf = (uint8_t*)malloc(chunk_bytes);
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
          codec_compress(&c, d_data, chunk_bytes, d_compressed, 0, stream) ==
            0);

    // Wait for compress to finish, then D2H
    CU(Fail, cuStreamSynchronize(stream));
    CU(Fail, cuMemcpyDtoH(h_compressed, (CUdeviceptr)d_compressed, comp_pool));
    CU(Fail,
       cuMemcpyDtoH(
         h_comp_sizes, (CUdeviceptr)c.d_comp_sizes, n_chunks * sizeof(size_t)));

    // Verify: decompress each chunk and compare
    int round_errors = 0;
    for (size_t t = 0; t < n_chunks; ++t) {
      const uint8_t* comp_chunk = h_compressed + t * c.max_output_size;
      size_t result =
        ZSTD_decompress(decomp_buf, chunk_bytes, comp_chunk, h_comp_sizes[t]);
      if (ZSTD_isError(result)) {
        log_error("  round %d chunk %zu: ZSTD_decompress failed: %s",
                  round,
                  t,
                  ZSTD_getErrorName(result));
        round_errors++;
        continue;
      }
      if (result != chunk_bytes) {
        log_error("  round %d chunk %zu: size mismatch: expected %zu got %zu",
                  round,
                  t,
                  chunk_bytes,
                  result);
        round_errors++;
        continue;
      }

      const uint16_t* actual = (const uint16_t*)decomp_buf;
      const uint16_t* expected = h_data + t * (chunk_bytes / sizeof(uint16_t));
      int mismatch = 0;
      for (size_t e = 0; e < chunk_bytes / sizeof(uint16_t); ++e) {
        if (actual[e] != expected[e]) {
          if (mismatch == 0)
            log_error("  round %d chunk %zu elem %zu: expected %u got %u",
                      round,
                      t,
                      e,
                      expected[e],
                      actual[e]);
          mismatch++;
        }
      }
      if (mismatch > 0) {
        log_error("  round %d chunk %zu: %d mismatches", round, t, mismatch);
        round_errors++;
      }
    }

    if (round_errors > 0) {
      log_error("  round %d: %d chunk errors", round, round_errors);
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

// LZ4 roundtrip: compress with codec_compress(CODEC_LZ4), decompress with
// nvcomp LZ4 decompress on GPU, compare with original.
static int
test_compress_lz4_roundtrip(void)
{
  log_info("=== test_compress_lz4_roundtrip ===");

  const size_t chunk_bytes = 1048576; // 1 MiB
  const size_t n_chunks = 96;
  const size_t pool_bytes = n_chunks * chunk_bytes;

  struct codec c = { 0 };
  uint16_t* h_data = NULL;
  uint8_t* h_result = NULL;
  size_t* h_comp_sizes = NULL;
  void* d_data = NULL;
  void* d_compressed = NULL;
  void* d_decomp = NULL;
  void* d_temp = NULL;
  CUstream stream = 0;
  int ok = 0;

  CHECK(Fail, codec_init(&c, CODEC_LZ4, chunk_bytes, n_chunks) == 0);

  const size_t comp_pool = n_chunks * c.max_output_size;

  log_info("  chunk_bytes=%zu n_chunks=%zu max_comp=%zu",
           chunk_bytes,
           n_chunks,
           c.max_output_size);

  h_data = (uint16_t*)malloc(pool_bytes);
  CHECK(Fail, h_data);
  h_result = (uint8_t*)malloc(pool_bytes);
  CHECK(Fail, h_result);
  h_comp_sizes = (size_t*)malloc(n_chunks * sizeof(size_t));
  CHECK(Fail, h_comp_sizes);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_data, pool_bytes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_compressed, comp_pool));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_decomp, pool_bytes));

  // Fill host data
  {
    const size_t elems = pool_bytes / sizeof(uint16_t);
    for (size_t i = 0; i < elems; ++i)
      h_data[i] = source_value_at(i, elems);
  }

  // H2D
  CU(Fail, cuMemcpyHtoD((CUdeviceptr)d_data, h_data, pool_bytes));

  // Compress
  CHECK(Fail,
        codec_compress(&c, d_data, chunk_bytes, d_compressed, 0, stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));

  // Read compressed sizes
  CU(Fail,
     cuMemcpyDtoH(
       h_comp_sizes, (CUdeviceptr)c.d_comp_sizes, n_chunks * sizeof(size_t)));

  // Set up nvcomp LZ4 batch decompress on GPU
  // Build pointer arrays for decompress: compressed input ptrs, decompressed
  // output ptrs, compressed sizes, decompressed sizes.
  {
    void** h_comp_ptrs = (void**)malloc(n_chunks * sizeof(void*));
    void** h_decomp_ptrs = (void**)malloc(n_chunks * sizeof(void*));
    size_t* h_uncomp_sizes = (size_t*)malloc(n_chunks * sizeof(size_t));
    size_t* h_actual_uncomp_sizes = (size_t*)malloc(n_chunks * sizeof(size_t));
    int* h_statuses = (int*)calloc(n_chunks, sizeof(int));
    CHECK(Fail,
          h_comp_ptrs && h_decomp_ptrs && h_uncomp_sizes &&
            h_actual_uncomp_sizes && h_statuses);

    for (size_t t = 0; t < n_chunks; ++t) {
      h_comp_ptrs[t] = (uint8_t*)d_compressed + t * c.max_output_size;
      h_decomp_ptrs[t] = (uint8_t*)d_decomp + t * chunk_bytes;
      h_uncomp_sizes[t] = chunk_bytes;
    }

    // Allocate device arrays for nvcomp decompress
    void** d_comp_ptrs = NULL;
    void** d_decomp_ptrs = NULL;
    size_t* d_comp_sizes_arr = NULL;
    size_t* d_uncomp_sizes = NULL;
    size_t* d_actual_uncomp_sizes = NULL;
    int* d_statuses = NULL;

    CU(Fail2, cuMemAlloc((CUdeviceptr*)&d_comp_ptrs, n_chunks * sizeof(void*)));
    CU(Fail2,
       cuMemAlloc((CUdeviceptr*)&d_decomp_ptrs, n_chunks * sizeof(void*)));
    CU(Fail2,
       cuMemAlloc((CUdeviceptr*)&d_comp_sizes_arr, n_chunks * sizeof(size_t)));
    CU(Fail2,
       cuMemAlloc((CUdeviceptr*)&d_uncomp_sizes, n_chunks * sizeof(size_t)));
    CU(Fail2,
       cuMemAlloc((CUdeviceptr*)&d_actual_uncomp_sizes,
                  n_chunks * sizeof(size_t)));
    CU(Fail2, cuMemAlloc((CUdeviceptr*)&d_statuses, n_chunks * sizeof(int)));

    CU(Fail2,
       cuMemcpyHtoD(
         (CUdeviceptr)d_comp_ptrs, h_comp_ptrs, n_chunks * sizeof(void*)));
    CU(Fail2,
       cuMemcpyHtoD(
         (CUdeviceptr)d_decomp_ptrs, h_decomp_ptrs, n_chunks * sizeof(void*)));
    CU(Fail2,
       cuMemcpyHtoD((CUdeviceptr)d_comp_sizes_arr,
                    h_comp_sizes,
                    n_chunks * sizeof(size_t)));
    CU(Fail2,
       cuMemcpyHtoD((CUdeviceptr)d_uncomp_sizes,
                    h_uncomp_sizes,
                    n_chunks * sizeof(size_t)));

    // Get temp buffer size for LZ4 decompress
    size_t temp_bytes = 0;
    nvcompBatchedLZ4DecompressOpts_t decomp_opts =
      nvcompBatchedLZ4DecompressDefaultOpts;
    {
      nvcompStatus_t status =
        nvcompBatchedLZ4DecompressGetTempSizeAsync(n_chunks,
                                                   chunk_bytes,
                                                   decomp_opts,
                                                   &temp_bytes,
                                                   n_chunks * chunk_bytes);
      CHECK(Fail2, status == nvcompSuccess);
    }

    if (temp_bytes == 0)
      temp_bytes = 1; // cuMemAlloc requires size > 0
    CU(Fail2, cuMemAlloc((CUdeviceptr*)&d_temp, temp_bytes));

    // Run LZ4 decompress
    {
      nvcompStatus_t status =
        nvcompBatchedLZ4DecompressAsync((const void* const*)d_comp_ptrs,
                                        d_comp_sizes_arr,
                                        d_uncomp_sizes,
                                        d_actual_uncomp_sizes,
                                        n_chunks,
                                        d_temp,
                                        temp_bytes,
                                        (void* const*)d_decomp_ptrs,
                                        decomp_opts,
                                        (nvcompStatus_t*)d_statuses,
                                        (cudaStream_t)stream);
      CHECK(Fail2, status == nvcompSuccess);
    }

    CU(Fail2, cuStreamSynchronize(stream));

    // D2H decompressed data
    CU(Fail2, cuMemcpyDtoH(h_result, (CUdeviceptr)d_decomp, pool_bytes));

    // Verify
    int errors = 0;
    const uint16_t* original = h_data;
    const uint16_t* result = (const uint16_t*)h_result;
    const size_t total_elems = pool_bytes / sizeof(uint16_t);
    for (size_t i = 0; i < total_elems; ++i) {
      if (original[i] != result[i]) {
        if (errors < 5)
          log_error(
            "  elem %zu: expected %u got %u", i, original[i], result[i]);
        errors++;
      }
    }

    if (errors > 0) {
      log_error("  %d mismatches", errors);
      goto Fail2;
    }

    ok = 1;

  Fail2:
    cuMemFree((CUdeviceptr)d_comp_ptrs);
    cuMemFree((CUdeviceptr)d_decomp_ptrs);
    cuMemFree((CUdeviceptr)d_comp_sizes_arr);
    cuMemFree((CUdeviceptr)d_uncomp_sizes);
    cuMemFree((CUdeviceptr)d_actual_uncomp_sizes);
    cuMemFree((CUdeviceptr)d_statuses);
    free(h_comp_ptrs);
    free(h_decomp_ptrs);
    free(h_uncomp_sizes);
    free(h_actual_uncomp_sizes);
    free(h_statuses);
  }

Fail:
  free(h_data);
  free(h_result);
  free(h_comp_sizes);
  cuMemFree((CUdeviceptr)d_data);
  cuMemFree((CUdeviceptr)d_compressed);
  cuMemFree((CUdeviceptr)d_decomp);
  cuMemFree((CUdeviceptr)d_temp);
  cuStreamDestroy(stream);
  codec_free(&c);

  if (ok) {
    log_info("  PASS");
    return 0;
  }
  log_error("  FAIL");
  return 1;
}

RUN_GPU_TESTS({ "compress_roundtrip", test_compress_roundtrip },
              { "compress_lz4_roundtrip", test_compress_lz4_roundtrip }, )
