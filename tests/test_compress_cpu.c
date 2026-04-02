#include "cpu/compress.h"
#include "util/prelude.h"

#ifdef HAVE_BLOSC
#include <blosc.h>
#endif
#include <lz4.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#define CHUNK_BYTES 4096
#define BATCH_SIZE 16

static void
fill_pattern(void* buf, size_t bytes, uint8_t seed)
{
  uint8_t* p = (uint8_t*)buf;
  for (size_t i = 0; i < bytes; ++i)
    p[i] = (uint8_t)(seed + (i % 251));
}

static int
test_codec_none(void)
{
  log_info("=== test_compress_cpu_none ===");

  void* src = NULL;
  void* dst = NULL;

  size_t max_out = compress_cpu_max_output_size(CODEC_NONE, CHUNK_BYTES);
  CHECK(Fail, max_out == CHUNK_BYTES);

  src = malloc(BATCH_SIZE * CHUNK_BYTES);
  dst = calloc(BATCH_SIZE, max_out);
  size_t comp_sizes[BATCH_SIZE];
  CHECK(Fail, src && dst);

  for (int i = 0; i < BATCH_SIZE; ++i)
    fill_pattern((char*)src + i * CHUNK_BYTES, CHUNK_BYTES, (uint8_t)i);

  CHECK(Fail,
        compress_cpu((struct codec_config){ .id = CODEC_NONE },
                     src,
                     CHUNK_BYTES,
                     dst,
                     max_out,
                     comp_sizes,
                     CHUNK_BYTES,
                     BATCH_SIZE,
                     1) == 0);

  for (int i = 0; i < BATCH_SIZE; ++i) {
    CHECK(Fail, comp_sizes[i] == CHUNK_BYTES);
    CHECK(Fail,
          memcmp((char*)src + i * CHUNK_BYTES,
                 (char*)dst + i * max_out,
                 CHUNK_BYTES) == 0);
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

static int
test_codec_lz4(void)
{
  log_info("=== test_compress_cpu_lz4 ===");

  void* src = NULL;
  void* dst = NULL;

  size_t max_out = compress_cpu_max_output_size(CODEC_LZ4, CHUNK_BYTES);
  CHECK(Fail, max_out > 0);

  src = malloc(BATCH_SIZE * CHUNK_BYTES);
  dst = calloc(BATCH_SIZE, max_out);
  size_t comp_sizes[BATCH_SIZE];
  CHECK(Fail, src && dst);

  for (int i = 0; i < BATCH_SIZE; ++i)
    fill_pattern((char*)src + i * CHUNK_BYTES, CHUNK_BYTES, (uint8_t)i);

  CHECK(Fail,
        compress_cpu((struct codec_config){ .id = CODEC_LZ4, .level = 1 },
                     src,
                     CHUNK_BYTES,
                     dst,
                     max_out,
                     comp_sizes,
                     CHUNK_BYTES,
                     BATCH_SIZE,
                     1) == 0);

  // Decompress and verify round-trip
  void* recovered = malloc(CHUNK_BYTES);
  CHECK(Fail, recovered);

  for (int i = 0; i < BATCH_SIZE; ++i) {
    CHECK(Fail, comp_sizes[i] > 0 && comp_sizes[i] <= max_out);
    int rc = LZ4_decompress_safe((const char*)dst + i * max_out,
                                 (char*)recovered,
                                 (int)comp_sizes[i],
                                 CHUNK_BYTES);
    CHECK(Fail, rc == CHUNK_BYTES);
    CHECK(Fail,
          memcmp((char*)src + i * CHUNK_BYTES, recovered, CHUNK_BYTES) == 0);
  }

  free(src);
  free(dst);
  free(recovered);
  log_info("  PASS");
  return 0;
Fail:
  free(src);
  free(dst);
  log_error("  FAIL");
  return 1;
}

static int
test_codec_zstd(void)
{
  log_info("=== test_compress_cpu_zstd ===");

  void* src = NULL;
  void* dst = NULL;

  size_t max_out = compress_cpu_max_output_size(CODEC_ZSTD, CHUNK_BYTES);
  CHECK(Fail, max_out > 0);

  src = malloc(BATCH_SIZE * CHUNK_BYTES);
  dst = calloc(BATCH_SIZE, max_out);
  size_t comp_sizes[BATCH_SIZE];
  CHECK(Fail, src && dst);

  for (int i = 0; i < BATCH_SIZE; ++i)
    fill_pattern((char*)src + i * CHUNK_BYTES, CHUNK_BYTES, (uint8_t)i);

  CHECK(Fail,
        compress_cpu((struct codec_config){ .id = CODEC_ZSTD },
                     src,
                     CHUNK_BYTES,
                     dst,
                     max_out,
                     comp_sizes,
                     CHUNK_BYTES,
                     BATCH_SIZE,
                     1) == 0);

  // Decompress and verify round-trip
  void* recovered = malloc(CHUNK_BYTES);
  CHECK(Fail, recovered);

  for (int i = 0; i < BATCH_SIZE; ++i) {
    CHECK(Fail, comp_sizes[i] > 0 && comp_sizes[i] <= max_out);
    size_t rc = ZSTD_decompress(
      recovered, CHUNK_BYTES, (const char*)dst + i * max_out, comp_sizes[i]);
    CHECK(Fail, !ZSTD_isError(rc) && rc == CHUNK_BYTES);
    CHECK(Fail,
          memcmp((char*)src + i * CHUNK_BYTES, recovered, CHUNK_BYTES) == 0);
  }

  free(src);
  free(dst);
  free(recovered);
  log_info("  PASS");
  return 0;
Fail:
  free(src);
  free(dst);
  log_error("  FAIL");
  return 1;
}

#ifdef HAVE_BLOSC
static int
test_codec_blosc(enum compression_codec id, const char* name)
{
  log_info("=== test_compress_cpu_%s ===", name);

  void* src = NULL;
  void* dst = NULL;
  void* recovered = NULL;

  size_t max_out = compress_cpu_max_output_size(id, CHUNK_BYTES);
  CHECK(Fail, max_out > 0);

  src = malloc(BATCH_SIZE * CHUNK_BYTES);
  dst = calloc(BATCH_SIZE, max_out);
  size_t comp_sizes[BATCH_SIZE];
  CHECK(Fail, src && dst);

  for (int i = 0; i < BATCH_SIZE; ++i)
    fill_pattern((char*)src + i * CHUNK_BYTES, CHUNK_BYTES, (uint8_t)i);

  struct codec_config codec = { .id = id, .level = 5, .shuffle = 1 };
  CHECK(Fail,
        compress_cpu(codec,
                     src,
                     CHUNK_BYTES,
                     dst,
                     max_out,
                     comp_sizes,
                     CHUNK_BYTES,
                     BATCH_SIZE,
                     1) == 0);

  // Decompress and verify round-trip
  recovered = malloc(CHUNK_BYTES);
  CHECK(Fail, recovered);

  for (int i = 0; i < BATCH_SIZE; ++i) {
    CHECK(Fail, comp_sizes[i] > 0 && comp_sizes[i] <= max_out);
    int rc = blosc_decompress_ctx(
      (const char*)dst + i * max_out, recovered, CHUNK_BYTES, 1);
    CHECK(Fail, rc == (int)CHUNK_BYTES);
    CHECK(Fail,
          memcmp((char*)src + i * CHUNK_BYTES, recovered, CHUNK_BYTES) == 0);
  }

  free(src);
  free(dst);
  free(recovered);
  log_info("  PASS");
  return 0;
Fail:
  free(src);
  free(dst);
  free(recovered);
  log_error("  FAIL");
  return 1;
}
#endif // HAVE_BLOSC

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_codec_none();
  rc |= test_codec_lz4();
  rc |= test_codec_zstd();
#ifdef HAVE_BLOSC
  rc |= test_codec_blosc(CODEC_BLOSC_LZ4, "blosc_lz4");
  rc |= test_codec_blosc(CODEC_BLOSC_ZSTD, "blosc_zstd");
#endif
  return rc;
}
