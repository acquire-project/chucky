#include "cpu/compress.h"
#include "cpu/compress_blosc.h"
#include "threadpool/threadpool.h"
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

static struct threadpool* g_pool;
static struct threadpool* g_pool1; // single-thread (0 workers + caller)

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
                     1,
                     g_pool) == 0);

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

  size_t max_out =
    compress_cpu_max_output_size(CODEC_LZ4_NON_STANDARD, CHUNK_BYTES);
  CHECK(Fail, max_out > 0);

  src = malloc(BATCH_SIZE * CHUNK_BYTES);
  dst = calloc(BATCH_SIZE, max_out);
  size_t comp_sizes[BATCH_SIZE];
  CHECK(Fail, src && dst);

  for (int i = 0; i < BATCH_SIZE; ++i)
    fill_pattern((char*)src + i * CHUNK_BYTES, CHUNK_BYTES, (uint8_t)i);

  CHECK(Fail,
        compress_cpu(
          (struct codec_config){ .id = CODEC_LZ4_NON_STANDARD, .level = 1 },
          src,
          CHUNK_BYTES,
          dst,
          max_out,
          comp_sizes,
          CHUNK_BYTES,
          BATCH_SIZE,
          1,
          g_pool) == 0);

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
                     1,
                     g_pool) == 0);

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

static int
test_nthreads_1(void)
{
  log_info("=== test_compress_cpu_nthreads_1 ===");

  enum
  {
    MT_BATCH = 2048
  };
  void* src = NULL;
  void* dst = NULL;
  void* recovered = NULL;
  size_t* comp_sizes = NULL;

  size_t max_out = compress_cpu_max_output_size(CODEC_ZSTD, CHUNK_BYTES);
  CHECK(Fail, max_out > 0);

  src = malloc((size_t)MT_BATCH * CHUNK_BYTES);
  dst = calloc(MT_BATCH, max_out);
  comp_sizes = (size_t*)calloc(MT_BATCH, sizeof(size_t));
  CHECK(Fail, src && dst && comp_sizes);

  for (int i = 0; i < MT_BATCH; ++i)
    fill_pattern((char*)src + (size_t)i * CHUNK_BYTES, CHUNK_BYTES, (uint8_t)i);

  CHECK(Fail,
        compress_cpu((struct codec_config){ .id = CODEC_ZSTD },
                     src,
                     CHUNK_BYTES,
                     dst,
                     max_out,
                     comp_sizes,
                     CHUNK_BYTES,
                     MT_BATCH,
                     1,
                     g_pool1) == 0);

  // Decompress and verify round-trip
  recovered = malloc(CHUNK_BYTES);
  CHECK(Fail, recovered);

  for (int i = 0; i < MT_BATCH; ++i) {
    CHECK(Fail, comp_sizes[i] > 0 && comp_sizes[i] <= max_out);
    size_t rc = ZSTD_decompress(recovered,
                                CHUNK_BYTES,
                                (const char*)dst + (size_t)i * max_out,
                                comp_sizes[i]);
    CHECK(Fail, !ZSTD_isError(rc) && rc == CHUNK_BYTES);
    CHECK(Fail,
          memcmp(
            (char*)src + (size_t)i * CHUNK_BYTES, recovered, CHUNK_BYTES) == 0);
  }

  free(src);
  free(dst);
  free(comp_sizes);
  free(recovered);
  log_info("  PASS");
  return 0;
Fail:
  free(src);
  free(dst);
  free(comp_sizes);
  free(recovered);
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

  struct codec_config codec = {
    .id = id, .level = 5, .shuffle = 1, .blosc_block_bytes = 16 * 1024
  };
  CHECK(Fail,
        compress_cpu(codec,
                     src,
                     CHUNK_BYTES,
                     dst,
                     max_out,
                     comp_sizes,
                     CHUNK_BYTES,
                     BATCH_SIZE,
                     1,
                     g_pool) == 0);

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
static int
test_blosc_explicit_blocks(void)
{
  enum
  {
    BYTES = 256 * 1024
  };
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  const uint32_t block_sizes[] = { 128, 4096, 4097, 16384, 65536, 524288 };
  uint8_t* src = malloc(BYTES);
  uint8_t* dst = malloc(BYTES + BLOSC_MAX_OVERHEAD);
  uint8_t* reference = malloc(BYTES + BLOSC_MAX_OVERHEAD);
  uint8_t* decoded = malloc(BYTES);
  CHECK(Fail, src && dst && reference && decoded);
  fill_pattern(src, BYTES, 17);
  for (size_t c = 0; c < countof(codecs); ++c) {
    struct codec_config config = {
      .id = codecs[c],
      .level = 5,
      .shuffle = CODEC_SHUFFLE_BIT,
    };
    size_t size = 0;
    CHECK(Fail, compress_blosc_validate(config) != 0);
    CHECK(Fail,
          compress_cpu(config,
                       src,
                       BYTES,
                       dst,
                       BYTES + BLOSC_MAX_OVERHEAD,
                       &size,
                       BYTES,
                       1,
                       2,
                       g_pool) != 0);
    config.level = 0;
    CHECK(Fail,
          compress_cpu(config,
                       src,
                       BYTES,
                       dst,
                       BYTES + BLOSC_MAX_OVERHEAD,
                       &size,
                       BYTES,
                       1,
                       2,
                       g_pool) != 0);
    config.blosc_block_bytes = 127;
    CHECK(Fail, compress_blosc_validate(config) != 0);
    config.blosc_block_bytes = UINT32_MAX;
    CHECK(Fail, compress_blosc_validate(config) != 0);
    for (size_t b = 0; b < countof(block_sizes); ++b)
      for (int level = 0; level <= 5; level += 5) {
        config.level = (uint8_t)level;
        config.blosc_block_bytes = block_sizes[b];
        CHECK(Fail,
              compress_cpu(config,
                           src,
                           BYTES,
                           dst,
                           BYTES + BLOSC_MAX_OVERHEAD,
                           &size,
                           BYTES,
                           1,
                           2,
                           g_pool) == 0);
        // Check compatibility with C-Blosc for the same settings.
        const int expected = blosc_compress_ctx(config.level,
                                                config.shuffle,
                                                2,
                                                BYTES,
                                                src,
                                                reference,
                                                BYTES + BLOSC_MAX_OVERHEAD,
                                                c == 0 ? "lz4" : "zstd",
                                                block_sizes[b],
                                                1);
        CHECK(Fail, expected > 0 && size == (size_t)expected);
        CHECK(Fail, memcmp(dst, reference, size) == 0);
        CHECK(Fail, blosc_decompress_ctx(dst, decoded, BYTES, 1) == BYTES);
        CHECK(Fail, memcmp(src, decoded, BYTES) == 0);
        if (config.id == CODEC_BLOSC_ZSTD && block_sizes[b] % 2 == 0) {
          size_t nbytes, cbytes, actual_block;
          blosc_cbuffer_sizes(dst, &nbytes, &cbytes, &actual_block);
          CHECK(Fail,
                actual_block ==
                  (block_sizes[b] < BYTES ? block_sizes[b] : BYTES));
        }
      }
  }
  free(src);
  free(dst);
  free(reference);
  free(decoded);
  return 0;
Fail:
  free(src);
  free(dst);
  free(reference);
  free(decoded);
  return 1;
}
#endif // HAVE_BLOSC

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  g_pool = threadpool_new(3);  // 4-way parallelism
  g_pool1 = threadpool_new(0); // caller-only
  if (!g_pool || !g_pool1) {
    log_error("threadpool_new failed");
    return 1;
  }

  int rc = 0;
  rc |= test_codec_none();
  rc |= test_codec_lz4();
  rc |= test_codec_zstd();
  rc |= test_nthreads_1();
#ifdef HAVE_BLOSC
  rc |= test_codec_blosc(CODEC_BLOSC_LZ4, "blosc_lz4");
  rc |= test_codec_blosc(CODEC_BLOSC_ZSTD, "blosc_zstd");
  rc |= test_blosc_explicit_blocks();
#endif

  threadpool_free(g_pool);
  threadpool_free(g_pool1);
  return rc;
}
