#include "gpu/blosc.frame.h"
#include "gpu/blosc.shuffle.h"
#include "gpu/compress.h"
#include "gpu/prelude.cuda.h"
#include "stream.gpu.h"
#include "util/prelude.h"

#include <blosc.h>
#include <limits.h>
#include <nvcomp/lz4.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_runner.h"

enum
{
  TEST_BLOCK_BYTES = 16 * 1024
};

enum pattern
{
  PATTERN_ZERO,
  PATTERN_RAMP,
  PATTERN_RANDOM,
  PATTERN_MIXED,
};

static void
byte_shuffle_reference(const uint8_t* src,
                       uint8_t* dst,
                       size_t chunk_bytes,
                       size_t typesize);
static void
bitshuffle_reference(const uint8_t* src,
                     uint8_t* dst,
                     size_t chunk_bytes,
                     size_t typesize);

static uint32_t
get_u32le(const uint8_t* p)
{
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

static void
fill_input(uint8_t* p, size_t bytes, enum pattern pattern, size_t block_bytes)
{
  if (pattern == PATTERN_MIXED) {
    fill_input(p, bytes, PATTERN_RAMP, block_bytes);
    // A random middle block must remain a raw block record inside an
    // otherwise compressed frame, including when a shuffle is enabled.
    if (bytes >= 2 * block_bytes)
      fill_input(p + block_bytes, block_bytes, PATTERN_RANDOM, block_bytes);
    return;
  }
  if (pattern == PATTERN_ZERO) {
    memset(p, 0, bytes);
    return;
  }
  if (pattern == PATTERN_RAMP) {
    for (size_t i = 0; i < bytes; ++i)
      p[i] = (uint8_t)(i % 251);
    return;
  }

  uint32_t state = 0x9e3779b9u;
  for (size_t i = 0; i < bytes; ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    p[i] = (uint8_t)state;
  }
}

static int
verify_chunk(const struct codec* c,
             const uint8_t* encoded,
             size_t encoded_bytes,
             const uint8_t* expected,
             int expect_memcpy,
             int expect_mixed)
{
  uint8_t* recovered = NULL;
  uint8_t* filtered = NULL;
  size_t validated_nbytes = 0;
  size_t meta_typesize = 0;
  int flags = 0;
  int result = 1;
  const size_t block_bytes = c->chunk_bytes < c->config.blosc_block_bytes
                               ? c->chunk_bytes
                               : c->config.blosc_block_bytes;
  const size_t nblocks = (c->chunk_bytes + block_bytes - 1) / block_bytes;

  CHECK(Fail, encoded_bytes > 0 && encoded_bytes <= c->max_output_size);
  CHECK(Fail, encoded[0] == 2 && encoded[1] == 1);
  CHECK(Fail, encoded[3] == c->typesize);
  CHECK(Fail, get_u32le(encoded + 4) == c->chunk_bytes);
  CHECK(Fail, get_u32le(encoded + 8) == block_bytes);
  CHECK(Fail, get_u32le(encoded + 12) == encoded_bytes);
  CHECK(Fail, (encoded[2] & 0x10) != 0);
  CHECK(Fail,
        ((encoded[2] >> 5) & 7) == (c->config.id == CODEC_BLOSC_LZ4 ? 1 : 4));
  CHECK(Fail,
        ((encoded[2] & BLOSC_DOSHUFFLE) != 0) ==
          (c->config.shuffle == CODEC_SHUFFLE_BYTE));
  CHECK(Fail,
        ((encoded[2] & BLOSC_DOBITSHUFFLE) != 0) ==
          (c->config.shuffle == CODEC_SHUFFLE_BIT));
  CHECK(Fail, ((encoded[2] & BLOSC_MEMCPYED) != 0) == expect_memcpy);
  if (expect_memcpy) {
    CHECK(Fail, encoded_bytes == c->chunk_bytes + BLOSC_MAX_OVERHEAD);
    CHECK(Fail,
          memcmp(encoded + BLOSC_MAX_OVERHEAD, expected, c->chunk_bytes) == 0);
  } else {
    size_t offset = BLOSC_MAX_OVERHEAD + nblocks * sizeof(uint32_t);
    size_t raw_blocks = 0;
    size_t compressed_blocks = 0;
    filtered = (uint8_t*)malloc(block_bytes);
    CHECK(Fail, filtered);
    for (size_t block = 0; block < nblocks; ++block) {
      const size_t remaining = c->chunk_bytes - block * block_bytes;
      const size_t bytes = remaining < block_bytes ? remaining : block_bytes;
      CHECK(Fail, get_u32le(encoded + 16 + 4 * block) == offset);
      CHECK(Fail, offset + sizeof(uint32_t) <= encoded_bytes);
      const size_t csize = get_u32le(encoded + offset);
      CHECK(Fail, csize > 0 && csize <= bytes);
      CHECK(Fail, offset + sizeof(uint32_t) + csize <= encoded_bytes);
      if (csize == bytes) {
        ++raw_blocks;
        const uint8_t* src = expected + block * block_bytes;
        if (c->config.shuffle == CODEC_SHUFFLE_BYTE)
          byte_shuffle_reference(src, filtered, bytes, c->typesize);
        else if (c->config.shuffle == CODEC_SHUFFLE_BIT)
          bitshuffle_reference(src, filtered, bytes, c->typesize);
        else
          memcpy(filtered, src, bytes);
        CHECK(Fail, memcmp(encoded + offset + 4, filtered, bytes) == 0);
      } else {
        ++compressed_blocks;
      }
      offset += sizeof(uint32_t) + csize;
    }
    CHECK(Fail, offset == encoded_bytes);
    if (expect_mixed)
      CHECK(Fail, raw_blocks > 0 && compressed_blocks > 0);
  }

  CHECK(Fail,
        blosc_cbuffer_validate(encoded, encoded_bytes, &validated_nbytes) == 0);
  CHECK(Fail, validated_nbytes == c->chunk_bytes);
  blosc_cbuffer_metainfo(encoded, &meta_typesize, &flags);
  CHECK(Fail, meta_typesize == c->typesize);
  CHECK(Fail, flags == (encoded[2] & 7));
  CHECK(Fail,
        strcmp(blosc_cbuffer_complib(encoded),
               c->config.id == CODEC_BLOSC_LZ4 ? "LZ4" : "Zstd") == 0);

  recovered = (uint8_t*)malloc(c->chunk_bytes);
  CHECK(Fail, recovered);
  CHECK(Fail,
        blosc_decompress_ctx(encoded, recovered, c->chunk_bytes, 1) ==
          (int)c->chunk_bytes);
  CHECK(Fail, memcmp(recovered, expected, c->chunk_bytes) == 0);
  result = 0;

Fail:
  free(filtered);
  free(recovered);
  return result;
}

static int
run_case(struct codec_config config,
         size_t typesize,
         size_t chunk_bytes,
         enum pattern pattern,
         int expect_memcpy,
         size_t actual_batch)
{
  const size_t batch = 3;
  const size_t input_stride = align_up(chunk_bytes, codec_alignment(config.id));
  struct codec c = { 0 };
  uint8_t* input = NULL;
  uint8_t* encoded = NULL;
  size_t* sizes = NULL;
  CUdeviceptr d_input = 0;
  CUdeviceptr d_encoded = 0;
  CUstream stream = 0;
  int result = 1;

  CHECK(Fail,
        codec_init_config(&c, config, typesize, chunk_bytes, batch, 0) == 0);
  CHECK(Fail, c.max_output_size == chunk_bytes + BLOSC_MAX_OVERHEAD);
  CHECK(Fail,
        c.output_stride ==
          align_up(c.max_output_size, codec_output_alignment(config.id)));
  CHECK(Fail,
        c.block_bytes == (chunk_bytes < config.blosc_block_bytes
                            ? chunk_bytes
                            : config.blosc_block_bytes));
  CHECK(Fail,
        c.blocks_per_chunk == (chunk_bytes + config.blosc_block_bytes - 1) /
                                config.blosc_block_bytes);

  input = (uint8_t*)malloc(batch * input_stride);
  encoded = (uint8_t*)malloc(batch * c.output_stride);
  sizes = (size_t*)malloc(batch * sizeof(size_t));
  CHECK(Fail, input && encoded && sizes);
  for (size_t i = 0; i < batch; ++i)
    fill_input(input + i * input_stride, chunk_bytes, pattern, c.block_bytes);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_input, batch * input_stride));
  CU(Fail, cuMemAlloc(&d_encoded, batch * c.output_stride));
  CU(Fail, cuMemcpyHtoD(d_input, input, batch * input_stride));
  // Pageable H2D copies can return before the default-stream DMA finishes.
  // The nonblocking test stream does not inherit that dependency.
  CU(Fail, cuStreamSynchronize(NULL));
  CU(Fail, cuMemsetD8Async(d_encoded, 0xa5, batch * c.output_stride, stream));
  CHECK(Fail,
        codec_compress(&c,
                       (const void*)(uintptr_t)d_input,
                       input_stride,
                       (void*)(uintptr_t)d_encoded,
                       actual_batch,
                       stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(encoded, d_encoded, batch * c.output_stride));
  CU(Fail,
     cuMemcpyDtoH(
       sizes, (CUdeviceptr)c.d_comp_sizes, actual_batch * sizeof(size_t)));

  for (size_t i = 0; i < actual_batch; ++i)
    CHECK(Fail,
          verify_chunk(&c,
                       encoded + i * c.output_stride,
                       sizes[i],
                       input + i * input_stride,
                       expect_memcpy,
                       pattern == PATTERN_MIXED) == 0);
  for (size_t i = actual_batch * c.output_stride; i < batch * c.output_stride;
       ++i)
    CHECK(Fail, encoded[i] == 0xa5);
  result = 0;

Fail:
  cu_mem_free(d_input);
  cu_mem_free(d_encoded);
  cu_stream_destroy(stream);
  codec_free(&c);
  free(input);
  free(encoded);
  free(sizes);
  return result;
}

static int
test_compressed_matrix(void)
{
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  const enum codec_shuffle shuffles[] = {
    CODEC_SHUFFLE_NONE,
    CODEC_SHUFFLE_BYTE,
    CODEC_SHUFFLE_BIT,
  };
  const size_t typesizes[] = { 1, 2, 4, 8 };
  for (size_t c = 0; c < countof(codecs); ++c)
    for (size_t s = 0; s < countof(shuffles); ++s)
      for (size_t t = 0; t < countof(typesizes); ++t) {
        struct codec_config config = { .id = codecs[c],
                                       .level = 5,
                                       .shuffle = shuffles[s],
                                       .blosc_block_bytes = 16 * 1024 };
        CHECK(Fail,
              run_case(config, typesizes[t], 64 * 1024, PATTERN_RAMP, 0, 2) ==
                0);
      }
  {
    // The full blocks are bitshuffled, while the last four-byte block has an
    // element count not divisible by eight and must remain unshuffled.
    struct codec_config nonmultiple = { .id = CODEC_BLOSC_LZ4,
                                        .level = 5,
                                        .shuffle = CODEC_SHUFFLE_BIT,
                                        .blosc_block_bytes = 16 * 1024 };
    CHECK(Fail, run_case(nonmultiple, 4, 65540, PATTERN_RAMP, 0, 1) == 0);
  }
  {
    struct codec_config transformed = { .id = CODEC_BLOSC_ZSTD,
                                        .level = 5,
                                        .shuffle = CODEC_SHUFFLE_BIT,
                                        .blosc_block_bytes = 16 * 1024 };
    CHECK(Fail, run_case(transformed, 8, 4096, PATTERN_RAMP, 0, 2) == 0);
  }
  return 0;
Fail:
  return 1;
}

static int
test_memcpy_fallbacks(void)
{
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  for (size_t i = 0; i < countof(codecs); ++i) {
    struct codec_config level_zero_byte = { .id = codecs[i],
                                            .level = 0,
                                            .shuffle = CODEC_SHUFFLE_BYTE,
                                            .blosc_block_bytes = 16 * 1024 };
    struct codec_config level_zero_bit = { .id = codecs[i],
                                           .level = 0,
                                           .shuffle = CODEC_SHUFFLE_BIT,
                                           .blosc_block_bytes = 16 * 1024 };
    struct codec_config enabled_bit = { .id = codecs[i],
                                        .level = 5,
                                        .shuffle = CODEC_SHUFFLE_BIT,
                                        .blosc_block_bytes = 16 * 1024 };
    struct codec_config incompressible = enabled_bit;
    incompressible.shuffle = CODEC_SHUFFLE_NONE;
    CHECK(Fail, run_case(level_zero_byte, 8, 4096, PATTERN_ZERO, 1, 3) == 0);
    CHECK(Fail, run_case(level_zero_bit, 8, 4096, PATTERN_ZERO, 1, 3) == 0);
    CHECK(Fail,
          run_case(
            level_zero_bit, 8, 2 * TEST_BLOCK_BYTES + 39, PATTERN_RAMP, 1, 2) ==
            0);
    CHECK(Fail, run_case(enabled_bit, 4, 127, PATTERN_ZERO, 1, 3) == 0);
    CHECK(Fail,
          run_case(incompressible, 4, 64 * 1024, PATTERN_RANDOM, 1, 3) == 0);
    CHECK(Fail,
          run_case(
            enabled_bit, 8, 2 * TEST_BLOCK_BYTES + 39, PATTERN_RANDOM, 1, 2) ==
            0);
  }
  return 0;
Fail:
  return 1;
}

static int
test_multiblock_boundaries(void)
{
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  const enum codec_shuffle shuffles[] = {
    CODEC_SHUFFLE_NONE,
    CODEC_SHUFFLE_BYTE,
    CODEC_SHUFFLE_BIT,
  };
  // Exercise both the byte tail and the bitshuffle element-count boundary
  // independently in the final block. The last case crosses a 256-block scan.
  const size_t chunk_sizes[] = {
    TEST_BLOCK_BYTES - 1,       TEST_BLOCK_BYTES,
    TEST_BLOCK_BYTES + 1,       2 * TEST_BLOCK_BYTES + 64,
    2 * TEST_BLOCK_BYTES + 67,  2 * TEST_BLOCK_BYTES + 75,
    257 * TEST_BLOCK_BYTES + 7,
  };
  for (size_t codec = 0; codec < countof(codecs); ++codec)
    for (size_t shuffle = 0; shuffle < countof(shuffles); ++shuffle) {
      const struct codec_config config = { .id = codecs[codec],
                                           .level = 5,
                                           .shuffle = shuffles[shuffle],
                                           .blosc_block_bytes = 16 * 1024 };
      for (size_t size = 0; size < countof(chunk_sizes); ++size)
        CHECK(Fail,
              run_case(config, 8, chunk_sizes[size], PATTERN_RAMP, 0, 2) == 0);
      CHECK(Fail,
            run_case(
              config, 8, 3 * TEST_BLOCK_BYTES + 67, PATTERN_MIXED, 0, 2) == 0);
      // The public codec accepts typesizes that do not divide 16 KiB. Their
      // complete-element/tail rules must be applied within every block.
      const size_t odd_typesizes[] = { 3, 255 };
      for (size_t type = 0; type < countof(odd_typesizes); ++type)
        CHECK(Fail,
              run_case(config,
                       odd_typesizes[type],
                       3 * TEST_BLOCK_BYTES + 67,
                       PATTERN_MIXED,
                       0,
                       1) == 0);
    }
  {
    const struct codec_config lz4 = { .id = CODEC_BLOSC_LZ4,
                                      .level = 5,
                                      .shuffle = CODEC_SHUFFLE_BIT,
                                      .blosc_block_bytes = 16 * 1024 };
    CHECK(Fail,
          run_case(lz4,
                   8,
                   nvcompLZ4CompressionMaxAllowedChunkSize + 67,
                   PATTERN_RAMP,
                   0,
                   1) == 0);
  }
  return 0;

Fail:
  return 1;
}

static int
run_bound_case(struct codec* c,
               struct codec_config config,
               size_t typesize,
               size_t chunk_bytes,
               size_t actual_batch,
               CUstream stream)
{
  const size_t input_stride = align_up(chunk_bytes, codec_alignment(config.id));
  const enum pattern pattern =
    chunk_bytes >= 3 * TEST_BLOCK_BYTES ? PATTERN_MIXED : PATTERN_RAMP;
  uint8_t* input = NULL;
  uint8_t* encoded = NULL;
  size_t* sizes = NULL;
  CUdeviceptr d_input = 0;
  CUdeviceptr d_encoded = 0;
  int result = 1;

  CHECK(Fail,
        c->config.id == config.id && c->config.level == config.level &&
          c->config.shuffle == config.shuffle &&
          c->config.blosc_block_bytes == config.blosc_block_bytes);
  CHECK(Fail, c->chunk_bytes == chunk_bytes && c->typesize == typesize);
  CHECK(Fail,
        c->blocks_per_chunk == (chunk_bytes + config.blosc_block_bytes - 1) /
                                 config.blosc_block_bytes);
  input = (uint8_t*)calloc(c->batch_size, input_stride);
  encoded = (uint8_t*)malloc(c->batch_size * c->output_stride);
  sizes = (size_t*)malloc(actual_batch * sizeof(size_t));
  CHECK(Fail, input && encoded && sizes);
  for (size_t chunk = 0; chunk < actual_batch; ++chunk)
    fill_input(
      input + chunk * input_stride, chunk_bytes, pattern, c->block_bytes);
  CU(Fail, cuMemAlloc(&d_input, c->batch_size * input_stride));
  CU(Fail, cuMemAlloc(&d_encoded, c->batch_size * c->output_stride));
  CU(Fail, cuMemcpyHtoD(d_input, input, c->batch_size * input_stride));
  CU(Fail, cuStreamSynchronize(NULL));
  CU(
    Fail,
    cuMemsetD8Async(d_encoded, 0xa5, c->batch_size * c->output_stride, stream));
  CHECK(Fail,
        codec_compress(c,
                       (const void*)(uintptr_t)d_input,
                       input_stride,
                       (void*)(uintptr_t)d_encoded,
                       actual_batch,
                       stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(encoded, d_encoded, c->batch_size * c->output_stride));
  CU(Fail,
     cuMemcpyDtoH(
       sizes, (CUdeviceptr)c->d_comp_sizes, actual_batch * sizeof(size_t)));
  for (size_t chunk = 0; chunk < actual_batch; ++chunk)
    CHECK(Fail,
          verify_chunk(c,
                       encoded + chunk * c->output_stride,
                       sizes[chunk],
                       input + chunk * input_stride,
                       0,
                       pattern == PATTERN_MIXED) == 0);
  for (size_t i = actual_batch * c->output_stride;
       i < c->batch_size * c->output_stride;
       ++i)
    CHECK(Fail, encoded[i] == 0xa5);
  result = 0;

Fail:
  cu_mem_free(d_input);
  cu_mem_free(d_encoded);
  free(input);
  free(encoded);
  free(sizes);
  return result;
}

static int
run_rebound_case(struct codec* c,
                 struct codec_config config,
                 size_t typesize,
                 size_t chunk_bytes,
                 size_t actual_batch,
                 CUstream stream)
{
  if (codec_bind(c, config, typesize, chunk_bytes, stream))
    return 1;
  return run_bound_case(c, config, typesize, chunk_bytes, actual_batch, stream);
}

// Rejected bindings must leave a usable previous binding. Exercise this
// without rebinding it, including the no-op bind of an initial filter.
static int
test_preparation_reservation(void)
{
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  struct codec c = { 0 };
  CUstream stream = 0;
  int result = 1;
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  for (size_t k = 0; k < countof(codecs); ++k) {
    for (int reserve = 0; reserve <= 1; ++reserve) {
      struct codec_config initial = { .id = codecs[k],
                                      .level = 3,
                                      .shuffle = CODEC_SHUFFLE_NONE,
                                      .blosc_block_bytes = TEST_BLOCK_BYTES };
      CHECK(Fail, codec_init_config(&c, initial, 2, 65536, 3, reserve) == 0);
      for (int shuffle = CODEC_SHUFFLE_BYTE; shuffle <= CODEC_SHUFFLE_BIT;
           ++shuffle) {
        struct codec_config next = initial;
        next.shuffle = (enum codec_shuffle)shuffle;
        next.level = 7;
        if (reserve) {
          CHECK(Fail, run_rebound_case(&c, next, 3, 32775, 2, stream) == 0);
          CHECK(Fail, run_rebound_case(&c, initial, 2, 65536, 3, stream) == 0);
        } else {
          CHECK(Fail, codec_bind(&c, next, 3, 32775, stream) != 0);
          CHECK(Fail, run_bound_case(&c, initial, 2, 65536, 3, stream) == 0);
        }
      }
      codec_free(&c);
    }
    for (int shuffle = CODEC_SHUFFLE_BYTE; shuffle <= CODEC_SHUFFLE_BIT;
         ++shuffle) {
      struct codec_config initial = { .id = codecs[k],
                                      .level = 3,
                                      .shuffle = (enum codec_shuffle)shuffle,
                                      .blosc_block_bytes = TEST_BLOCK_BYTES };
      // Even byte shuffle of one-byte elements reserves storage from config;
      // a later element-width change can use it without a new reservation.
      CHECK(Fail, codec_init_config(&c, initial, 1, 65536, 3, 0) == 0);
      CHECK(Fail, run_rebound_case(&c, initial, 1, 65536, 3, stream) == 0);
      CHECK(Fail, run_rebound_case(&c, initial, 8, 32775, 2, stream) == 0);
      codec_free(&c);
    }
  }
  result = 0;
Fail:
  codec_free(&c);
  cu_stream_destroy(stream);
  return result;
}

static int
test_explicit_block_sizes(void)
{
  const uint32_t block_sizes[] = { 128,   129,   256,   4096,   4097,
                                   16384, 32768, 65536, 1048576 };
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  for (size_t c = 0; c < countof(codecs); ++c)
    for (int shuffle = CODEC_SHUFFLE_NONE; shuffle <= CODEC_SHUFFLE_BIT;
         ++shuffle)
      for (size_t b = 0; b < countof(block_sizes); ++b) {
        struct codec_config config = {
          .id = codecs[c],
          .level = 5,
          .shuffle = (enum codec_shuffle)shuffle,
          .blosc_block_bytes = block_sizes[b],
        };
        const size_t chunks[] = { block_sizes[b],
                                  block_sizes[b] + 1,
                                  3 * block_sizes[b] + 67 };
        for (size_t n = 0; n < countof(chunks); ++n)
          CHECK(Fail, run_case(config, 8, chunks[n], PATTERN_ZERO, 0, 2) == 0);
        CHECK(Fail, run_case(config, 8, 127, PATTERN_ZERO, 1, 1) == 0);
        if (block_sizes[b] >= 4096)
          CHECK(Fail, run_case(config, 8, chunks[2], PATTERN_MIXED, 0, 2) == 0);
        config.level = 0;
        CHECK(Fail, run_case(config, 8, chunks[2], PATTERN_RAMP, 1, 1) == 0);
      }
  return 0;
Fail:
  return 1;
}

static int
test_rebind_and_rejection(void)
{
  struct codec c = { 0 };
  CUstream stream = 0;
  struct codec_config initial = { .id = CODEC_BLOSC_ZSTD,
                                  .level = 2,
                                  .shuffle = CODEC_SHUFFLE_NONE,
                                  .blosc_block_bytes = 16 * 1024 };
  struct codec_config rebound = { .id = CODEC_BLOSC_ZSTD,
                                  .level = 8,
                                  .shuffle = CODEC_SHUFFLE_BIT,
                                  .blosc_block_bytes = 16 * 1024 };
  struct codec_config invalid = { .id = CODEC_BLOSC_ZSTD,
                                  .level = 5,
                                  .shuffle = (enum codec_shuffle)99,
                                  .blosc_block_bytes = 16 * 1024 };
  int result = 1;

  CHECK(Fail, codec_validate_gpu(invalid, 4, 4096) != 0);
  {
    struct codec_config lz4 = initial;
    lz4.id = CODEC_BLOSC_LZ4;
    CHECK(Fail,
          codec_validate_gpu(lz4, 4, nvcompLZ4CompressionMaxAllowedChunkSize) ==
            0);
    CHECK(Fail,
          codec_validate_gpu(
            lz4, 4, nvcompLZ4CompressionMaxAllowedChunkSize + 1) == 0);
    CHECK(Fail,
          codec_validate_gpu(lz4, 4, (size_t)INT_MAX - BLOSC_MAX_OVERHEAD) ==
            0);
    CHECK(Fail,
          codec_validate_gpu(
            lz4, 4, (size_t)INT_MAX - BLOSC_MAX_OVERHEAD + 1) != 0);
    CHECK(Fail, codec_validate_gpu(lz4, 4, 0) != 0);
    lz4.blosc_block_bytes = nvcompLZ4CompressionMaxAllowedChunkSize + 4;
    CHECK(Fail, codec_validate_gpu(lz4, 4, lz4.blosc_block_bytes) != 0);
    lz4.blosc_block_bytes = 4097;
    CHECK(Fail, codec_validate_gpu(lz4, 4, 8192) == 0);
    CHECK(Fail, codec_validate_gpu(lz4, 4, 4096) == 0);
    lz4.blosc_block_bytes = 0;
    CHECK(Fail, codec_validate_gpu(lz4, 4, 4096) != 0);
    CHECK(Fail, codec_device_bytes(lz4, 4096, 2, 1) == 0);
    CHECK(Fail, codec_init_config(&c, lz4, 4, 4096, 2, 1) != 0);
    lz4.level = 0;
    CHECK(Fail, codec_validate_gpu(lz4, 4, 4096) != 0);
    lz4.blosc_block_bytes = 127;
    CHECK(Fail, codec_validate_gpu(lz4, 4, 4096) != 0);
    lz4.blosc_block_bytes = UINT32_MAX;
    CHECK(Fail, codec_validate_gpu(lz4, 4, 4096) != 0);
  }
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, codec_init_config(&c, initial, 2, 65536, 3, 1) == 0);
  CHECK(Fail, run_rebound_case(&c, initial, 2, 65536, 3, stream) == 0);
  CHECK(Fail, run_rebound_case(&c, rebound, 8, 4096, 1, stream) == 0);
  CHECK(Fail, c.config.level == 8 && c.config.shuffle == CODEC_SHUFFLE_BIT);
  CHECK(Fail, run_rebound_case(&c, rebound, 8, 32775, 2, stream) == 0);
  rebound.shuffle = CODEC_SHUFFLE_BYTE;
  CHECK(Fail, run_rebound_case(&c, rebound, 4, 65536, 3, stream) == 0);
  CHECK(Fail, run_rebound_case(&c, initial, 2, 16387, 1, stream) == 0);
  CHECK(Fail, codec_bind(&c, invalid, 8, 4096, stream) != 0);
  CHECK(Fail, run_bound_case(&c, initial, 2, 16387, 1, stream) == 0);
  // Block-size changes must fail without changing the active state.
  rebound.blosc_block_bytes = 4096;
  CHECK(Fail, codec_bind(&c, rebound, 8, 4096, stream) != 0);
  CHECK(Fail, run_bound_case(&c, initial, 2, 16387, 1, stream) == 0);
  rebound.blosc_block_bytes = 65536;
  CHECK(Fail, codec_bind(&c, rebound, 8, 4096, stream) != 0);
  CHECK(Fail, run_bound_case(&c, initial, 2, 16387, 1, stream) == 0);
  CHECK(Fail, run_rebound_case(&c, initial, 2, 65536, 3, stream) == 0);
  codec_free(&c);
  // Odd blocks reserve preparation storage for alignment alone. Reuse it for
  // either filter, shrink to a single block, then return to the full batch.
  initial.blosc_block_bytes = 4097;
  CHECK(Fail, codec_init_config(&c, initial, 2, 32775, 3, 0) == 0);
  CHECK(Fail, run_rebound_case(&c, initial, 2, 32775, 3, stream) == 0);
  rebound = initial;
  rebound.shuffle = CODEC_SHUFFLE_BYTE;
  CHECK(Fail, run_rebound_case(&c, rebound, 3, 32775, 2, stream) == 0);
  rebound.shuffle = CODEC_SHUFFLE_BIT;
  CHECK(Fail, run_rebound_case(&c, rebound, 8, 4096, 1, stream) == 0);
  CHECK(Fail, run_rebound_case(&c, initial, 2, 32775, 3, stream) == 0);
  result = 0;

Fail:
  codec_free(&c);
  cu_stream_destroy(stream);
  return result;
}

static int
test_codec_allocation_estimate(void)
{
  const enum compression_codec codecs[] = { CODEC_BLOSC_LZ4, CODEC_BLOSC_ZSTD };
  const enum codec_shuffle shuffles[] = {
    CODEC_SHUFFLE_NONE,
    CODEC_SHUFFLE_BYTE,
    CODEC_SHUFFLE_BIT,
  };
  const uint32_t block_sizes[] = {
    4096, 4097, 16384, 65536, 262144, 1048576, 2097152,
  };
  const uint8_t levels[] = { 0, 3 };
  const size_t chunk_bytes = 1048576;
  const size_t batch = 64;
  struct codec c = { 0 };
  for (size_t k = 0; k < countof(codecs); ++k) {
    for (size_t b = 0; b < countof(block_sizes); ++b) {
      for (size_t l = 0; l < countof(levels); ++l) {
        for (size_t s = 0; s < countof(shuffles); ++s) {
          for (int reserve = 0; reserve <= 1; ++reserve) {
            const struct codec_config config = {
              .id = codecs[k],
              .level = levels[l],
              .shuffle = shuffles[s],
              .blosc_block_bytes = block_sizes[b],
            };
            CHECK(Fail,
                  codec_init_config(
                    &c, config, 2, chunk_bytes, batch, reserve) == 0);
            const void* allocations[] = {
              c.d_comp_sizes,  c.d_uncomp_sizes, c.d_ptrs,
              c.d_block_input, c.d_block_sizes,  c.d_block_offsets,
              c.d_block_data,  c.d_temp,
            };
            size_t allocated = 0;
            for (size_t i = 0; i < countof(allocations); ++i) {
              if (!allocations[i])
                continue;
              CUdeviceptr base = 0;
              size_t bytes = 0;
              CU(Fail,
                 cuMemGetAddressRange(
                   &base, &bytes, (CUdeviceptr)(uintptr_t)allocations[i]));
              CHECK(Fail, base == (CUdeviceptr)(uintptr_t)allocations[i]);
              CHECK(Fail, allocated <= SIZE_MAX - bytes);
              allocated += bytes;
            }
            CHECK(Fail,
                  allocated ==
                    codec_device_bytes(config, chunk_bytes, batch, reserve));
            codec_free(&c);
          }
        }
      }
    }
  }
  return 0;

Fail:
  codec_free(&c);
  return 1;
}

static int
check_stream_memory_estimate(enum compression_codec codec)
{
  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 0, 128, 256 });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 128, 256 });
  dims[0].chunks_per_shard = 4;
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = codec,
               .level = 5,
               .shuffle = CODEC_SHUFFLE_NONE,
               .blosc_block_bytes = 16 * 1024 },
    .epochs_per_batch = 2,
  };

  const enum codec_shuffle shuffles[] = {
    CODEC_SHUFFLE_NONE,
    CODEC_SHUFFLE_BYTE,
    CODEC_SHUFFLE_BIT,
  };
  struct tile_stream_memory_info info0 = { 0 };
  config.codec.blosc_block_bytes = 0;
  CHECK(Fail, tile_stream_gpu_memory_estimate(&config, 0, &info0) != 0);
  const uint32_t block_sizes[] = { 4096, 4097, 16384, 65536, 131072 };
  size_t first_bytes = 0;
  for (size_t b = 0; b < countof(block_sizes); ++b) {
    config.codec.blosc_block_bytes = block_sizes[b];
    for (size_t s = 0; s < countof(shuffles); ++s) {
      config.codec.shuffle = shuffles[s];
      const int reserve = shuffles[s] != CODEC_SHUFFLE_NONE;
      struct tile_stream_memory_info info = { 0 };
      CHECK(Fail, tile_stream_gpu_memory_estimate(&config, 0, &info) == 0);
      CHECK(Fail, info.max_output_size > 16);
      const size_t chunk_bytes = info.max_output_size - 16;
      const size_t batch = info.epochs_per_batch * info.total_chunks;
      CHECK(Fail,
            info.codec_bytes ==
              codec_device_bytes(config.codec, chunk_bytes, batch, reserve));
      CHECK(Fail,
            info.compressed_pool_bytes ==
              2 * batch * codec_output_stride(config.codec.id, chunk_bytes));
      if (s == 0) {
        if (b == 0)
          first_bytes = info.codec_bytes;
        else if (b == countof(block_sizes) - 1)
          CHECK(Fail, info.codec_bytes != first_bytes);
      }
    }
  }
  return 0;

Fail:
  return 1;
}

static int
test_memory_estimate(void)
{
  return check_stream_memory_estimate(CODEC_BLOSC_LZ4) ||
         check_stream_memory_estimate(CODEC_BLOSC_ZSTD);
}

static void
byte_shuffle_reference(const uint8_t* src,
                       uint8_t* dst,
                       size_t chunk_bytes,
                       size_t typesize)
{
  const size_t nelem = chunk_bytes / typesize;
  const size_t complete = nelem * typesize;
  for (size_t byte = 0; byte < typesize; ++byte)
    for (size_t elem = 0; elem < nelem; ++elem)
      dst[byte * nelem + elem] = src[elem * typesize + byte];
  memcpy(dst + complete, src + complete, chunk_bytes - complete);
}

static void
bitshuffle_reference(const uint8_t* src,
                     uint8_t* dst,
                     size_t chunk_bytes,
                     size_t typesize)
{
  const size_t nelem = chunk_bytes / typesize;
  const size_t complete = nelem * typesize;
  if ((nelem & 7) != 0) {
    memcpy(dst, src, chunk_bytes);
    return;
  }

  const size_t groups = nelem / 8;
  for (size_t byte = 0; byte < typesize; ++byte)
    for (unsigned bit = 0; bit < 8; ++bit)
      for (size_t group = 0; group < groups; ++group) {
        uint8_t packed = 0;
        for (unsigned elem = 0; elem < 8; ++elem) {
          const uint8_t value = src[(group * 8 + elem) * typesize + byte];
          packed |= (uint8_t)(((value >> bit) & 1u) << elem);
        }
        dst[(byte * 8 + bit) * groups + group] = packed;
      }
  memcpy(dst + complete, src + complete, chunk_bytes - complete);
}

static int
run_prepare_case(enum codec_shuffle shuffle,
                 size_t typesize,
                 size_t chunk_bytes,
                 size_t block_bytes)
{
  const size_t batch = 2;
  const size_t input_stride = align_up(chunk_bytes, 64);
  const size_t block_stride = align_up(block_bytes, 64) + 64;
  const size_t blocks = (chunk_bytes + block_bytes - 1) / block_bytes;
  const size_t output_bytes = batch * blocks * block_stride;
  uint8_t* input = NULL;
  uint8_t* expected = NULL;
  uint8_t* actual = NULL;
  CUdeviceptr d_input = 0;
  CUdeviceptr d_output = 0;
  CUstream stream = 0;
  int result = 1;

  input = (uint8_t*)calloc(batch, input_stride);
  expected = (uint8_t*)malloc(output_bytes);
  actual = (uint8_t*)malloc(output_bytes);
  CHECK(Fail, input && expected && actual);
  memset(expected, 0xa5, output_bytes);
  for (size_t chunk = 0; chunk < batch; ++chunk) {
    for (size_t i = 0; i < chunk_bytes; ++i)
      input[chunk * input_stride + i] = (uint8_t)(i + 17 * chunk);
    for (size_t block = 0; block < blocks; ++block) {
      const size_t offset = block * block_bytes;
      const size_t bytes =
        chunk_bytes - offset < block_bytes ? chunk_bytes - offset : block_bytes;
      const uint8_t* src = input + chunk * input_stride + offset;
      uint8_t* dst = expected + (chunk * blocks + block) * block_stride;
      if (shuffle == CODEC_SHUFFLE_BIT)
        bitshuffle_reference(src, dst, bytes, typesize);
      else if (shuffle == CODEC_SHUFFLE_BYTE)
        byte_shuffle_reference(src, dst, bytes, typesize);
      else
        memcpy(dst, src, bytes);
    }
  }

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_input, batch * input_stride));
  CU(Fail, cuMemAlloc(&d_output, output_bytes));
  CU(Fail, cuMemcpyHtoD(d_input, input, batch * input_stride));
  CU(Fail, cuStreamSynchronize(NULL));
  CU(Fail, cuMemsetD8Async(d_output, 0xa5, output_bytes, stream));
  CHECK(
    Fail,
    gpu_blosc_prepare_blocks_async(
      (struct gpu_blosc_frame_layout){
        CODEC_BLOSC_LZ4, shuffle, typesize, chunk_bytes, block_bytes },
      (struct gpu_blosc_input){ (const void*)(uintptr_t)d_input, input_stride },
      (void*)(uintptr_t)d_output,
      block_stride,
      batch,
      stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(actual, d_output, output_bytes));
  // Include padding and the partial final slot: preparation must only write
  // each block's logical bytes, even when the retained stride is larger.
  CHECK(Fail, memcmp(actual, expected, output_bytes) == 0);
  result = 0;

Fail:
  cu_mem_free(d_input);
  cu_mem_free(d_output);
  cu_stream_destroy(stream);
  free(input);
  free(expected);
  free(actual);
  return result;
}

static int
test_shuffle_filters(void)
{
  // Pinned C-Blosc 1.21.6 scalar-format vector: sixteen 2-byte elements whose
  // input bytes are 0..31.
  const uint8_t golden[] = { 0x00, 0x00, 0xaa, 0xaa, 0xcc, 0xcc, 0xf0, 0xf0,
                             0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0xff, 0xff, 0xaa, 0xaa, 0xcc, 0xcc, 0xf0, 0xf0,
                             0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  uint8_t input[countof(golden)];
  uint8_t actual[countof(golden)];
  for (size_t i = 0; i < countof(input); ++i)
    input[i] = (uint8_t)i;
  bitshuffle_reference(input, actual, sizeof(input), 2);
  CHECK(Fail, memcmp(actual, golden, sizeof(golden)) == 0);

  const size_t typesizes[] = { 1, 2, 3, 4, 8 };
  const size_t block_sizes[] = { 128, 129, 4097 };
  for (size_t i = 0; i < countof(typesizes); ++i) {
    const size_t bytes = 16 * typesizes[i] + typesizes[i] - 1;
    CHECK(Fail,
          run_prepare_case(CODEC_SHUFFLE_BIT, typesizes[i], bytes, bytes) == 0);
    for (int shuffle = CODEC_SHUFFLE_NONE; shuffle <= CODEC_SHUFFLE_BIT;
         ++shuffle)
      for (size_t b = 0; b < countof(block_sizes); ++b)
        CHECK(Fail,
              run_prepare_case((enum codec_shuffle)shuffle,
                               typesizes[i],
                               3 * block_sizes[b] + 19,
                               block_sizes[b]) == 0);
  }
  CHECK(Fail, run_prepare_case(CODEC_SHUFFLE_BIT, 8, 65539, 65539) == 0);
  CHECK(Fail, run_prepare_case(CODEC_SHUFFLE_BIT, 4, 71, 71) == 0);
  CHECK(Fail, run_prepare_case(CODEC_SHUFFLE_BYTE, 8, 65539, 65539) == 0);
  return 0;

Fail:
  return 1;
}

static int
test_documented_frame(void)
{
  const uint8_t input[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const uint8_t expected[] = {
    0x02, 0x01, 0x32, 0x02, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x18, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
  };
  const struct codec_config config = {
    .id = CODEC_BLOSC_LZ4,
    .level = 0,
    .shuffle = CODEC_SHUFFLE_NONE,
    .blosc_block_bytes = 16384,
  };
  struct codec c = { 0 };
  CUdeviceptr d_input = 0, d_output = 0;
  CUstream stream = NULL;
  uint8_t encoded[sizeof(expected)];
  uint8_t recovered[sizeof(input)];
  size_t encoded_size = 0;
  int result = 1;
  CHECK(Fail, codec_init_config(&c, config, 2, sizeof(input), 1, 0) == 0);
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_input, sizeof(input)));
  CU(Fail, cuMemAlloc(&d_output, c.output_stride));
  CU(Fail, cuMemcpyHtoD(d_input, input, sizeof(input)));
  CU(Fail, cuStreamSynchronize(NULL));
  CHECK(Fail,
        codec_compress(&c,
                       (const void*)(uintptr_t)d_input,
                       sizeof(input),
                       (void*)(uintptr_t)d_output,
                       1,
                       stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(encoded, d_output, sizeof(encoded)));
  CU(Fail,
     cuMemcpyDtoH(&encoded_size,
                  (CUdeviceptr)(uintptr_t)c.d_comp_sizes,
                  sizeof(encoded_size)));
  CHECK(Fail, encoded_size == sizeof(expected));
  CHECK(Fail, memcmp(encoded, expected, sizeof(expected)) == 0);
  CHECK(Fail,
        blosc_decompress_ctx(encoded, recovered, sizeof(recovered), 1) ==
          (int)sizeof(recovered));
  CHECK(Fail, memcmp(input, recovered, sizeof(input)) == 0);
  result = 0;

Fail:
  codec_free(&c);
  cu_mem_free(d_input);
  cu_mem_free(d_output);
  cu_stream_destroy(stream);
  return result;
}

static int
test_frame_boundary(void)
{
  enum
  {
    CHUNK_BYTES = 256,
    BATCH = 3,
    STRIDE = 512,
  };
  uint8_t input[BATCH * CHUNK_BYTES];
  uint8_t block_data[BATCH * STRIDE];
  uint8_t encoded[BATCH * STRIDE];
  const size_t block_sizes[BATCH] = { CHUNK_BYTES - 9,
                                      CHUNK_BYTES - 8,
                                      CHUNK_BYTES - 7 };
  size_t actual_block_sizes[BATCH];
  size_t sizes[BATCH];
  CUdeviceptr d_input = 0;
  CUdeviceptr d_inputs = 0;
  const void* inputs[BATCH];
  CUdeviceptr d_block_data = 0;
  CUdeviceptr d_block_sizes = 0;
  CUdeviceptr d_block_offsets = 0;
  CUdeviceptr d_encoded = 0;
  CUdeviceptr d_sizes = 0;
  CUstream stream = 0;
  int result = 1;

  fill_input(input, sizeof(input), PATTERN_RAMP, CHUNK_BYTES);
  memset(block_data, 0xa5, sizeof(block_data));
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_input, sizeof(input)));
  CU(Fail, cuMemAlloc(&d_inputs, sizeof(inputs)));
  for (size_t i = 0; i < BATCH; ++i)
    inputs[i] = (const void*)(uintptr_t)(d_input + i * CHUNK_BYTES);
  CU(Fail, cuMemcpyHtoD(d_inputs, inputs, sizeof(inputs)));
  CU(Fail, cuMemAlloc(&d_block_data, sizeof(block_data)));
  CU(Fail, cuMemAlloc(&d_block_sizes, sizeof(block_sizes)));
  CU(Fail, cuMemAlloc(&d_block_offsets, sizeof(block_sizes)));
  CU(Fail, cuMemAlloc(&d_encoded, sizeof(encoded)));
  CU(Fail, cuMemAlloc(&d_sizes, sizeof(sizes)));
  CU(Fail, cuMemcpyHtoD(d_input, input, sizeof(input)));
  CU(Fail, cuMemcpyHtoD(d_block_data, block_data, sizeof(block_data)));
  CU(Fail, cuMemcpyHtoD(d_block_sizes, block_sizes, sizeof(block_sizes)));
  CU(Fail, cuStreamSynchronize(NULL));
  CU(Fail, cuMemsetD8Async(d_encoded, 0x5a, sizeof(encoded), stream));
  CHECK(
    Fail,
    gpu_blosc_pack_async(
      (struct gpu_blosc_frame_layout){
        CODEC_BLOSC_LZ4, CODEC_SHUFFLE_NONE, 1, CHUNK_BYTES, CHUNK_BYTES },
      (struct gpu_blosc_input){ (const void*)(uintptr_t)d_input, CHUNK_BYTES },
      (struct gpu_blosc_blocks){ (const void* const*)(uintptr_t)d_inputs,
                                 (const void*)(uintptr_t)d_block_data,
                                 STRIDE,
                                 (const size_t*)(uintptr_t)d_block_sizes,
                                 (size_t*)(uintptr_t)d_block_offsets },
      (struct gpu_blosc_output){
        (void*)(uintptr_t)d_encoded, STRIDE, (size_t*)(uintptr_t)d_sizes },
      BATCH,
      0,
      stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(encoded, d_encoded, sizeof(encoded)));
  CU(Fail, cuMemcpyDtoH(sizes, d_sizes, sizeof(sizes)));
  CU(Fail,
     cuMemcpyDtoH(actual_block_sizes, d_block_sizes, sizeof(block_sizes)));

  CHECK(Fail,
        memcmp(actual_block_sizes, block_sizes, sizeof(block_sizes)) == 0);
  CHECK(Fail, sizes[0] == CHUNK_BYTES + 15);
  CHECK(Fail, (encoded[2] & BLOSC_MEMCPYED) == 0);
  CHECK(Fail, get_u32le(encoded + 16) == 20);
  CHECK(Fail, get_u32le(encoded + 20) == CHUNK_BYTES - 9);
  // Near the fallback threshold, publishing final frame sizes must preserve
  // the block sizes and gather the compressed payload without raw overwrites.
  for (size_t i = 24; i < CHUNK_BYTES + 15; ++i)
    CHECK(Fail, encoded[i] == 0xa5);
  for (size_t i = 1; i < BATCH; ++i) {
    const uint8_t* chunk = encoded + i * STRIDE;
    CHECK(Fail, sizes[i] == CHUNK_BYTES + BLOSC_MAX_OVERHEAD);
    CHECK(Fail, (chunk[2] & BLOSC_MEMCPYED) != 0);
    CHECK(Fail,
          memcmp(chunk + BLOSC_MAX_OVERHEAD,
                 input + i * CHUNK_BYTES,
                 CHUNK_BYTES) == 0);
  }
  result = 0;

Fail:
  cu_mem_free(d_inputs);
  cu_mem_free(d_input);
  cu_mem_free(d_block_data);
  cu_mem_free(d_block_sizes);
  cu_mem_free(d_block_offsets);
  cu_mem_free(d_encoded);
  cu_mem_free(d_sizes);
  cu_stream_destroy(stream);
  return result;
}

RUN_GPU_TESTS({ "blosc_compressed_matrix", test_compressed_matrix },
              { "blosc_explicit_block_sizes", test_explicit_block_sizes },
              { "blosc_memcpy_fallbacks", test_memcpy_fallbacks },
              { "blosc_multiblock_boundaries", test_multiblock_boundaries },
              { "blosc_rebind_and_rejection", test_rebind_and_rejection },
              { "blosc_memory_estimate", test_memory_estimate },
              { "blosc_preparation_reservation", test_preparation_reservation },
              { "blosc_codec_allocation_estimate",
                test_codec_allocation_estimate },
              { "blosc_shuffle_filters", test_shuffle_filters },
              { "blosc_frame_boundary", test_frame_boundary },
              { "blosc_documented_frame", test_documented_frame }, )
