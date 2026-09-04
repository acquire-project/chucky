#include "gpu/blosc.frame.h"
#include "gpu/compress.h"
#include "gpu/prelude.cuda.h"
#include "stream.gpu.h"
#include "util/prelude.h"

#include <blosc.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_runner.h"

enum pattern
{
  PATTERN_ZERO,
  PATTERN_RAMP,
  PATTERN_RANDOM,
};

static uint32_t
get_u32le(const uint8_t* p)
{
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

static void
fill_input(uint8_t* p, size_t bytes, enum pattern pattern)
{
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
             int expect_memcpy)
{
  uint8_t* recovered = NULL;
  size_t validated_nbytes = 0;
  size_t meta_typesize = 0;
  int flags = 0;
  int result = 1;

  CHECK(Fail, encoded_bytes > 0 && encoded_bytes <= c->max_output_size);
  CHECK(Fail, encoded[0] == 2 && encoded[1] == 1);
  CHECK(Fail, encoded[3] == c->typesize);
  CHECK(Fail, get_u32le(encoded + 4) == c->chunk_bytes);
  CHECK(Fail, get_u32le(encoded + 8) == c->chunk_bytes);
  CHECK(Fail, get_u32le(encoded + 12) == encoded_bytes);
  CHECK(Fail, (encoded[2] & 0x10) != 0);
  CHECK(Fail, ((encoded[2] >> 5) & 7) == (c->type == CODEC_BLOSC_LZ4 ? 1 : 4));
  CHECK(Fail,
        ((encoded[2] & BLOSC_DOSHUFFLE) != 0) ==
          (c->config.shuffle == CODEC_SHUFFLE_BYTE));
  CHECK(Fail, ((encoded[2] & BLOSC_MEMCPYED) != 0) == expect_memcpy);
  if (expect_memcpy) {
    CHECK(Fail, encoded_bytes == c->chunk_bytes + BLOSC_MAX_OVERHEAD);
    CHECK(Fail,
          memcmp(encoded + BLOSC_MAX_OVERHEAD, expected, c->chunk_bytes) == 0);
  } else {
    CHECK(Fail, get_u32le(encoded + 16) == 20);
    CHECK(Fail, get_u32le(encoded + 20) + 24 == encoded_bytes);
  }

  CHECK(Fail,
        blosc_cbuffer_validate(encoded, encoded_bytes, &validated_nbytes) == 0);
  CHECK(Fail, validated_nbytes == c->chunk_bytes);
  blosc_cbuffer_metainfo(encoded, &meta_typesize, &flags);
  CHECK(Fail, meta_typesize == c->typesize);
  CHECK(Fail, flags == (encoded[2] & 7));
  CHECK(Fail,
        strcmp(blosc_cbuffer_complib(encoded),
               c->type == CODEC_BLOSC_LZ4 ? "LZ4" : "Zstd") == 0);

  recovered = (uint8_t*)malloc(c->chunk_bytes);
  CHECK(Fail, recovered);
  CHECK(Fail,
        blosc_decompress_ctx(encoded, recovered, c->chunk_bytes, 1) ==
          (int)c->chunk_bytes);
  CHECK(Fail, memcmp(recovered, expected, c->chunk_bytes) == 0);
  result = 0;

Fail:
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
        codec_init_config(&c,
                          config,
                          typesize,
                          chunk_bytes,
                          batch,
                          config.shuffle == CODEC_SHUFFLE_BYTE) == 0);
  CHECK(Fail, c.max_output_size == chunk_bytes + BLOSC_MAX_OVERHEAD);
  CHECK(Fail, c.output_stride >= c.max_output_size);
  {
    const int reserve = config.shuffle == CODEC_SHUFFLE_BYTE;
    const size_t expected = 2 * batch * sizeof(size_t) +
                            2 * batch * sizeof(void*) + c.temp_bytes +
                            (reserve ? batch * c.shuffle_stride : 0);
    CHECK(Fail,
          codec_device_bytes(config.id, chunk_bytes, batch, reserve) ==
            expected);
  }

  input = (uint8_t*)malloc(batch * input_stride);
  encoded = (uint8_t*)malloc(batch * c.output_stride);
  sizes = (size_t*)malloc(batch * sizeof(size_t));
  CHECK(Fail, input && encoded && sizes);
  for (size_t i = 0; i < batch; ++i)
    fill_input(input + i * input_stride, chunk_bytes, pattern);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_input, batch * input_stride));
  CU(Fail, cuMemAlloc(&d_encoded, batch * c.output_stride));
  CU(Fail, cuMemcpyHtoD(d_input, input, batch * input_stride));
  CHECK(Fail,
        codec_compress(&c,
                       (const void*)(uintptr_t)d_input,
                       input_stride,
                       (void*)(uintptr_t)d_encoded,
                       actual_batch,
                       stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(encoded, d_encoded, actual_batch * c.output_stride));
  CU(Fail,
     cuMemcpyDtoH(
       sizes, (CUdeviceptr)c.d_comp_sizes, actual_batch * sizeof(size_t)));

  for (size_t i = 0; i < actual_batch; ++i)
    CHECK(Fail,
          verify_chunk(&c,
                       encoded + i * c.output_stride,
                       sizes[i],
                       input + i * input_stride,
                       expect_memcpy) == 0);
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
  const enum codec_shuffle shuffles[] = { CODEC_SHUFFLE_NONE,
                                          CODEC_SHUFFLE_BYTE };
  const size_t typesizes[] = { 1, 2, 4, 8 };
  for (size_t c = 0; c < countof(codecs); ++c)
    for (size_t s = 0; s < countof(shuffles); ++s)
      for (size_t t = 0; t < countof(typesizes); ++t) {
        struct codec_config config = { .id = codecs[c],
                                       .level = 5,
                                       .shuffle = shuffles[s] };
        CHECK(Fail,
              run_case(config, typesizes[t], 64 * 1024, PATTERN_ZERO, 0, 2) ==
                0);
      }
  {
    struct codec_config tail = { .id = CODEC_BLOSC_ZSTD,
                                 .level = 5,
                                 .shuffle = CODEC_SHUFFLE_BYTE };
    CHECK(Fail, run_case(tail, 8, 65539, PATTERN_RAMP, 0, 2) == 0);
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
    struct codec_config level_zero = { .id = codecs[i],
                                       .level = 0,
                                       .shuffle = CODEC_SHUFFLE_BYTE };
    struct codec_config enabled = { .id = codecs[i],
                                    .level = 5,
                                    .shuffle = CODEC_SHUFFLE_BYTE };
    struct codec_config incompressible = enabled;
    incompressible.shuffle = CODEC_SHUFFLE_NONE;
    CHECK(Fail, run_case(level_zero, 8, 4096, PATTERN_ZERO, 1, 3) == 0);
    CHECK(Fail, run_case(enabled, 4, 127, PATTERN_ZERO, 1, 3) == 0);
    CHECK(Fail,
          run_case(incompressible, 4, 64 * 1024, PATTERN_RANDOM, 1, 3) == 0);
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
                                  .shuffle = CODEC_SHUFFLE_NONE };
  struct codec_config rebound = { .id = CODEC_BLOSC_ZSTD,
                                  .level = 8,
                                  .shuffle = CODEC_SHUFFLE_BYTE };
  struct codec_config bitshuffle = { .id = CODEC_BLOSC_ZSTD,
                                     .level = 5,
                                     .shuffle = CODEC_SHUFFLE_BIT };
  int result = 1;

  CHECK(Fail, codec_validate_gpu(bitshuffle, 4, 4096) != 0);
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, codec_init_config(&c, initial, 2, 65536, 3, 1) == 0);
  CHECK(Fail, codec_bind(&c, rebound, 8, 4096, stream) == 0);
  CHECK(Fail, c.chunk_bytes == 4096 && c.typesize == 8);
  CHECK(Fail, c.config.level == 8 && c.config.shuffle == CODEC_SHUFFLE_BYTE);
  CHECK(Fail, codec_bind(&c, bitshuffle, 8, 4096, stream) != 0);
  result = 0;

Fail:
  codec_free(&c);
  cu_stream_destroy(stream);
  return result;
}

static int
test_memory_estimate(void)
{
  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 0, 16, 16 });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 8, 8 });
  dims[0].chunks_per_shard = 4;
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_BLOSC_ZSTD,
               .level = 5,
               .shuffle = CODEC_SHUFFLE_NONE },
    .epochs_per_batch = 2,
  };

  for (int shuffled = 0; shuffled < 2; ++shuffled) {
    config.codec.shuffle = shuffled ? CODEC_SHUFFLE_BYTE : CODEC_SHUFFLE_NONE;
    struct tile_stream_memory_info info = { 0 };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, 0, &info) == 0);
    CHECK(Fail, info.max_output_size > 16);
    const size_t chunk_bytes = info.max_output_size - 16;
    const size_t batch = info.epochs_per_batch * info.total_chunks;
    CHECK(Fail,
          info.codec_bytes ==
            codec_device_bytes(config.codec.id, chunk_bytes, batch, shuffled));
    CHECK(Fail,
          info.compressed_pool_bytes ==
            2 * batch * codec_output_stride(config.codec.id, chunk_bytes));
  }
  return 0;

Fail:
  return 1;
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
  uint8_t encoded[BATCH * STRIDE];
  size_t sizes[BATCH] = { CHUNK_BYTES - 9, CHUNK_BYTES - 8, CHUNK_BYTES - 7 };
  CUdeviceptr d_input = 0;
  CUdeviceptr d_encoded = 0;
  CUdeviceptr d_sizes = 0;
  CUstream stream = 0;
  int result = 1;

  fill_input(input, sizeof(input), PATTERN_RAMP);
  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc(&d_input, sizeof(input)));
  CU(Fail, cuMemAlloc(&d_encoded, sizeof(encoded)));
  CU(Fail, cuMemAlloc(&d_sizes, sizeof(sizes)));
  CU(Fail, cuMemcpyHtoD(d_input, input, sizeof(input)));
  CU(Fail, cuMemcpyHtoD(d_sizes, sizes, sizeof(sizes)));
  CHECK(Fail,
        gpu_blosc_finalize_async(CODEC_BLOSC_LZ4,
                                 CODEC_SHUFFLE_NONE,
                                 1,
                                 CHUNK_BYTES,
                                 (const void*)(uintptr_t)d_input,
                                 CHUNK_BYTES,
                                 (void*)(uintptr_t)d_encoded,
                                 STRIDE,
                                 (size_t*)(uintptr_t)d_sizes,
                                 BATCH,
                                 0,
                                 stream) == 0);
  CU(Fail, cuStreamSynchronize(stream));
  CU(Fail, cuMemcpyDtoH(encoded, d_encoded, sizeof(encoded)));
  CU(Fail, cuMemcpyDtoH(sizes, d_sizes, sizeof(sizes)));

  CHECK(Fail, sizes[0] == CHUNK_BYTES + 15);
  CHECK(Fail, (encoded[2] & BLOSC_MEMCPYED) == 0);
  CHECK(Fail, get_u32le(encoded + 16) == 20);
  CHECK(Fail, get_u32le(encoded + 20) == CHUNK_BYTES - 9);
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
  cu_mem_free(d_input);
  cu_mem_free(d_encoded);
  cu_mem_free(d_sizes);
  cu_stream_destroy(stream);
  return result;
}

RUN_GPU_TESTS({ "blosc_compressed_matrix", test_compressed_matrix },
              { "blosc_memcpy_fallbacks", test_memcpy_fallbacks },
              { "blosc_rebind_and_rejection", test_rebind_and_rejection },
              { "blosc_memory_estimate", test_memory_estimate },
              { "blosc_frame_boundary", test_frame_boundary }, )
