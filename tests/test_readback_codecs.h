// Shared codec matrix for single-array and NGFF reader interoperability.
#pragma once

#include "types.codec.h"
#include <stddef.h>

struct test_readback_codec
{
  const char* name;
  struct codec_config codec;
};

static inline const struct test_readback_codec*
test_readback_codecs(int gpu, size_t* count)
{
  static const struct test_readback_codec cpu_cases[] = {
    // lz4 omitted: no zarr v3 LZ4 codec spec; zarr-python can't read it.
    { "none", { .id = CODEC_NONE } },
    { "zstd", { .id = CODEC_ZSTD } },
    { "blosc_lz4",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 16 * 1024 } },
  };
  static const struct test_readback_codec gpu_cases[] = {
    { "none", { .id = CODEC_NONE } },
    { "zstd", { .id = CODEC_ZSTD } },
    { "blosc_lz4_noshuffle",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_NONE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_lz4_shuffle",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_lz4_bitshuffle",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BIT,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd_noshuffle",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_NONE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd_shuffle",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd_bitshuffle",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BIT,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_lz4_unaligned_blocks",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 4097 } },
    { "blosc_zstd_unaligned_blocks",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BIT,
        .blosc_block_bytes = 4097 } },
  };
  *count = gpu ? sizeof(gpu_cases) / sizeof(gpu_cases[0])
               : sizeof(cpu_cases) / sizeof(cpu_cases[0]);
  return gpu ? gpu_cases : cpu_cases;
}
