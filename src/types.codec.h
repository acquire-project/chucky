#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum compression_codec
  {
    CODEC_NONE,
    CODEC_LZ4,
    CODEC_ZSTD,
    CODEC_BLOSC_LZ4,
    CODEC_BLOSC_ZSTD,
  };

  enum codec_shuffle
  {
    CODEC_SHUFFLE_NONE = 0,
    CODEC_SHUFFLE_BYTE = 1,
    CODEC_SHUFFLE_BIT = 2,
  };

  struct codec_config
  {
    enum compression_codec id;
    uint8_t level;              // passed to codec as-is
    enum codec_shuffle shuffle; // blosc only
  };

#ifdef __cplusplus
}
#endif
