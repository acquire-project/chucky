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

  struct codec_config
  {
    enum compression_codec id;
    uint8_t level;   // 0 = codec default
    uint8_t shuffle; // blosc only: 0=none, 1=byte, 2=bit
  };

  static inline int codec_is_blosc(enum compression_codec c)
  {
    return c == CODEC_BLOSC_LZ4 || c == CODEC_BLOSC_ZSTD;
  }

#ifdef __cplusplus
}
#endif
