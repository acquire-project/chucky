#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum compression_codec
  {
    CODEC_NONE,
    CODEC_LZ4,
    CODEC_ZSTD,
  };

#ifdef __cplusplus
}
#endif
