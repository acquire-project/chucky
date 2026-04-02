#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum compression_codec
  {
    CODEC_NONE,
    CODEC_LZ4_RAW, // Raw LZ4 block format (not framed). Not interoperable
                    // with existing zarr v3 readers (e.g. numcodecs).
    CODEC_ZSTD,
  };

#ifdef __cplusplus
}
#endif
