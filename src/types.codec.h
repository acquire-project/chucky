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

  // Query alignment for a codec type without full init.
  size_t codec_alignment(enum compression_codec type);

  // Query max compressed output size per chunk (no GPU allocation).
  size_t codec_max_output_size(enum compression_codec type, size_t chunk_bytes);

#ifdef __cplusplus
}
#endif
