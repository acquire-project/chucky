#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum compression_codec
  {
    CODEC_NONE,
    CODEC_LZ4_NON_STANDARD, // Raw LZ4 block format (not framed). Not part of
                            // the zarr v3 standard.
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
    uint8_t level;              // LZ4: 1..12 (HC), 0 rejected.
                                // ZSTD: 0 valid (ZSTD default).
                                // Blosc: 0 = store only; on GPU, 1..9 are
                                // accepted hints for nvCOMP's single mode.
    enum codec_shuffle shuffle; // blosc only
    // Requested Blosc block size in bytes (128..715827542).
    // Required for Blosc, including level 0; zero is invalid.
    // Ignored by other codecs.
    uint32_t blosc_block_bytes;
  };

  // Validate backend-independent Blosc settings. Non-Blosc codecs return 0.
  int codec_config_validate_blosc(struct codec_config config);

  int codec_is_blosc(enum compression_codec c);

  int codec_is_gpu_supported(enum compression_codec c);

#ifdef __cplusplus
}
#endif
