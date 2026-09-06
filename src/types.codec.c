#include "types.codec.h"
#include "log/log.h"

int
codec_config_validate_blosc(struct codec_config config)
{
  if (!codec_is_blosc(config.id))
    return 0;
  if (config.level > 9) {
    log_error("blosc level must be 0..9 (got %u)", config.level);
    return 1;
  }
  if (config.shuffle < CODEC_SHUFFLE_NONE ||
      config.shuffle > CODEC_SHUFFLE_BIT) {
    log_error("invalid blosc shuffle mode %d", (int)config.shuffle);
    return 1;
  }
  const uint32_t max_block_bytes = (INT32_MAX - 255 * 4) / 3;
  if (config.blosc_block_bytes < 128 ||
      config.blosc_block_bytes > max_block_bytes) {
    log_error(
      "blosc_block_bytes must be 128..%u (got %u); no automatic default",
      max_block_bytes,
      config.blosc_block_bytes);
    return 1;
  }
  return 0;
}

int
codec_is_blosc(enum compression_codec c)
{
  return c == CODEC_BLOSC_LZ4 || c == CODEC_BLOSC_ZSTD;
}

int
codec_is_gpu_supported(enum compression_codec c)
{
  return c == CODEC_NONE || c == CODEC_LZ4_NON_STANDARD || c == CODEC_ZSTD ||
         codec_is_blosc(c);
}
