#include "cpu/compress.h"

#include "util/prelude.h"

#include <lz4.h>
#include <stdatomic.h>
#include <string.h>
#include <zstd.h>

#define ZSTD_COMPRESS_LEVEL 3

size_t
compress_cpu_max_output_size(enum compression_codec type, size_t chunk_bytes)
{
  switch (type) {
    case CODEC_NONE:
      return chunk_bytes;
    case CODEC_LZ4:
      return (size_t)LZ4_compressBound((int)chunk_bytes);
    case CODEC_ZSTD:
      return ZSTD_compressBound(chunk_bytes);
    default:
      return 0;
  }
}

int
compress_cpu(enum compression_codec codec,
             const void* src,
             size_t input_stride,
             void* dst,
             size_t max_output_size,
             size_t* comp_sizes,
             size_t chunk_bytes,
             size_t batch_size)
{
  int i;
  switch (codec) {
    case CODEC_NONE:
#pragma omp parallel for schedule(static) if (batch_size > 1024)
      for (i = 0; i < (int)batch_size; ++i) {
        memcpy((char*)dst + i * max_output_size,
               (const char*)src + i * input_stride,
               chunk_bytes);
        comp_sizes[i] = chunk_bytes;
      }
      return 0;

    case CODEC_LZ4: {
      _Atomic int err = 0;
#pragma omp parallel for schedule(dynamic) if (batch_size > 1024)
      for (i = 0; i < (int)batch_size; ++i) {
        if (err)
          continue;
        const char* in = (const char*)src + i * input_stride;
        char* out = (char*)dst + i * max_output_size;
        int rc =
          LZ4_compress_default(in, out, (int)chunk_bytes, (int)max_output_size);
        if (rc <= 0)
          err = 1;
        else
          comp_sizes[i] = (size_t)rc;
      }
      return err;
    }

    case CODEC_ZSTD: {
      _Atomic int err = 0;
#pragma omp parallel for schedule(dynamic) if (batch_size > 1024)
      for (i = 0; i < (int)batch_size; ++i) {
        if (err)
          continue;
        const void* in = (const char*)src + i * input_stride;
        void* out = (char*)dst + i * max_output_size;
        size_t rc = ZSTD_compress(
          out, max_output_size, in, chunk_bytes, ZSTD_COMPRESS_LEVEL);
        if (ZSTD_isError(rc))
          err = 1;
        else
          comp_sizes[i] = rc;
      }
      return err;
    }

    default:
      return 1;
  }
}
