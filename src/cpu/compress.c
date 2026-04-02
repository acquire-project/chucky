#include "cpu/compress.h"
#include "cpu/compress_blosc.h"

#include "util/prelude.h"

#include <lz4hc.h>
#include <omp.h>
#include <stdatomic.h>
#include <string.h>
#include <zstd.h>

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
    case CODEC_BLOSC_LZ4:
    case CODEC_BLOSC_ZSTD:
      return compress_blosc_max_output_size(chunk_bytes);
    default:
      return 0;
  }
}

int
compress_cpu(struct codec_config codec,
             const void* src,
             size_t input_stride,
             void* dst,
             size_t max_output_size,
             size_t* comp_sizes,
             size_t chunk_bytes,
             size_t batch_size,
             size_t bytes_per_element,
             int nthreads)
{
  int i;
  switch (codec.id) {
    case CODEC_NONE:
#pragma omp parallel for schedule(static) if (batch_size > 1024) num_threads(nthreads)
      for (i = 0; i < (int)batch_size; ++i) {
        memcpy((char*)dst + i * max_output_size,
               (const char*)src + i * input_stride,
               chunk_bytes);
        comp_sizes[i] = chunk_bytes;
      }
      return 0;

    case CODEC_LZ4: {
      _Atomic int err = 0;
      int level = codec.level;
#pragma omp parallel for schedule(dynamic) if (batch_size > 1024) num_threads(nthreads)
      for (i = 0; i < (int)batch_size; ++i) {
        if (err)
          continue;
        const char* in = (const char*)src + i * input_stride;
        char* out = (char*)dst + i * max_output_size;
        int rc = LZ4_compress_HC(
          in, out, (int)chunk_bytes, (int)max_output_size, level);
        if (rc <= 0)
          err = 1;
        else
          comp_sizes[i] = (size_t)rc;
      }
      return err;
    }

    case CODEC_ZSTD: {
      int level = codec.level;
      _Atomic int err = 0;
#pragma omp parallel for schedule(dynamic) if (batch_size > 1024) num_threads(nthreads)
      for (i = 0; i < (int)batch_size; ++i) {
        if (err)
          continue;
        const void* in = (const char*)src + i * input_stride;
        void* out = (char*)dst + i * max_output_size;
        size_t rc = ZSTD_compress(out, max_output_size, in, chunk_bytes, level);
        if (ZSTD_isError(rc))
          err = 1;
        else
          comp_sizes[i] = rc;
      }
      return err;
    }

    case CODEC_BLOSC_LZ4:
    case CODEC_BLOSC_ZSTD:
      return compress_blosc(codec,
                            src,
                            input_stride,
                            dst,
                            max_output_size,
                            comp_sizes,
                            chunk_bytes,
                            batch_size,
                            bytes_per_element,
                            nthreads);

    default:
      return 1;
  }
}
