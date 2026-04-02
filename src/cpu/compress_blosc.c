#include "cpu/compress_blosc.h"

#include <blosc.h>
#include <omp.h>
#include <stdatomic.h>

// Real impl always succeeds for blosc ids. The stub overrides this to
// return 1 with an error, providing build-time unavailability detection.
int
compress_blosc_validate(struct codec_config codec)
{
  if (!codec_is_blosc(codec.id))
    return 1;
  return 0;
}

size_t
compress_blosc_max_output_size(size_t chunk_bytes)
{
  return chunk_bytes + BLOSC_MAX_OVERHEAD;
}

int
compress_blosc(struct codec_config codec,
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
  const char* compname =
    codec.id == CODEC_BLOSC_LZ4 ? BLOSC_LZ4_COMPNAME : BLOSC_ZSTD_COMPNAME;
  int clevel = codec.level;
  int doshuffle = codec.shuffle;
  size_t typesize = bytes_per_element > 0 ? bytes_per_element : 1;
  _Atomic int err = 0;
  int i;
#pragma omp parallel for schedule(dynamic) if (batch_size > 1024) num_threads(nthreads)
  for (i = 0; i < (int)batch_size; ++i) {
    if (err)
      continue;
    const void* in = (const char*)src + i * input_stride;
    void* out = (char*)dst + i * max_output_size;
    int rc = blosc_compress_ctx(clevel,
                                doshuffle,
                                typesize,
                                chunk_bytes,
                                in,
                                out,
                                max_output_size,
                                compname,
                                0,  // blocksize (auto)
                                1); // numinternalthreads
    if (rc <= 0)
      err = 1;
    else
      comp_sizes[i] = (size_t)rc;
  }
  return err;
}
