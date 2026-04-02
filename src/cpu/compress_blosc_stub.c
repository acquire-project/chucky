#include "cpu/compress_blosc.h"

#include "util/prelude.h"

int
compress_blosc_validate(struct codec_config codec)
{
  if (!codec_is_blosc(codec.id))
    return 1;
  log_error("blosc codec requested but not compiled in");
  return 1;
}

size_t
compress_blosc_max_output_size(size_t chunk_bytes)
{
  return chunk_bytes + 16;
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
               int max_threads)
{
  (void)codec;
  (void)src;
  (void)input_stride;
  (void)dst;
  (void)max_output_size;
  (void)comp_sizes;
  (void)chunk_bytes;
  (void)batch_size;
  (void)bytes_per_element;
  (void)max_threads;
  log_error("blosc codec requested but not compiled in");
  return 1;
}
