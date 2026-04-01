#include "cpu/compress_blosc.h"

#include "util/prelude.h"

size_t
compress_blosc_max_output_size(size_t chunk_bytes)
{
  // Conservative estimate matching blosc's BLOSC_MAX_OVERHEAD (16 bytes).
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
               size_t bytes_per_element)
{
  (void)src;
  (void)input_stride;
  (void)dst;
  (void)max_output_size;
  (void)comp_sizes;
  (void)chunk_bytes;
  (void)batch_size;
  (void)bytes_per_element;
  log_error("blosc codec requested but not compiled in");
  return COMPRESS_BLOSC_NOT_AVAILABLE;
}
