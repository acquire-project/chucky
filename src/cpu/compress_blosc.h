#pragma once

#include "types.codec.h"
#include <stddef.h>

// Return codes for compress_blosc:
//   0 = success
//   1 = compression error
//   2 = blosc not available (stub)
#define COMPRESS_BLOSC_NOT_AVAILABLE 2

size_t
compress_blosc_max_output_size(size_t chunk_bytes);

int
compress_blosc(struct codec_config codec,
               const void* src,
               size_t input_stride,
               void* dst,
               size_t max_output_size,
               size_t* comp_sizes,
               size_t chunk_bytes,
               size_t batch_size,
               size_t bytes_per_element);
