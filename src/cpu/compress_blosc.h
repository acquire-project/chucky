#pragma once

#include "types.codec.h"
#include <stddef.h>

// Check blosc codec availability. Returns 0 if blosc is available, non-zero
// if not (stub build). Rejects non-blosc codec ids.
// Level range is validated separately in validate_config.
int
compress_blosc_validate(struct codec_config codec);

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
               size_t bytes_per_element,
               int nthreads);
