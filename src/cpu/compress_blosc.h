#pragma once

#include "types.codec.h"
#include <stddef.h>

struct threadpool;

// Check Blosc availability and settings, including an explicit block size.
// Returns 0 on success, non-zero for invalid config or an unavailable library.
// Rejects non-Blosc codec ids.
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
               struct threadpool* pool);
