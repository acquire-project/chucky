#pragma once

#include "types.codec.h"
#include <stddef.h>

// Max compressed output size per chunk for CPU codecs (lz4/zstd).
size_t compress_cpu_max_output_size(enum compression_codec type,
                                    size_t chunk_bytes);

// Compress batch_size chunks.
//   Input:  src + i * input_stride  (each chunk_bytes bytes)
//   Output: dst + i * max_output_size
//   comp_sizes[i] receives actual compressed size.
// Returns 0 on success.
int compress_cpu(enum compression_codec codec,
                 const void* src,
                 size_t input_stride,
                 void* dst,
                 size_t max_output_size,
                 size_t* comp_sizes,
                 size_t chunk_bytes,
                 size_t batch_size);
