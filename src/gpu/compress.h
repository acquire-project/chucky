#pragma once

#include "types.codec.h"
#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct codec
  {
    enum compression_codec type;
    size_t alignment;       // required input chunk alignment
    size_t max_output_size; // max compressed bytes per chunk
    size_t chunk_bytes;     // uncompressed bytes per chunk
    size_t batch_size;      // number of chunks

    // Device state (owned, allocated by codec_init)
    size_t* d_comp_sizes;   // [batch_size] filled by codec_compress
    size_t* d_uncomp_sizes; // [batch_size] pre-filled with chunk_bytes
    void** d_ptrs;          // [2 * batch_size] scratch for nvcomp ptr arrays
    void* d_temp;           // workspace
    size_t temp_bytes;      // workspace size
  };

  // Query alignment for a codec type without full init.
  size_t codec_alignment(enum compression_codec type);

  // Query max compressed output size per chunk (no GPU allocation).
  size_t codec_max_output_size(enum compression_codec type, size_t chunk_bytes);

  // Query nvcomp workspace temp bytes (no GPU allocation).
  size_t codec_temp_bytes(enum compression_codec type,
                          size_t chunk_bytes,
                          size_t batch_size);

  // Init codec context. Allocates device memory. Returns 0 on success.
  int codec_init(struct codec* c,
                 enum compression_codec type,
                 size_t chunk_bytes,
                 size_t batch_size);

  // Free device resources.
  void codec_free(struct codec* c);

  // Compress batch_size chunks.
  //   Input:  d_input  + i * input_stride  (each chunk_bytes bytes)
  //   Output: d_output + i * max_output_size
  //   c->d_comp_sizes[i] filled with actual compressed size.
  // CODEC_NONE: single cuMemcpyDtoDAsync of batch_size * chunk_bytes bytes.
  // actual_batch_size: number of chunks to compress (0 = use c->batch_size).
  //   Must be <= c->batch_size. Allows partial batch compression without
  //   re-initializing the codec.
  // Returns 0 on success.
  int codec_compress(struct codec* c,
                     const void* d_input,
                     size_t input_stride,
                     void* d_output,
                     size_t actual_batch_size,
                     CUstream stream);

#ifdef __cplusplus
}
#endif
