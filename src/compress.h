#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum compression_codec
  {
    CODEC_NONE,
    CODEC_LZ4,
    CODEC_ZSTD,
  };

  struct codec
  {
    enum compression_codec type;
    size_t alignment;       // required input chunk alignment
    size_t max_output_size; // max compressed bytes per chunk
    size_t chunk_size;      // uncompressed bytes per chunk
    size_t batch_size;      // number of chunks

    // Device state (owned, allocated by codec_init)
    size_t* d_comp_sizes;   // [batch_size] filled by codec_compress
    size_t* d_uncomp_sizes; // [batch_size] pre-filled with chunk_size
    void** d_ptrs;          // [2 * batch_size] scratch for nvcomp ptr arrays
    void* d_temp;           // workspace
    size_t temp_bytes;      // workspace size
  };

  // Query alignment for a codec type without full init.
  size_t codec_alignment(enum compression_codec type);

  // Init codec context. Allocates device memory. Returns 1 on success.
  int codec_init(struct codec* c,
                 enum compression_codec type,
                 size_t chunk_size,
                 size_t batch_size);

  // Free device resources.
  void codec_free(struct codec* c);

  // Compress batch_size chunks.
  //   Input:  d_input  + i * input_stride  (each chunk_size bytes)
  //   Output: d_output + i * max_output_size
  //   c->d_comp_sizes[i] filled with actual compressed size.
  // CODEC_NONE: single cuMemcpyDtoDAsync of batch_size * chunk_size bytes.
  // Returns 1 on success.
  int codec_compress(struct codec* c,
                     const void* d_input,
                     size_t input_stride,
                     void* d_output,
                     CUstream stream);

#ifdef __cplusplus
}
#endif
