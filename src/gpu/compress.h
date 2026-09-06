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
    struct codec_config config;
    size_t max_output_size; // max valid encoded bytes per chunk
    size_t output_stride;   // fixed device slot stride (may be larger)
    size_t chunk_bytes;     // active uncompressed bytes per chunk
    size_t chunk_capacity;  // largest input size accepted by this instance
    size_t typesize;        // active Blosc element size
    size_t batch_size;      // number of chunks

    // Raw codecs use one block per chunk.
    size_t block_bytes;
    size_t blocks_per_chunk;
    size_t block_output_stride;
    size_t block_input_stride;
    void* d_block_input;     // optional aligned, filtered block slots
    void* d_block_data;      // aligned raw nvCOMP block destinations
    size_t* d_block_sizes;   // raw nvCOMP output sizes
    size_t* d_block_offsets; // per-block record offsets within each frame

    // Owned device state
    size_t* d_comp_sizes;   // encoded sizes [batch_size]
    size_t* d_uncomp_sizes; // one size per nvCOMP input at capacity
    void** d_ptrs;          // twice the nvCOMP input count
    void* d_temp;           // workspace
    size_t temp_bytes;      // workspace size
  };

  // Alignment required of every uncompressed chunk handed to the codec.
  size_t codec_alignment(enum compression_codec type);

  // Alignment required of every compressed chunk in the output pool.
  size_t codec_output_alignment(enum compression_codec type);

  // Maximum valid encoded size per chunk. Does not allocate device memory.
  size_t codec_max_output_size(enum compression_codec type, size_t chunk_bytes);

  // Required byte stride between encoded chunks in the output buffer.
  size_t codec_output_stride(enum compression_codec type, size_t chunk_bytes);

  // Required device bytes, excluding runtime overhead. Does not allocate.
  // Includes preparation for configured shuffle or block alignment.
  // reserve_shuffle adds storage for future shuffled bindings.
  size_t codec_device_bytes(struct codec_config config,
                            size_t chunk_bytes,
                            size_t batch_size,
                            int reserve_shuffle);

  // Init raw codec context. Blosc requires codec_init_config with an explicit
  // block size. Allocates device memory. Returns 0 on success.
  int codec_init(struct codec* c,
                 enum compression_codec type,
                 size_t chunk_bytes,
                 size_t batch_size);

  // Initial shuffle/alignment always reserves preparation storage.
  // reserve_shuffle additionally supports future shuffled bindings.
  int codec_init_config(struct codec* c,
                        struct codec_config config,
                        size_t typesize,
                        size_t chunk_bytes,
                        size_t batch_size,
                        int reserve_shuffle);

  // Bind per-array state and geometry. Blosc block size must match the initial
  // configuration; chunk_bytes must not exceed chunk_capacity. Required
  // preparation must fit existing storage. Rejection preserves the binding.
  int codec_bind(struct codec* c,
                 struct codec_config config,
                 size_t typesize,
                 size_t chunk_bytes,
                 CUstream stream);

  // Validate the GPU-specific Blosc contract without allocating device state.
  int codec_validate_gpu(struct codec_config config,
                         size_t typesize,
                         size_t chunk_bytes);

  // Free device resources.
  void codec_free(struct codec* c);

  // Compress batch_size chunks.
  //   Input:  d_input  + i * input_stride  (each chunk_bytes bytes)
  //   Output: d_output + i * output_stride (CODEC_NONE packs at chunk_bytes)
  //   c->d_comp_sizes[i] filled with actual compressed size.
  // actual_batch_size: number of chunks to compress (0 = use c->batch_size).
  //   Must be <= c->batch_size.
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
