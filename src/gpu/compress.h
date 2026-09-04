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
    struct codec_config config;
    size_t max_output_size; // max valid encoded bytes per chunk
    size_t output_stride;   // fixed device slot stride (may be larger)
    size_t chunk_bytes;     // active uncompressed bytes per chunk
    size_t chunk_capacity;  // largest input size accepted by this instance
    size_t typesize;        // active Blosc element size
    size_t shuffle_stride;  // aligned per-chunk scratch stride
    int has_shuffle_scratch;
    size_t batch_size; // number of chunks

    // Blosc partitions the existing contiguous chunk layout into small nvCOMP
    // inputs. Rebinding changes active geometry, never these allocations.
    size_t block_bytes;
    size_t blocks_per_chunk;
    size_t block_capacity; // maximum flattened chunk/block count
    size_t block_output_stride;
    void* d_block_data;      // aligned raw nvCOMP block destinations
    size_t* d_block_sizes;   // raw nvCOMP output sizes
    size_t* d_block_offsets; // per-block record offsets within each frame

    // Device state (owned, allocated by codec_init)
    size_t* d_comp_sizes;   // [batch_size] filled by codec_compress
    size_t* d_uncomp_sizes; // [block_capacity for Blosc, else batch_size]
    void** d_ptrs;          // twice the nvCOMP input count
    void* d_temp;           // workspace
    size_t temp_bytes;      // workspace size
    void* d_shuffle;        // Blosc byte/bit-shuffle batch scratch
  };

  // Alignment required of every uncompressed chunk handed to the codec.
  size_t codec_alignment(enum compression_codec type);

  // Alignment required of every compressed chunk in the output pool.
  size_t codec_output_alignment(enum compression_codec type);

  // Query the maximum valid encoded size per chunk (no GPU allocation). Raw
  // nvCOMP codecs return an aligned slot bound; Blosc returns nbytes + 16.
  size_t codec_max_output_size(enum compression_codec type, size_t chunk_bytes);

  // Fixed device slot stride. Blosc packs into aligned nbytes + 16 slots;
  // its raw nvCOMP block destinations are separate codec-owned scratch.
  size_t codec_output_stride(enum compression_codec type, size_t chunk_bytes);

  // Total device bytes codec_init_config allocates (size arrays, ptr table,
  // nvCOMP temp, and optional shuffle scratch) — sizing mirror for the
  // memory estimate (no GPU allocation).
  size_t codec_device_bytes(enum compression_codec type,
                            size_t chunk_bytes,
                            size_t batch_size,
                            int reserve_shuffle);

  size_t codec_temp_bytes(enum compression_codec type,
                          size_t chunk_bytes,
                          size_t batch_size);

  // Init codec context. Allocates device memory. Returns 0 on success.
  int codec_init(struct codec* c,
                 enum compression_codec type,
                 size_t chunk_bytes,
                 size_t batch_size);

  int codec_init_config(struct codec* c,
                        struct codec_config config,
                        size_t typesize,
                        size_t chunk_bytes,
                        size_t batch_size,
                        int reserve_shuffle);

  // Select the active input geometry for a shared codec instance. The codec
  // allocations remain sized to chunk_capacity/output_stride; the per-chunk
  // size arrays are refreshed on stream before the next compression/aggregate.
  int codec_set_chunk_bytes(struct codec* c,
                            size_t chunk_bytes,
                            CUstream stream);

  // Bind all active per-array Blosc state as well as chunk geometry.
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
  //   Output: d_output + i * output_stride
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
