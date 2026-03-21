#ifndef TEST_GPU_HELPERS_H
#define TEST_GPU_HELPERS_H

#include "stream.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

// Canonical rank-3 bounded dims: "zyx" sizes {4,4,6}, chunks {2,2,3}, cps
// {2,2,2}.
uint8_t
make_test_dims_3d(struct dimension* dims);

// Canonical rank-3 unbounded dims: "zyx" sizes {0,4,6}, chunks {2,2,3}, cps
// {2,2,2}.
uint8_t
make_test_dims_3d_unbounded(struct dimension* dims);

// Build a rank-3 tile_stream_configuration for testing.
// Shape: dims {4,4,6}, chunks {2,2,3}, chunks_per_shard {2,2,2}.
int
make_test_config(struct tile_stream_configuration* config,
                 struct dimension* dims,
                 enum compression_codec codec,
                 uint8_t epochs_per_batch);

// Fill one epoch of chunk pool on device.
// Each chunk t gets all elements set to fill_fn(t).
int
fill_pool_epoch(CUdeviceptr pool_buf,
                uint64_t n_chunks,
                uint64_t chunk_stride,
                size_t bpe,
                uint16_t (*fill_fn)(uint64_t chunk));

// Canned fill value functions for testing.
uint16_t
fill_epoch0(uint64_t c);
uint16_t
fill_epoch1(uint64_t c);
uint16_t
fill_epoch2(uint64_t c);
uint16_t
fill_epoch3(uint64_t c);

#endif // TEST_GPU_HELPERS_H
