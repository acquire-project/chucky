#include "test_gpu_helpers.h"
#include "dimension.h"

#include "prelude.cuda.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>

uint8_t
make_test_dims_3d(struct dimension* dims)
{
  uint8_t rank = dims_create(dims, "zyx", (uint64_t[]){ 4, 4, 6 });
  dims_set_tile_sizes(dims, rank, (uint64_t[]){ 2, 2, 3 });
  dims_set_shard_counts(dims, rank, (uint64_t[]){ 1, 1, 1 });
  return rank;
}

uint8_t
make_test_dims_3d_unbounded(struct dimension* dims)
{
  uint8_t rank = dims_create(dims, "zyx", (uint64_t[]){ 0, 4, 6 });
  dims_set_tile_sizes(dims, rank, (uint64_t[]){ 2, 2, 3 });
  dims[0].tiles_per_shard = 2; // unbounded: must set directly
  dims_set_shard_counts(dims, rank, (uint64_t[]){ 0, 1, 1 });
  return rank;
}

int
make_test_config(struct tile_stream_configuration* config,
                 struct dimension* dims,
                 enum compression_codec codec,
                 uint8_t epochs_per_batch)
{
  make_test_dims_3d(dims);

  memset(config, 0, sizeof(*config));
  config->rank = 3;
  config->dimensions = dims;
  config->dtype = lod_dtype_u16;
  config->buffer_capacity_bytes = 4096;
  config->codec = codec;
  config->shard_alignment = 0;
  config->epochs_per_batch = epochs_per_batch;
  return 0;
}

int
make_test_config_full(struct tile_stream_configuration* config,
                      struct dimension* dims,
                      enum compression_codec codec,
                      uint8_t epochs_per_batch,
                      struct shard_sink* sink)
{
  make_test_config(config, dims, codec, epochs_per_batch);
  config->shard_sink = sink;
  return 0;
}

int
fill_pool_epoch(CUdeviceptr pool_buf,
                uint64_t tiles,
                uint64_t tile_stride,
                size_t bpe,
                uint16_t (*fill_fn)(uint64_t tile))
{
  size_t epoch_bytes = tiles * tile_stride * bpe;
  uint16_t* h = (uint16_t*)malloc(epoch_bytes);
  CHECK(Fail, h);
  memset(h, 0, epoch_bytes);

  for (uint64_t t = 0; t < tiles; ++t) {
    uint16_t val = fill_fn(t);
    uint16_t* tile_data = h + t * tile_stride;
    for (uint64_t e = 0; e < tile_stride; ++e)
      tile_data[e] = val;
  }

  CU(Fail, cuMemcpyHtoD(pool_buf, h, epoch_bytes));
  free(h);
  return 0;

Fail:
  free(h);
  return 1;
}

uint16_t
fill_epoch0(uint64_t t)
{
  return (uint16_t)(t + 1);
}

uint16_t
fill_epoch1(uint64_t t)
{
  return (uint16_t)(t + 100);
}

uint16_t
fill_epoch2(uint64_t t)
{
  return (uint16_t)(t + 200);
}

uint16_t
fill_epoch3(uint64_t t)
{
  return (uint16_t)(t + 300);
}
