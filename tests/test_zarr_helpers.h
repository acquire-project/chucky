// Shared test helpers for creating zarr stores via the new layered API.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "ngff/ngff_axis.h"
#include "ngff/ngff_multiscale.h"
#include "types.codec.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"
#include "zarr/zarr_array.h"

#include <stdint.h>

// --- Single zarr v3 array ---

struct test_zarr_sink
{
  struct store* store;
  struct shard_pool* pool;
  struct zarr_array* array;
};

int
test_zarr_sink_open(struct test_zarr_sink* z,
                    const char* store_path,
                    const char* array_name,
                    const struct dimension* dims,
                    uint8_t rank,
                    enum dtype data_type,
                    double fill_value,
                    struct codec_config codec,
                    int unbuffered);

struct shard_sink*
test_zarr_sink_as_shard_sink(struct test_zarr_sink* z);

size_t
test_zarr_sink_pending_bytes(struct test_zarr_sink* z);

void
test_zarr_sink_flush(struct test_zarr_sink* z);

void
test_zarr_sink_close(struct test_zarr_sink* z);

// --- OME-NGFF multiscale ---

struct test_zarr_multiscale
{
  struct store* store;
  struct shard_pool* pool;
  struct ngff_multiscale* ms;
};

int
test_zarr_multiscale_open(struct test_zarr_multiscale* z,
                          const char* store_path,
                          const char* array_name,
                          const struct dimension* dims,
                          uint8_t rank,
                          enum dtype data_type,
                          double fill_value,
                          int nlod,
                          struct codec_config codec,
                          const struct ngff_axis* axes,
                          int unbuffered);

struct shard_sink*
test_zarr_multiscale_as_shard_sink(struct test_zarr_multiscale* z);

void
test_zarr_multiscale_flush(struct test_zarr_multiscale* z);

void
test_zarr_multiscale_close(struct test_zarr_multiscale* z);
