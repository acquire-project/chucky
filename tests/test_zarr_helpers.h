// Shared test helpers for creating zarr stores via the new layered API.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "ngff.h"
#include "store.h"
#include "types.codec.h"
#include "zarr.h"

#include <stdint.h>

struct shard_pool;

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

// The store passed in is owned from here on, even when the open fails.
int
test_zarr_sink_open_with_pool(struct test_zarr_sink* z,
                              struct store* store,
                              const char* array_name,
                              const struct dimension* dims,
                              uint8_t rank,
                              enum dtype data_type,
                              struct codec_config codec);

struct shard_sink*
test_zarr_sink_as_shard_sink(struct test_zarr_sink* z);

int
test_zarr_sink_has_error(const struct test_zarr_sink* z);

void
test_zarr_sink_flush(struct test_zarr_sink* z);

void
test_zarr_sink_close(struct test_zarr_sink* z);

// --- OME-NGFF multiscale ---

struct test_zarr_multiscale
{
  struct store* store;
  struct ngff_multiscale* ms;
};

int
test_zarr_multiscale_open(struct test_zarr_multiscale* z,
                          const char* store_path,
                          const char* array_name,
                          const struct dimension* dims,
                          uint8_t rank,
                          enum dtype data_type,
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
