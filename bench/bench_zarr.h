#pragma once

#include "dimension.h"
#include "dtype.h"
#include "ngff.h"
#include "store.h"
#include "types.codec.h"
#include "writer.h"
#include "zarr.h"

#include <stddef.h>
#include <stdint.h>

// Polymorphic handle for zarr sinks (FS/S3 x single/multiscale).
// At most one of array/ms is non-NULL.
struct bench_zarr_handle
{
  struct store* store;
  struct zarr_array* array;   // non-NULL for single-array
  struct ngff_multiscale* ms; // non-NULL for multiscale
};

int
bench_zarr_open_fs(struct bench_zarr_handle* z,
                   const char* store_path,
                   const char* array_name,
                   const struct dimension* dims,
                   uint8_t rank,
                   enum dtype data_type,
                   double fill_value,
                   struct codec_config codec,
                   int is_multiscale);

int
bench_zarr_open_s3(struct bench_zarr_handle* z,
                   const char* bucket,
                   const char* prefix,
                   const char* array_name,
                   const char* region,
                   const char* endpoint,
                   double throughput_gbps,
                   const struct dimension* dims,
                   uint8_t rank,
                   enum dtype data_type,
                   double fill_value,
                   struct codec_config codec,
                   int is_multiscale);

struct shard_sink*
bench_zarr_as_shard_sink(struct bench_zarr_handle* z);

// Flush all pending I/O. Returns non-zero on error.
int
bench_zarr_flush(struct bench_zarr_handle* z);

// Return number of bytes queued but not yet written.
size_t
bench_zarr_pending_bytes(struct bench_zarr_handle* z);

// Close and free resources. Safe to call on a zero-initialized handle.
void
bench_zarr_close(struct bench_zarr_handle* z);
