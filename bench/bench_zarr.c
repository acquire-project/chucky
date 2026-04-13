#include "bench_zarr.h"
#include "util/prelude.h"
#include "zarr/store.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>

static int
write_intermediate_bench(const char* partial, void* ctx)
{
  struct store* store = (struct store*)ctx;
  store->mkdirs(store, partial);
  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", partial);
  return zarr_write_group(store, key, NULL);
}

int
bench_zarr_open_fs(struct bench_zarr_handle* z,
                   const char* store_path,
                   const char* array_name,
                   const struct dimension* dims,
                   uint8_t rank,
                   enum dtype data_type,
                   double fill_value,
                   struct codec_config codec,
                   int is_multiscale)
{
  *z = (struct bench_zarr_handle){ 0 };

  z->store = store_fs_create(store_path, 1 /* unbuffered */);
  CHECK(Fail, z->store);
  z->store->mkdirs(z->store, ".");

  // Write root group
  CHECK(Fail_store, zarr_write_group(z->store, "zarr.json", NULL) == 0);

  // Write intermediate groups
  if (array_name && array_name[0]) {
    CHECK(Fail_store,
          zarr_for_each_intermediate(
            array_name, write_intermediate_bench, z->store) == 0);
    CHECK(Fail_store, z->store->mkdirs(z->store, array_name) == 0);
  }

  if (is_multiscale) {
    struct ngff_multiscale_config mscfg = {
      .data_type = data_type,
      .fill_value = fill_value,
      .rank = rank,
      .dimensions = dims,
      .nlod = 0,
      .codec = codec,
    };
    z->ms =
      ngff_multiscale_create(z->store, array_name ? array_name : "", &mscfg);
    CHECK(Fail_store, z->ms);
  } else {
    struct zarr_array_config acfg = {
      .data_type = data_type,
      .fill_value = fill_value,
      .rank = rank,
      .dimensions = dims,
      .codec = codec,
    };
    z->array = zarr_array_create(z->store, array_name ? array_name : "", &acfg);
    CHECK(Fail_store, z->array);
  }
  return 0;

Fail_store:
  store_destroy(z->store);
  z->store = NULL;
Fail:
  return 1;
}

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
                   int is_multiscale)
{
  *z = (struct bench_zarr_handle){ 0 };

  struct store_s3_config scfg = {
    .bucket = bucket,
    .prefix = prefix,
    .region = region,
    .endpoint = endpoint,
    .throughput_gbps = throughput_gbps,
  };
  store_s3_config_set_defaults(&scfg);

  z->store = store_s3_create(&scfg);
  CHECK(Fail, z->store);
  z->store->mkdirs(z->store, ".");

  // Write root group
  CHECK(Fail_store, zarr_write_group(z->store, "zarr.json", NULL) == 0);

  // Write intermediate groups
  if (array_name && array_name[0]) {
    CHECK(Fail_store,
          zarr_for_each_intermediate(
            array_name, write_intermediate_bench, z->store) == 0);
    CHECK(Fail_store, z->store->mkdirs(z->store, array_name) == 0);
  }

  if (is_multiscale) {
    struct ngff_multiscale_config mscfg = {
      .data_type = data_type,
      .fill_value = fill_value,
      .rank = rank,
      .dimensions = dims,
      .nlod = 0,
      .codec = codec,
    };
    z->ms =
      ngff_multiscale_create(z->store, array_name ? array_name : "", &mscfg);
    CHECK(Fail_store, z->ms);
  } else {
    struct zarr_array_config acfg = {
      .data_type = data_type,
      .fill_value = fill_value,
      .rank = rank,
      .dimensions = dims,
      .codec = codec,
    };
    z->array = zarr_array_create(z->store, array_name ? array_name : "", &acfg);
    CHECK(Fail_store, z->array);
  }
  return 0;

Fail_store:
  store_destroy(z->store);
  z->store = NULL;
Fail:
  return 1;
}

struct shard_sink*
bench_zarr_as_shard_sink(struct bench_zarr_handle* z)
{
  if (z->ms)
    return ngff_multiscale_as_shard_sink(z->ms);
  return zarr_array_as_shard_sink(z->array);
}

int
bench_zarr_flush(struct bench_zarr_handle* z)
{
  int err = 0;
  err |= ngff_multiscale_flush(z->ms);
  err |= zarr_array_flush(z->array);
  return err;
}

size_t
bench_zarr_pending_bytes(struct bench_zarr_handle* z)
{
  if (z->ms)
    return ngff_multiscale_pending_bytes(z->ms);
  return zarr_array_pending_bytes(z->array);
}

void
bench_zarr_close(struct bench_zarr_handle* z)
{
  ngff_multiscale_destroy(z->ms);
  zarr_array_destroy(z->array);
  store_destroy(z->store);
  *z = (struct bench_zarr_handle){ 0 };
}
