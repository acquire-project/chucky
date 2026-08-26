#include "test_zarr_helpers.h"
#include "defs.limits.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <string.h>

// --- Intermediate group callback ---

struct intermediate_ctx
{
  struct store* store;
};

static int
write_intermediate(const char* partial, void* ctx)
{
  struct intermediate_ctx* c = (struct intermediate_ctx*)ctx;
  c->store->mkdirs(c->store, partial);
  struct zarr_group* g = zarr_group_create(c->store, partial);
  if (!g)
    return 1;
  zarr_group_destroy(g);
  return 0;
}

static int
write_root_and_intermediates(struct store* store, const char* array_name)
{
  struct zarr_group* root = zarr_group_create(store, "");
  CHECK(Fail, root);
  zarr_group_destroy(root);
  if (array_name && array_name[0]) {
    struct intermediate_ctx ictx = { .store = store };
    CHECK(Fail,
          zarr_for_each_intermediate(array_name, write_intermediate, &ictx) ==
            0);
    CHECK(Fail, store->mkdirs(store, array_name) == 0);
  }
  return 0;
Fail:
  return 1;
}

// --- Single array ---

int
test_zarr_sink_open(struct test_zarr_sink* z,
                    const char* store_path,
                    const char* array_name,
                    const struct dimension* dims,
                    uint8_t rank,
                    enum dtype data_type,
                    double fill_value,
                    struct codec_config codec,
                    int unbuffered)
{
  return test_zarr_sink_open_in_store(z,
                                      store_fs_create(store_path, unbuffered),
                                      array_name,
                                      dims,
                                      rank,
                                      data_type,
                                      fill_value,
                                      codec);
}

int
test_zarr_sink_open_in_store(struct test_zarr_sink* z,
                             struct store* store,
                             const char* array_name,
                             const struct dimension* dims,
                             uint8_t rank,
                             enum dtype data_type,
                             double fill_value,
                             struct codec_config codec)
{
  *z = (struct test_zarr_sink){ 0 };

  CHECK(Fail, store);
  z->store = store;
  z->store->mkdirs(z->store, ".");

  CHECK(Fail, write_root_and_intermediates(z->store, array_name) == 0);

  struct zarr_array_config acfg = {
    .data_type = data_type,
    .fill_value = fill_value,
    .rank = rank,
    .dimensions = dims,
    .codec = codec,
  };
  z->array = zarr_array_create(z->store, array_name ? array_name : "", &acfg);
  CHECK(Fail, z->array);
  return 0;

Fail:
  test_zarr_sink_close(z);
  return 1;
}

int
test_zarr_sink_open_with_pool(struct test_zarr_sink* z,
                              struct store* store,
                              const char* array_name,
                              const struct dimension* dims,
                              uint8_t rank,
                              enum dtype data_type,
                              struct codec_config codec)
{
  *z = (struct test_zarr_sink){ 0 };
  CHECK(Fail, store);
  z->store = store;

  CHECK(Fail, array_name && array_name[0]);
  CHECK(Fail, dims && rank > 0 && rank <= MAX_ZARR_RANK);
  CHECK(Fail, store->mkdirs(store, array_name) == 0);

  uint64_t shard_counts[MAX_ZARR_RANK], chunks_per_shard[MAX_ZARR_RANK];
  const uint64_t nslots =
    dims_compute_shard_geometry(dims, rank, shard_counts, chunks_per_shard);
  CHECK(Fail, nslots > 0);

  z->pool = store->create_pool(store, nslots);
  CHECK(Fail, z->pool);

  struct zarr_array_config acfg = {
    .data_type = data_type,
    .fill_value = 0,
    .rank = rank,
    .dimensions = dims,
    .codec = codec,
  };
  z->array = zarr_array_create_with_pool(store, z->pool, 0, array_name, &acfg);
  CHECK(Fail, z->array);
  return 0;

Fail:
  test_zarr_sink_close(z);
  return 1;
}

struct shard_sink*
test_zarr_sink_as_shard_sink(struct test_zarr_sink* z)
{
  return zarr_array_as_shard_sink(z->array);
}

int
test_zarr_sink_has_error(const struct test_zarr_sink* z)
{
  return zarr_array_has_error(z->array);
}

int
test_zarr_sink_flush(struct test_zarr_sink* z)
{
  return zarr_array_flush(z->array);
}

void
test_zarr_sink_close(struct test_zarr_sink* z)
{
  zarr_array_destroy(z->array);
  shard_pool_destroy(z->pool);
  store_destroy(z->store);
  *z = (struct test_zarr_sink){ 0 };
}

// --- Multiscale ---

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
                          int unbuffered)
{
  return test_zarr_multiscale_open_in_store(
    z,
    store_fs_create(store_path, unbuffered),
    array_name,
    dims,
    rank,
    data_type,
    nlod,
    codec,
    axes);
}

int
test_zarr_multiscale_open_in_store(struct test_zarr_multiscale* z,
                                   struct store* store,
                                   const char* array_name,
                                   const struct dimension* dims,
                                   uint8_t rank,
                                   enum dtype data_type,
                                   int nlod,
                                   struct codec_config codec,
                                   const struct ngff_axis* axes)
{
  *z = (struct test_zarr_multiscale){ 0 };

  CHECK(Fail, store);
  z->store = store;
  z->store->mkdirs(z->store, ".");

  CHECK(Fail, write_root_and_intermediates(z->store, array_name) == 0);

  struct ngff_multiscale_config mscfg = {
    .data_type = data_type,
    .rank = rank,
    .dimensions = dims,
    .nlod = nlod,
    .codec = codec,
    .axes = axes,
  };
  z->ms =
    ngff_multiscale_create(z->store, array_name ? array_name : "", &mscfg);
  CHECK(Fail, z->ms);
  return 0;

Fail:
  test_zarr_multiscale_close(z);
  return 1;
}

struct shard_sink*
test_zarr_multiscale_as_shard_sink(struct test_zarr_multiscale* z)
{
  return ngff_multiscale_as_shard_sink(z->ms);
}

int
test_zarr_multiscale_flush(struct test_zarr_multiscale* z)
{
  return ngff_multiscale_flush(z->ms);
}

void
test_zarr_multiscale_close(struct test_zarr_multiscale* z)
{
  ngff_multiscale_destroy(z->ms);
  store_destroy(z->store);
  *z = (struct test_zarr_multiscale){ 0 };
}
