#include "test_zarr_helpers.h"
#include "defs.limits.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/store.h"
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
  *z = (struct test_zarr_sink){ 0 };

  z->store = store_fs_create(store_path, unbuffered);
  CHECK(Fail, z->store);
  z->store->mkdirs(z->store, ".");

  CHECK(Fail_store, write_root_and_intermediates(z->store, array_name) == 0);

  struct zarr_array_config acfg = {
    .data_type = data_type,
    .fill_value = fill_value,
    .rank = rank,
    .dimensions = dims,
    .codec = codec,
  };
  z->array = zarr_array_create(z->store, array_name ? array_name : "", &acfg);
  CHECK(Fail_store, z->array);
  return 0;

Fail_store:
  store_destroy(z->store);
  z->store = NULL;
Fail:
  return 1;
}

struct shard_sink*
test_zarr_sink_as_shard_sink(struct test_zarr_sink* z)
{
  return zarr_array_as_shard_sink(z->array);
}

void
test_zarr_sink_flush(struct test_zarr_sink* z)
{
  zarr_array_flush(z->array);
}

void
test_zarr_sink_close(struct test_zarr_sink* z)
{
  zarr_array_destroy(z->array);
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
                          double fill_value,
                          int nlod,
                          struct codec_config codec,
                          const struct ngff_axis* axes,
                          int unbuffered)
{
  *z = (struct test_zarr_multiscale){ 0 };

  z->store = store_fs_create(store_path, unbuffered);
  CHECK(Fail, z->store);
  z->store->mkdirs(z->store, ".");

  CHECK(Fail_store, write_root_and_intermediates(z->store, array_name) == 0);

  struct ngff_multiscale_config mscfg = {
    .data_type = data_type,
    .fill_value = fill_value,
    .rank = rank,
    .dimensions = dims,
    .nlod = nlod,
    .codec = codec,
    .axes = axes,
  };
  z->ms =
    ngff_multiscale_create(z->store, array_name ? array_name : "", &mscfg);
  CHECK(Fail_store, z->ms);
  return 0;

Fail_store:
  store_destroy(z->store);
  z->store = NULL;
Fail:
  return 1;
}

struct shard_sink*
test_zarr_multiscale_as_shard_sink(struct test_zarr_multiscale* z)
{
  return ngff_multiscale_as_shard_sink(z->ms);
}

void
test_zarr_multiscale_flush(struct test_zarr_multiscale* z)
{
  ngff_multiscale_flush(z->ms);
}

void
test_zarr_multiscale_close(struct test_zarr_multiscale* z)
{
  ngff_multiscale_destroy(z->ms);
  store_destroy(z->store);
  *z = (struct test_zarr_multiscale){ 0 };
}
