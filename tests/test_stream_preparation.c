#include "platform/platform_io.h"
#include "test_platform.h"
#include "test_shard_verify.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"
#include "zarr/zarr_array.h"

#ifdef TEST_PREPARATION_GPU
#include "gpu/prelude.cuda.h"
#include "multiarray.gpu.h"
#include "stream.gpu.h"
#define stream_type tile_stream_gpu
#define stream_create tile_stream_gpu_create
#define stream_writer tile_stream_gpu_writer
#define stream_destroy tile_stream_gpu_destroy
#define stream_reset tile_stream_gpu_reset_metrics
#define multiarray_type multiarray_tile_stream_gpu
#define multiarray_create multiarray_tile_stream_gpu_create
#define multiarray_writer multiarray_tile_stream_gpu_writer
#define multiarray_destroy multiarray_tile_stream_gpu_destroy
#else
#include "multiarray.cpu.h"
#include "stream.cpu.h"
#define stream_type tile_stream_cpu
#define stream_create tile_stream_cpu_create
#define stream_writer tile_stream_cpu_writer
#define stream_destroy tile_stream_cpu_destroy
#define stream_reset tile_stream_cpu_reset_metrics
#define multiarray_type multiarray_tile_stream_cpu
#define multiarray_create multiarray_tile_stream_cpu_create
#define multiarray_writer multiarray_tile_stream_cpu_writer
#define multiarray_destroy multiarray_tile_stream_cpu_destroy
#endif

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
verify_shard(const char* path, int first, int frames)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return 1;
  uint8_t bytes[16384];
  size_t len = fread(bytes, 1, sizeof(bytes), f);
  int end = fgetc(f);
  fclose(f);
  uint64_t offset[2], size[2];
  if (end != EOF || shard_index_parse(bytes, len, 2, offset, size))
    return 1;
  for (int t = 0; t < 2; ++t) {
    if (first + t >= frames) {
      if (size[t] != UINT64_MAX)
        return 1;
      continue;
    }
    if (size[t] != 32 || offset[t] > len || size[t] > len - offset[t])
      return 1;
    for (int i = 0; i < 16; ++i) {
      uint16_t value;
      memcpy(&value, bytes + offset[t] + (size_t)i * 2, 2);
      if (value != (uint16_t)(100 + first + t))
        return 1;
    }
  }
  return 0;
}

static int
test_stream(int frames, int bounded, int destroy_only, int disabled)
{
  char root[256] = { 0 }, path[512];
  struct test_zarr_sink z = { 0 };
  struct stream_type* stream = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  struct dimension dims[] = {
    { .size = bounded ? 4 : 0,
      .chunk_size = 1,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 1 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 2 },
  };
  const struct codec_config codec = { .id = CODEC_NONE };
  CHECK(Done,
        test_zarr_sink_open_with_pool(
          &z, store_fs_create(root, 1), "0", dims, 3, dtype_u16, codec) == 0);
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 128,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
    .max_threads = 2,
    .max_nlod = 1,
    .epochs_per_batch = 1,
    .disable_shard_preparation = disabled,
  };
  stream = stream_create(&config, test_zarr_sink_as_shard_sink(&z));
  CHECK(Done, stream);
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x) {
      snprintf(path, sizeof(path), "%s/0/c/0/%d/%d", root, y, x);
      CHECK(Done, platform_path_exists(path) == !disabled);
    }
  struct writer* w = stream_writer(stream);
  for (int t = 0; t < frames; ++t) {
    uint16_t frame[64];
    for (int i = 0; i < 64; ++i)
      frame[i] = (uint16_t)(100 + t);
    CHECK(Done,
          writer_append_wait(w, (struct slice){ frame, frame + 64 }).error ==
            0);
  }
  if (frames && !destroy_only) {
    CHECK(Done, stream_reset(stream) == 0);
    for (uint64_t slot = 0; slot < 4; ++slot)
      CHECK(Done, z.pool->wait_prepared(z.pool, slot) == 0);
    const int next = (frames - 1) / 2 + 1;
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        snprintf(path, sizeof(path), "%s/0/c/%d/%d/%d", root, next, y, x);
        CHECK(Done,
              platform_path_exists(path) ==
                (!disabled && (!bounded || next < 2)));
        snprintf(path, sizeof(path), "%s/0/c/%d/%d/%d", root, next + 1, y, x);
        CHECK(Done, platform_path_exists(path) == 0);
      }
  }
  if (!destroy_only) {
    CHECK(Done, writer_flush(w).error == 0);
    CHECK(Done, writer_close(w).error == 0);
    CHECK(Done, writer_flush(w).error == 0 && writer_close(w).error == 0);
  }
  stream_destroy(stream);
  stream = NULL;
  CHECK(Done, test_zarr_sink_has_error(&z) == 0);
  for (int g = 0; g < (frames + 1) / 2; ++g)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        snprintf(path, sizeof(path), "%s/0/c/%d/%d/%d", root, g, y, x);
        CHECK(Done, verify_shard(path, g * 2, frames) == 0);
      }
  for (int g = (frames + 1) / 2; g < 5; ++g) {
    snprintf(path, sizeof(path), "%s/0/c/%d", root, g);
    CHECK(Done, platform_path_exists(path) == 0);
  }
  if (!frames) {
    snprintf(path, sizeof(path), "%s/0/c", root);
    CHECK(Done, platform_path_exists(path) == 0);
  }
  error = 0;
Done:
  stream_destroy(stream);
  test_zarr_sink_close(&z);
  test_tmpdir_remove(root);
  return error;
}

struct resize_count
{
  struct io_backend inner;
  _Atomic int resizes;
  _Atomic int opens;
};

static void
count_resizes(void* ctx, const struct io_request* request)
{
  struct resize_count* c = ctx;
  if (request->op == IO_OP_TRUNCATE)
    atomic_fetch_add(&c->resizes, 1);
  if (request->op == IO_OP_OPEN)
    atomic_fetch_add(&c->opens, 1);
  c->inner.execute(c->inner.ctx, request);
}

static struct io_backend
wrap_resizes(void* ctx, struct io_backend inner)
{
  struct resize_count* c = ctx;
  c->inner = inner;
  return (struct io_backend){ .ctx = c, .execute = count_resizes };
}

static int
test_prepared_files_need_no_resize(enum compression_codec codec_id)
{
  char root[256] = { 0 };
  struct test_zarr_sink z = { 0 };
  struct stream_type* stream = NULL;
  struct resize_count counts = { 0 };
  uint16_t* frame = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 128,
      .chunk_size = 64,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 1 },
    { .size = 256,
      .chunk_size = 128,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 2 },
  };
  struct codec_config codec = {
    .id = codec_id,
    .level = 3,
    .shuffle = CODEC_SHUFFLE_BIT,
    .blosc_block_bytes = 16384,
  };
  z.store = store_fs_create(root, 1);
  CHECK(Done, z.store && z.store->mkdirs(z.store, "0") == 0);
  z.pool = shard_pool_fs_create_wrapped(
    root,
    4,
    1,
    NULL,
    (struct shard_pool_fs_wrapper){ .ctx = &counts, .wrap = wrap_resizes });
  CHECK(Done, z.pool);
  struct zarr_array_config array_config = {
    .data_type = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
  };
  z.array = zarr_array_create_with_pool(z.store, z.pool, 0, "0", &array_config);
  CHECK(Done, z.array);
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 65536,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
    .max_threads = 2,
    .max_nlod = 1,
    .epochs_per_batch = 1,
  };
  stream = stream_create(&config, test_zarr_sink_as_shard_sink(&z));
  CHECK(Done, stream);
  frame = calloc(32768, sizeof(*frame));
  CHECK(Done, frame);
  struct writer* w = stream_writer(stream);
  for (int t = 0; t < 2; ++t)
    CHECK(Done,
          writer_append_wait(w, (struct slice){ frame, frame + 32768 }).error ==
            0);
  CHECK(Done, writer_flush(w).error == 0);
  CHECK(Done, writer_close(w).error == 0);
  CHECK(Done, atomic_load(&counts.opens) == 0);
  CHECK(Done, atomic_load(&counts.resizes) == 4);
  error = 0;
Done:
  stream_destroy(stream);
  test_zarr_sink_close(&z);
  free(frame);
  test_tmpdir_remove(root);
  return error;
}

struct preparation_probe
{
  struct shard_sink base;
  int prepares;
  int stops;
  int cancels;
  int fail_prepare;
  int fail_cancel;
};

static int
probe_prepare(struct shard_sink* self, uint8_t level, uint64_t capacity)
{
  (void)level;
  (void)capacity;
  struct preparation_probe* p = (struct preparation_probe*)self;
  ++p->prepares;
  return p->fail_prepare;
}

static void
probe_stop(struct shard_sink* self)
{
  ++((struct preparation_probe*)self)->stops;
}

static int
probe_cancel(struct shard_sink* self)
{
  struct preparation_probe* p = (struct preparation_probe*)self;
  ++p->cancels;
  return p->fail_cancel;
}

static struct preparation_probe
probe_init(int hooks)
{
  return (struct preparation_probe){ .base = {
                                       .prepare_shards =
                                         hooks & 1 ? probe_prepare : NULL,
                                       .stop_preparing =
                                         hooks & 2 ? probe_stop : NULL,
                                       .cancel_prepared =
                                         hooks & 4 ? probe_cancel : NULL,
                                     } };
}

static struct tile_stream_configuration
probe_config(struct dimension dims[2], int disabled)
{
  dims_create(dims, "ty", (uint64_t[]){ 4, 4 });
  dims_set_chunk_sizes(dims, 2, (uint64_t[]){ 1, 4 });
  dims_set_shard_counts(dims, 2, (uint64_t[]){ 2, 1 });
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 128,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_threads = 2,
    .max_nlod = 1,
    .disable_shard_preparation = disabled,
  };
}

static int
test_preparation_policy(int hooks, int disabled)
{
  struct dimension dims[2];
  struct tile_stream_configuration config = probe_config(dims, disabled);
  struct preparation_probe single = probe_init(hooks);
  struct preparation_probe multi = probe_init(hooks);
  struct stream_type* stream = NULL;
  struct multiarray_type* arrays = NULL;
  struct shard_sink* sinks[] = { &multi.base };
  const int enabled = hooks == 7 && !disabled;
  int error = 1;
  stream = stream_create(&config, &single.base);
  CHECK(Done, stream && single.prepares == enabled);
  arrays = multiarray_create(1, &config, sinks, 0);
  CHECK(Done, arrays && multi.prepares == enabled);
  CHECK(Done, writer_flush(stream_writer(stream)).error == 0);
  CHECK(Done,
        multiarray_writer(arrays)->flush(multiarray_writer(arrays)).error == 0);
  stream_destroy(stream);
  stream = NULL;
  multiarray_destroy(arrays);
  arrays = NULL;
  CHECK(Done, (single.stops > 0) == enabled && (single.cancels > 0) == enabled);
  CHECK(Done, (multi.stops > 0) == enabled && (multi.cancels > 0) == enabled);
  error = 0;
Done:
  stream_destroy(stream);
  multiarray_destroy(arrays);
  return error;
}

static int
test_multiarray_cleanup_failure(int destroy_only)
{
  struct dimension dims[2];
  struct tile_stream_configuration config = probe_config(dims, 0);
  struct tile_stream_configuration configs[] = { config, config };
  struct preparation_probe probes[] = { probe_init(7), probe_init(7) };
  probes[0].fail_cancel = 1;
  struct shard_sink* sinks[] = { &probes[0].base, &probes[1].base };
  struct multiarray_type* arrays = multiarray_create(2, configs, sinks, 0);
  int error = 1;
  CHECK(Done, arrays);
  if (!destroy_only) {
    CHECK(Done,
          multiarray_writer(arrays)->flush(multiarray_writer(arrays)).error !=
            0);
    CHECK(Done, probes[0].cancels > 0 && probes[1].cancels > 0);
  }
  multiarray_destroy(arrays);
  arrays = NULL;
  CHECK(Done, probes[0].cancels > 0 && probes[1].cancels > 0);
  error = 0;
Done:
  multiarray_destroy(arrays);
  return error;
}

static int
test_multiarray_preparation_rollback(void)
{
  struct dimension dims[2];
  struct tile_stream_configuration config = probe_config(dims, 0);
  struct tile_stream_configuration configs[] = { config, config };
  struct preparation_probe probes[] = { probe_init(7), probe_init(7) };
  probes[1].fail_prepare = 1;
  struct shard_sink* sinks[] = { &probes[0].base, &probes[1].base };
  struct multiarray_type* arrays = multiarray_create(2, configs, sinks, 0);
  const int failed = arrays || !probes[0].cancels || !probes[1].cancels;
  multiarray_destroy(arrays);
  return failed;
}

int
main(void)
{
#ifdef TEST_PREPARATION_GPU
  CUcontext context = NULL;
  CUdevice device;
  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&device, 0));
  CU(Fail, cu_ctx_create(&context, 0, device));
#endif
  int error = test_prepared_files_need_no_resize(CODEC_NONE);
  error |= test_prepared_files_need_no_resize(CODEC_BLOSC_LZ4);
  for (int disabled = 0; disabled < 2; ++disabled) {
    for (int hooks = 0; hooks < 8; ++hooks)
      error |= test_preparation_policy(hooks, disabled);
    for (int destroy_only = 0; destroy_only < 2; ++destroy_only) {
      error |= test_stream(0, 0, destroy_only, disabled);
      error |= test_stream(1, 0, destroy_only, disabled);
      error |= test_stream(4, 0, destroy_only, disabled);
      error |= test_stream(5, 0, destroy_only, disabled);
      error |= test_stream(4, 1, destroy_only, disabled);
    }
  }
  error |= test_multiarray_cleanup_failure(0);
  error |= test_multiarray_cleanup_failure(1);
  error |= test_multiarray_preparation_rollback();
#ifdef TEST_PREPARATION_GPU
  cuCtxDestroy(context);
#endif
  return error;
#ifdef TEST_PREPARATION_GPU
Fail:
  if (context)
    cuCtxDestroy(context);
  return 1;
#endif
}
