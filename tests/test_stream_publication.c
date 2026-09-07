// A stream starts empty even when its configured capacity is finite. Array
// creation by itself retains its fixed shape; attaching a stream publishes the
// empty readable extent synchronously without changing that capacity. Later
// publication follows completed IO without blocking append or requiring input.

#include "defs.limits.h"
#include "ngff/ngff_multiscale.h"
#include "platform/platform.h"
#include "test_io_faults.h"
#include "test_platform.h"
#include "test_shard_sink.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"

#ifdef TEST_STREAM_PUBLICATION_GPU
#include "gpu/stream.internal.h"
#include "multiarray.gpu.h"
#include "stream.gpu.h"
#include "test_runner.h"
typedef struct tile_stream_gpu test_stream;
typedef struct multiarray_tile_stream_gpu test_multiarray;
#define stream_create tile_stream_gpu_create
#define stream_destroy tile_stream_gpu_destroy
#define stream_writer tile_stream_gpu_writer
#define multiarray_create multiarray_tile_stream_gpu_create
#define multiarray_destroy multiarray_tile_stream_gpu_destroy
#define get_multiarray_writer multiarray_tile_stream_gpu_writer
#else
#include "multiarray.cpu.h"
#include "stream.cpu.h"
typedef struct tile_stream_cpu test_stream;
typedef struct multiarray_tile_stream_cpu test_multiarray;
#define stream_create tile_stream_cpu_create
#define stream_destroy tile_stream_cpu_destroy
#define stream_writer tile_stream_cpu_writer
#define multiarray_create multiarray_tile_stream_cpu_create
#define multiarray_destroy multiarray_tile_stream_cpu_destroy
#define get_multiarray_writer multiarray_tile_stream_cpu_writer
#endif

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

static int
metadata_has_shape(const char* root, const char* key, const char* shape)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", root, key);
  FILE* f = fopen(path, "rb");
  if (!f)
    return 0;
  char json[16384];
  size_t n = fread(json, 1, sizeof(json) - 1, f);
  int ok = !ferror(f) && feof(f);
  fclose(f);
  json[n] = '\0';
  return ok && strstr(json, shape) != NULL;
}

static void
finite_dims(struct dimension dims[4], uint64_t capacity)
{
  dims_create(dims, "tzyx", (uint64_t[]){ capacity, 2, 4, 4 });
  dims_set_chunk_sizes(dims, 4, (uint64_t[]){ 1, 1, 4, 4 });
  dims[0].chunks_per_shard = 2;
  dims[1].chunks_per_shard = 2;
  dims[2].chunks_per_shard = 1;
  dims[3].chunks_per_shard = 1;
}

static struct tile_stream_configuration
finite_config(struct dimension* dims, uint8_t rank)
{
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 2,
    .metadata_update_interval_s = 3600,
    .max_threads = 2,
  };
}

static int
test_finite_shape_and_capacity(void)
{
  char tmpdir[512] = { 0 };
  struct test_zarr_sink z = { 0 };
  test_stream* s = NULL;
  struct dimension dims[4];
  finite_dims(dims, 4);
  struct tile_stream_configuration cfg = finite_config(dims, 4);
  uint16_t data[4 * 2 * 4 * 4] = { 0 };
  int rc = 1;

  CHECK(Cleanup, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  CHECK(Cleanup,
        test_zarr_sink_open(
          &z, tmpdir, "array", dims, 4, dtype_u16, 0, cfg.codec, 0) == 0);
  CHECK(Cleanup,
        metadata_has_shape(tmpdir, "array/zarr.json", "\"shape\":[4,2,4,4]"));
  s = stream_create(&cfg, test_zarr_sink_as_shard_sink(&z));
  CHECK(Cleanup, s);
  CHECK(Cleanup,
        metadata_has_shape(tmpdir, "array/zarr.json", "\"shape\":[0,2,4,4]"));
  CHECK(Cleanup, dims[0].size == 4 && dims[1].size == 2);

  struct writer* w = stream_writer(s);
  struct slice all = { .beg = data,
                       .end = data + sizeof(data) / sizeof(*data) };
  struct writer_result r = writer_append_wait(w, all);
  CHECK(Cleanup, r.error == 0);
  struct slice extra = { .beg = data, .end = data + 1 };
  r = writer_append(w, extra);
  CHECK(Cleanup, r.error == writer_error_finished && r.rest.beg == extra.beg);
  CHECK(Cleanup, writer_flush(w).error == 0);
  CHECK(Cleanup, writer_close(w).error == 0);
  CHECK(Cleanup,
        metadata_has_shape(tmpdir, "array/zarr.json", "\"shape\":[4,2,4,4]"));
  rc = 0;

Cleanup:
  stream_destroy(s);
  test_zarr_sink_close(&z);
  if (tmpdir[0])
    test_tmpdir_remove(tmpdir);
  return rc;
}

static int
test_multiscale_initial_shape(void)
{
  char tmpdir[512] = { 0 };
  struct test_zarr_multiscale z = { 0 };
  test_stream* s = NULL;
  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 4, 8, 8 });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 4, 4 });
  for (int d = 0; d < 3; ++d) {
    dims[d].chunks_per_shard = 1;
    dims[d].downsample = 1;
  }
  struct tile_stream_configuration cfg = finite_config(dims, 3);
  uint64_t before[LOD_MAX_LEVELS][3] = { 0 };
  int nlod = 0;
  int rc = 1;

  CHECK(Cleanup, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  CHECK(Cleanup,
        test_zarr_multiscale_open(
          &z, tmpdir, "multi", dims, 3, dtype_u16, 0, cfg.codec, NULL, 0) == 0);
  while (nlod < LOD_MAX_LEVELS) {
    struct zarr_array* a = ngff_multiscale_level(z.ms, nlod);
    if (!a)
      break;
    const struct dimension* level = zarr_array_dimensions(a);
    for (int d = 0; d < 3; ++d)
      before[nlod][d] = level[d].size;
    CHECK(Cleanup, before[nlod][0] > 0);
    ++nlod;
  }
  CHECK(Cleanup, nlod > 1);
  s = stream_create(&cfg, test_zarr_multiscale_as_shard_sink(&z));
  CHECK(Cleanup, s);
  for (int lv = 0; lv < nlod; ++lv) {
    const struct dimension* level =
      zarr_array_dimensions(ngff_multiscale_level(z.ms, lv));
    CHECK(Cleanup, level[0].size == 0);
    CHECK(Cleanup,
          level[1].size == before[lv][1] && level[2].size == before[lv][2]);
    char key[64], shape[128];
    snprintf(key, sizeof(key), "multi/%d/zarr.json", lv);
    snprintf(shape,
             sizeof(shape),
             "\"shape\":[0,%llu,%llu]",
             (unsigned long long)before[lv][1],
             (unsigned long long)before[lv][2]);
    CHECK(Cleanup, metadata_has_shape(tmpdir, key, shape));
  }
  CHECK(Cleanup, dims[0].size == 4);
  rc = 0;

Cleanup:
  stream_destroy(s);
  test_zarr_multiscale_close(&z);
  if (tmpdir[0])
    test_tmpdir_remove(tmpdir);
  return rc;
}

static int
test_multiarray_initial_shape(void)
{
  char roots[2][512] = { { 0 } };
  struct test_zarr_sink z[2] = { { 0 } };
  test_multiarray* ms = NULL;
  struct dimension dims[2][4];
  struct tile_stream_configuration cfg[2];
  struct shard_sink* sinks[2];
  uint16_t plane[2 * 4 * 4] = { 0 };
  int rc = 1;
  for (int a = 0; a < 2; ++a) {
    finite_dims(dims[a], (uint64_t)(4 + a));
    cfg[a] = finite_config(dims[a], 4);
    CHECK(Cleanup, test_tmpdir_create(roots[a], sizeof(roots[a])) == 0);
    CHECK(
      Cleanup,
      test_zarr_sink_open(
        &z[a], roots[a], "array", dims[a], 4, dtype_u16, 0, cfg[a].codec, 0) ==
        0);
    sinks[a] = test_zarr_sink_as_shard_sink(&z[a]);
  }
  ms = multiarray_create(2, cfg, sinks, 0);
  CHECK(Cleanup, ms);
  for (int a = 0; a < 2; ++a)
    CHECK(
      Cleanup,
      metadata_has_shape(roots[a], "array/zarr.json", "\"shape\":[0,2,4,4]"));

  struct multiarray_writer* w = get_multiarray_writer(ms);
  struct slice input = { .beg = plane, .end = plane + 2 * 4 * 4 };
  CHECK(Cleanup, w->update(w, 0, input).error == 0);
  CHECK(Cleanup, w->flush(w).error == 0);
  CHECK(Cleanup, w->close(w).error == 0);
  CHECK(Cleanup,
        metadata_has_shape(roots[0], "array/zarr.json", "\"shape\":[1,2,4,4]"));
  CHECK(Cleanup,
        metadata_has_shape(roots[1], "array/zarr.json", "\"shape\":[0,2,4,4]"));
  CHECK(Cleanup, dims[0][0].size == 4 && dims[1][0].size == 5);
  rc = 0;

Cleanup:
  multiarray_destroy(ms);
  for (int a = 0; a < 2; ++a) {
    test_zarr_sink_close(&z[a]);
    if (roots[a][0])
      test_tmpdir_remove(roots[a]);
  }
  return rc;
}

static int
reject_update(struct shard_sink* self,
              uint8_t level,
              uint8_t n_append,
              const uint64_t* sizes)
{
  (void)level;
  (void)n_append;
  struct test_shard_sink* sink =
    container_of(self, struct test_shard_sink, base);
  ++sink->update_append_count;
  sink->last_append_size0 = sizes[0];
  return 1;
}

static int
test_initial_publication_failure(void)
{
  struct test_shard_sink sinks[2];
  for (int a = 0; a < 2; ++a)
    test_sink_init(&sinks[a], 1, 4096);
  sinks[1].base.update_append = reject_update;
  struct dimension dims[4];
  finite_dims(dims, 4);
  struct tile_stream_configuration cfg[2] = {
    finite_config(dims, 4),
    finite_config(dims, 4),
  };
  struct shard_sink* ptrs[2] = { &sinks[0].base, &sinks[1].base };
  test_stream* s = stream_create(&cfg[1], ptrs[1]);
  test_multiarray* ms = NULL;
  int rc = 1;
  CHECK(Cleanup, !s && sinks[1].update_append_count == 1);
  CHECK(Cleanup, sinks[1].last_append_size0 == 0);
  sinks[1].update_append_count = 0;
  ms = multiarray_create(2, cfg, ptrs, 0);
  CHECK(Cleanup, !ms);
  CHECK(Cleanup,
        sinks[0].update_append_count == 1 && sinks[1].update_append_count == 1);
  rc = 0;
Cleanup:
  stream_destroy(s);
  multiarray_destroy(ms);
  for (int a = 0; a < 2; ++a)
    test_sink_free(&sinks[a]);
  return rc;
}

#define PUBLICATION_TIMEOUT_MS 5000

struct append_args
{
  struct writer* w;
  struct slice input;
  _Atomic int done;
  int error;
};

static void
append_thread_fn(void* arg)
{
  struct append_args* a = (struct append_args*)arg;
  a->error = writer_append_wait(a->w, a->input).error;
  atomic_store(&a->done, 1);
}

// A closed generation queues its metadata behind a filesystem fence, without
// parking append. Releasing that fence must publish even if input stops.
static int
check_publication_follows_io(int fail_truncate)
{
  char tmpdir[512] = { 0 };
  struct dimension dims[] = {
    { .size = 8,
      .chunk_size = 1,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 1 },
    { .size = 8,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 2 },
  };
  struct io_faults faults;
  struct test_zarr_sink z = { 0 };
  test_stream* s = NULL;
  test_thread* thr = NULL;
  uint16_t data[2 * 8 * 8] = { 0 };
  _Atomic int gate = 0;
  struct append_args a = { 0 };
  int rc = 1;

  CHECK(Cleanup, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  CHECK(Cleanup,
        test_zarr_sink_open_with_pool(
          &z,
          io_faults_store_create(&faults, tmpdir, 1, NULL),
          "0",
          dims,
          3,
          dtype_u16,
          (struct codec_config){ .id = CODEC_NONE }) == 0);
  struct tile_stream_configuration cfg = finite_config(dims, 3);
  cfg.metadata_update_interval_s = 0;
  s = stream_create(&cfg, test_zarr_sink_as_shard_sink(&z));
  CHECK(Cleanup,
        s && metadata_has_shape(tmpdir, "0/zarr.json", "\"shape\":[0,8,8]"));
  CHECK(Cleanup, io_faults_inject_blocking_job(&faults, &gate) == 0);
  for (int i = 0; i < PUBLICATION_TIMEOUT_MS && atomic_load(&faults.armed); ++i)
    platform_sleep_ns(1000000LL);
  CHECK(Cleanup, atomic_load(&faults.armed) == 0);
  if (fail_truncate)
    io_faults_fail_next_truncate(&faults);

  a.w = stream_writer(s);
  a.input = (struct slice){ .beg = data, .end = data + 2 * 8 * 8 };
  CHECK(Cleanup, test_thread_start(&thr, append_thread_fn, &a) == 0);
  CHECK(Cleanup, test_wait_flag(&a.done, PUBLICATION_TIMEOUT_MS) == 0);
  test_thread_join(thr);
  thr = NULL;
  CHECK(Cleanup, fail_truncate || a.error == 0);
#ifdef TEST_STREAM_PUBLICATION_GPU
  if (!fail_truncate && s->engine.delivery.thread) {
    for (int i = 0; i < PUBLICATION_TIMEOUT_MS &&
                    gpu_delivery_job_state(&s->engine.delivery, 0, NULL) !=
                      DELIVERY_JOB_DONE;
         ++i)
      platform_sleep_ns(1000000LL);
    CHECK(Cleanup,
          gpu_delivery_job_state(&s->engine.delivery, 0, NULL) ==
            DELIVERY_JOB_DONE);
  }
#endif
  CHECK(Cleanup, atomic_load(&gate) == 0);
  CHECK(Cleanup,
        metadata_has_shape(tmpdir, "0/zarr.json", "\"shape\":[0,8,8]"));

  atomic_store(&gate, 1);
  if (!fail_truncate) {
    for (int i = 0;
         i < PUBLICATION_TIMEOUT_MS &&
         !metadata_has_shape(tmpdir, "0/zarr.json", "\"shape\":[2,8,8]");
         ++i)
      platform_sleep_ns(1000000LL);
    CHECK(Cleanup,
          metadata_has_shape(tmpdir, "0/zarr.json", "\"shape\":[2,8,8]"));
  }
  struct writer_result fr = writer_flush(a.w);
  struct writer_result cr = writer_close(a.w);
  CHECK(Cleanup, (fr.error != 0) == fail_truncate);
  CHECK(Cleanup, (cr.error != 0) == fail_truncate);
  CHECK(Cleanup,
        metadata_has_shape(tmpdir,
                           "0/zarr.json",
                           fail_truncate ? "\"shape\":[0,8,8]"
                                         : "\"shape\":[2,8,8]"));
  rc = 0;

Cleanup:
  atomic_store(&gate, 1);
  test_thread_join(thr);
  stream_destroy(s);
  test_zarr_sink_close(&z);
  if (tmpdir[0])
    test_tmpdir_remove(tmpdir);
  return rc;
}

static int
test_publication_follows_io(void)
{
  return check_publication_follows_io(0);
}

static int
test_publication_withholds_failed_io(void)
{
  return check_publication_follows_io(1);
}

#ifdef TEST_STREAM_PUBLICATION_GPU
RUN_GPU_TESTS({ "finite_shape_and_capacity", test_finite_shape_and_capacity },
              { "multiscale_initial_shape", test_multiscale_initial_shape },
              { "multiarray_initial_shape", test_multiarray_initial_shape },
              { "initial_publication_failure",
                test_initial_publication_failure },
              { "publication_follows_io", test_publication_follows_io },
              { "publication_withholds_failed_io",
                test_publication_withholds_failed_io }, )
#else
int
main(void)
{
  return test_finite_shape_and_capacity() | test_multiscale_initial_shape() |
         test_multiarray_initial_shape() | test_initial_publication_failure() |
         test_publication_follows_io() | test_publication_withholds_failed_io();
}
#endif
