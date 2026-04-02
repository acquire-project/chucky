// Test that zarr output is readable by zarr-python.
// Writes zarr stores with various codecs, then runs a Python validation script.

#include "cpu/compress_blosc.h"
#include "dimension.h"
#include "stream.cpu.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr_fs_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NT 4
#define NY 8
#define NX 12

static int
write_zarr(const char* store_path, struct codec_config codec)
{
  if (codec_is_blosc(codec.id) && compress_blosc_validate(codec))
    return 2;

  const int total = NT * NY * NX;
  uint16_t* src = (uint16_t*)malloc((size_t)total * sizeof(uint16_t));
  CHECK(Fail, src);
  for (int i = 0; i < total; ++i)
    src[i] = (uint16_t)(i & 0xFFFF);

  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 0, NY, NX });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 4, 6 });
  dims[0].chunks_per_shard = NT; // unbounded dim needs explicit cps
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });

  struct zarr_config zcfg = {
    .store_path = store_path,
    .array_name = "0",
    .data_type = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail_src, zs);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
  };

  struct tile_stream_cpu* s =
    tile_stream_cpu_create(&config, zarr_fs_sink_as_shard_sink(zs));
  CHECK(Fail_sink, s);

  struct slice input = { .beg = src, .end = src + total };
  struct writer_result r = writer_append(tile_stream_cpu_writer(s), input);
  CHECK(Fail_stream, r.error == 0);
  r = writer_flush(tile_stream_cpu_writer(s));
  CHECK(Fail_stream, r.error == 0);

  zarr_fs_sink_flush(zs);
  tile_stream_cpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  free(src);
  return 0;

Fail_stream:
  tile_stream_cpu_destroy(s);
Fail_sink:
  zarr_fs_sink_destroy(zs);
Fail_src:
  free(src);
Fail:
  return 1;
}

int
main(void)
{
  if (system("uv --version > /dev/null 2>&1") != 0) {
    log_error("uv not found — install it: https://docs.astral.sh/uv/");
    return 1;
  }

  char tmpdir[256];
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);

  struct
  {
    const char* name;
    struct codec_config codec;
  } codecs[] = {
    { "none", { .id = CODEC_NONE } },
    { "zstd", { .id = CODEC_ZSTD } },
    { "blosc_lz4",
      { .id = CODEC_BLOSC_LZ4, .level = 5, .shuffle = CODEC_SHUFFLE_BYTE } },
    { "blosc_zstd",
      { .id = CODEC_BLOSC_ZSTD, .level = 5, .shuffle = CODEC_SHUFFLE_BYTE } },
  };
  int n_codecs = (int)(sizeof(codecs) / sizeof(codecs[0]));

  int err = 0;
  for (int i = 0; i < n_codecs; ++i) {
    char store[512];
    snprintf(store, sizeof(store), "%s/%s", tmpdir, codecs[i].name);
    CHECK(Cleanup, test_mkdir(store) == 0);
    log_info("Writing %s ...", codecs[i].name);
    int wrc = write_zarr(store, codecs[i].codec);
    if (wrc == 2) { // blosc not available
      log_info("  skipped: %s (codec not available)", codecs[i].name);
      test_tmpdir_remove(store);
      continue;
    }
    if (wrc) {
      log_error("  write failed: %s", codecs[i].name);
      err = 1;
      goto Cleanup;
    }
  }

  {
    char cmd[1024];
    snprintf(cmd,
             sizeof(cmd),
             "uv run " SOURCE_DIR "/tests/validate_zarr.py %s %d %d %d",
             tmpdir,
             NT,
             NY,
             NX);
    log_info("Running: %s", cmd);
    if (system(cmd) != 0) {
      log_error("Python validation failed");
      err = 1;
    }
  }

Cleanup:
  if (err)
    log_error("Output preserved at: %s", tmpdir);
  else
    test_tmpdir_remove(tmpdir);
  return err;

Fail:
  return 1;
}
