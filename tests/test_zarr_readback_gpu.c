// Write GPU Blosc stores and validate them through zarr-python/numcodecs.

#include "gpu/prelude.cuda.h"
#include "stream.gpu.h"
#include "test_platform.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NT 4
#define NY 256
#define NX 256

static int
write_zarr(const char* store_path, struct codec_config codec)
{
  const int total = NT * NY * NX;
  uint8_t* allocation = (uint8_t*)malloc((size_t)total * sizeof(uint16_t) + 1);
  CHECK(Fail, allocation);
  // Caller buffers need not be aligned.
  uint8_t* src = allocation + 1;
  for (int i = 0; i < total; ++i) {
    const uint16_t value = (uint16_t)i;
    memcpy(src + (size_t)i * sizeof(value), &value, sizeof(value));
  }

  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 0, NY, NX });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 128, 128 });
  dims[0].chunks_per_shard = NT;
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });

  struct test_zarr_sink zs = { 0 };
  struct tile_stream_gpu* stream = NULL;
  CHECK(FailSrc,
        test_zarr_sink_open(
          &zs, store_path, "0", dims, 3, dtype_u16, 0, codec, 0) == 0);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
    .epochs_per_batch = 1,
  };
  stream = tile_stream_gpu_create(&config, test_zarr_sink_as_shard_sink(&zs));
  CHECK(FailSink, stream);

  struct slice input = { .beg = src, .end = src + total * sizeof(uint16_t) };
  CHECK(FailStream,
        writer_append(tile_stream_gpu_writer(stream), input).error == 0);
  CHECK(FailStream, writer_flush(tile_stream_gpu_writer(stream)).error == 0);
  CHECK(FailStream, test_zarr_sink_flush(&zs) == 0);

  tile_stream_gpu_destroy(stream);
  test_zarr_sink_close(&zs);
  free(allocation);
  return 0;

FailStream:
  tile_stream_gpu_destroy(stream);
FailSink:
  test_zarr_sink_close(&zs);
FailSrc:
  free(allocation);
Fail:
  return 1;
}

int
main(void)
{
  if (system("uv --version > " NULL_DEV " 2>&1") != 0)
    return 77;

  CUcontext context = 0;
  CUdevice device;
  CU(RunFail, cuInit(0));
  CU(RunFail, cuDeviceGet(&device, 0));
  CU(RunFail, cu_ctx_create(&context, 0, device));

  char tmpdir[256];
  CHECK(RunFail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);

  const struct
  {
    const char* name;
    struct codec_config codec;
  } cases[] = {
    { "blosc_lz4_noshuffle",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_NONE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_lz4_shuffle",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_lz4_bitshuffle",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BIT,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd_noshuffle",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_NONE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd_shuffle",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_zstd_bitshuffle",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BIT,
        .blosc_block_bytes = 16 * 1024 } },
    { "blosc_lz4_unaligned_blocks",
      { .id = CODEC_BLOSC_LZ4,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BYTE,
        .blosc_block_bytes = 4097 } },
    { "blosc_zstd_unaligned_blocks",
      { .id = CODEC_BLOSC_ZSTD,
        .level = 5,
        .shuffle = CODEC_SHUFFLE_BIT,
        .blosc_block_bytes = 4097 } },
  };

  int error = 0;
  for (size_t i = 0; i < countof(cases); ++i) {
    char store[512];
    snprintf(store, sizeof(store), "%s/%s", tmpdir, cases[i].name);
    if (test_mkdir(store) != 0) {
      error = 1;
      goto Cleanup;
    }
    log_info("Writing GPU %s ...", cases[i].name);
    if (write_zarr(store, cases[i].codec) != 0) {
      error = 1;
      goto Cleanup;
    }
  }

  {
    char command[1024];
    snprintf(command,
             sizeof(command),
             "uv run \"" SOURCE_DIR "/tests/validate_zarr.py\" \"%s\" %d %d %d",
             tmpdir,
             NT,
             NY,
             NX);
    if (system(command) != 0) {
      log_error("Python validation failed; output preserved at %s", tmpdir);
      error = 1;
    }
  }

Cleanup:
  if (!error)
    test_tmpdir_remove(tmpdir);
  cuCtxDestroy(context);
  return error;

RunFail:
  cuCtxDestroy(context);
  return 1;
}
