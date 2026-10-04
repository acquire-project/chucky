// Write the same NGFF fixture with either backend, then read it independently.
#include "test_ngff_fixture.h"
#include "test_platform.h"
#include "test_readback_codecs.h"
#include "test_zarr_helpers.h"
#include "util/prelude.h"
#include "writer.buffered.h"

#ifdef TEST_NGFF_READBACK_GPU
#include "gpu/prelude.cuda.h"
#include "stream.gpu.h"
#define stream_type tile_stream_gpu
#define stream_create tile_stream_gpu_create
#define stream_writer tile_stream_gpu_writer
#define stream_destroy tile_stream_gpu_destroy
#else
#include "stream.cpu.h"
#define stream_type tile_stream_cpu
#define stream_create tile_stream_cpu_create
#define stream_writer tile_stream_cpu_writer
#define stream_destroy tile_stream_cpu_destroy
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NT NGFF_READBACK_NT
#define NY NGFF_READBACK_NY
#define NX NGFF_READBACK_NX
#define NLEVELS NGFF_READBACK_LEVELS

static int
write_pyramid(const char* path, struct codec_config codec, int buffered)
{
  int err = 1;
  const size_t frame_bytes = NY * NX * sizeof(uint16_t);
  const size_t total_bytes = (NT + (buffered == 2)) * frame_bytes;
  // Preserve the GPU readback coverage for unaligned caller buffers.
  uint8_t* allocation = malloc(total_bytes + 1);
  struct test_zarr_multiscale sink = { 0 };
  struct stream_type* stream = NULL;
  struct buffered_writer* adapter = NULL;
  CHECK(Done, allocation);
  uint8_t* src = allocation + 1;
  ngff_readback_fill(src, 0, NT + (buffered == 2));

  struct dimension dims[3];
  // Two frames per shard: two complete shards and a final partial shard.
  ngff_readback_dimensions(dims, buffered == 2 ? NT : 0, 2, 2);
  CHECK(Done,
        test_zarr_multiscale_open(&sink,
                                  path,
                                  "pyramid",
                                  dims,
                                  3,
                                  dtype_u16,
                                  NLEVELS,
                                  codec,
                                  ngff_readback_axes,
                                  0) == 0);
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = frame_bytes,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
    .reduce_method = lod_reduce_mean,
    .max_nlod = NLEVELS,
    .epochs_per_batch = 1,
    .metadata_update_interval_s = 0,
    .max_threads = 2,
  };
  stream = stream_create(&config, test_zarr_multiscale_as_shard_sink(&sink));
  CHECK(Done, stream);
  struct writer* writer = stream_writer(stream);
  if (buffered) {
    struct buffered_writer_config buffering = { 2 * 1023 * 3, 2 * 997 };
    // Accept an extra frame so the bounded stream's unconsumed suffix is
    // deterministic, just as in the single-array readback tests.
    if (buffered == 2)
      buffering = (struct buffered_writer_config){ total_bytes, 0 };
    adapter = buffered_writer_create(writer, &buffering);
    CHECK(Done, adapter);
    writer = buffered_writer_as_writer(adapter);
  }
  if (buffered == 2) {
    CHECK(Done,
          writer_append_wait(writer, (struct slice){ src, src + total_bytes })
              .error == 0);
    memset(src, 0xFF, total_bytes);
  } else {
    for (int t = 0; t < NT; ++t) {
      uint8_t* frame = src + t * frame_bytes;
      CHECK(
        Done,
        writer_append_wait(writer, (struct slice){ frame, frame + frame_bytes })
            .error == 0);
      memset(frame, 0xFF, frame_bytes);
    }
  }
  CHECK(Done, writer_flush(writer).error == (buffered == 2));
  CHECK(Done, writer_close(writer).error == (buffered == 2));
  CHECK(Done, ngff_multiscale_flush(sink.ms) == 0);
  if (buffered == 2) {
    const struct buffered_writer_stats stats =
      buffered_writer_get_stats(adapter);
    CHECK(Done, stats.downstream_finished && stats.failed);
    CHECK(Done, stats.forwarded_bytes == NT * frame_bytes);
    CHECK(Done, stats.abandoned_bytes == frame_bytes);
  }
  err = 0;
Done:
  buffered_writer_destroy(adapter);
  stream_destroy(stream);
  test_zarr_multiscale_close(&sink);
  free(allocation);
  return err;
}

int
main(void)
{
  // Reader dependencies are required; missing uv must fail CI, not skip it.
  if (system("uv --version > " NULL_DEV " 2>&1") != 0) {
    log_error("uv is required for TensorStore/NGFF readback");
    return 1;
  }
  int err = 1;
  char tmpdir[256] = { 0 };
#ifdef TEST_NGFF_READBACK_GPU
  CUcontext context = 0;
  CUdevice device;
  CU(Done, cuInit(0));
  CU(Done, cuDeviceGet(&device, 0));
  CU(Done, cu_ctx_create(&context, 0, device));
#endif
  CHECK(Done, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  size_t n_codecs, written = 0;
  const struct test_readback_codec* codecs = test_readback_codecs(&n_codecs);
  for (int buffered = 0; buffered <= 2; ++buffered) {
    for (size_t i = 0; i < n_codecs; ++i) {
      char path[512];
      snprintf(path,
               sizeof(path),
               "%s/%s%s",
               tmpdir,
               codecs[i].name,
               buffered == 2 ? "_buffered_limit"
               : buffered    ? "_buffered"
                             : "");
      CHECK(Done, test_mkdir(path) == 0);
      log_info("Writing NGFF %s", path);
      CHECK(Done, write_pyramid(path, codecs[i].codec, buffered) == 0);
      ++written;
    }
  }
  char command[2048];
  snprintf(command,
           sizeof(command),
           "uv run \"" SOURCE_DIR "/tests/validate_ngff_readback.py\""
           " \"%s\" %d %d %d %zu",
           tmpdir,
           NT,
           NY,
           NX,
           written);
  CHECK(Done, system(command) == 0);
  err = 0;
Done:
  if (tmpdir[0]) {
    if (err)
      log_error("Readback failed; output preserved at %s", tmpdir);
    else
      test_tmpdir_remove(tmpdir);
  }
#ifdef TEST_NGFF_READBACK_GPU
  if (context)
    cuCtxDestroy(context);
#endif
  return err;
}
