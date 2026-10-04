// The Python reader controls appends over stdin; only command 0 closes output.
#include "test_ngff_fixture.h"
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

int
main(int argc, char** argv)
{
  size_t n_codecs;
  const struct test_readback_codec* codecs = test_readback_codecs(&n_codecs);
  if (argc == 2 && strcmp(argv[1], "--list") == 0) {
    for (size_t i = 0; i < n_codecs; ++i)
      puts(codecs[i].name);
    return 0;
  }
  if (argc != 7) {
    fprintf(stderr,
            "usage: %s endpoint bucket prefix codec buffered multipart\n",
            argv[0]);
    return 1;
  }
  int err = 1;
  const int buffered = atoi(argv[5]);
  const int multipart = atoi(argv[6]);
  const int shard_frames = multipart ? 16 : 2;
  const int nt = 2 * shard_frames + 1;
  const size_t frame_bytes =
    NGFF_READBACK_NY * NGFF_READBACK_NX * sizeof(uint16_t);
  const size_t total_bytes = (nt + (buffered == 2)) * frame_bytes;
  uint8_t* allocation = NULL;
  struct test_zarr_multiscale sink = { 0 };
  struct stream_type* stream = NULL;
  struct buffered_writer* adapter = NULL;
#ifdef TEST_NGFF_READBACK_GPU
  CUcontext context = 0;
  CUdevice device;
  CU(Done, cuInit(0));
  CU(Done, cuDeviceGet(&device, 0));
  CU(Done, cu_ctx_create(&context, 0, device));
#endif
  CHECK(Done, buffered >= 0 && buffered <= 2);
  size_t index = 0;
  while (index < n_codecs && strcmp(codecs[index].name, argv[4]) != 0)
    ++index;
  CHECK(Done, index < n_codecs);
  allocation = malloc(total_bytes + 1);
  CHECK(Done, allocation);
  uint8_t* src = allocation + 1;
  ngff_readback_fill(src, nt + (buffered == 2));
  struct dimension dims[3];
  ngff_readback_dimensions(
    dims, buffered == 2 ? (uint64_t)nt : 0, shard_frames, multipart ? 4 : 2);
  // A full L0 multipart shard is 16 * 512 * 512 * 2 = 8 MiB plus its index.
  struct store_s3_config s3 = {
    .bucket = argv[2],
    .prefix = argv[3],
    .region = "us-east-1",
    .endpoint = argv[1],
    .part_size = 5 * 1024 * 1024,
    .throughput_gbps = 1.0,
    .max_retries = 1,
    .backoff_scale_ms = 100,
    .max_backoff_secs = 1,
    .timeout_ns = 30000000000ULL,
  };
  CHECK(Done,
        test_zarr_multiscale_open_in_store(&sink,
                                           store_s3_create(&s3),
                                           "pyramid",
                                           dims,
                                           3,
                                           dtype_u16,
                                           NGFF_READBACK_LEVELS,
                                           codecs[index].codec,
                                           ngff_readback_axes) == 0);
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = frame_bytes,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codecs[index].codec,
    .reduce_method = lod_reduce_mean,
    .max_nlod = NGFF_READBACK_LEVELS,
    .epochs_per_batch = 1,
    .metadata_update_interval_s = 0,
    .max_threads = 2,
  };
  stream = stream_create(&config, test_zarr_multiscale_as_shard_sink(&sink));
  CHECK(Done, stream);
  struct writer* writer = stream_writer(stream);
  if (buffered) {
    struct buffered_writer_config buffering = { 2 * 1023 * 3, 2 * 997 };
    if (buffered == 2)
      buffering = (struct buffered_writer_config){ total_bytes, 0 };
    adapter = buffered_writer_create(writer, &buffering);
    CHECK(Done, adapter);
    writer = buffered_writer_as_writer(adapter);
  }
  printf("{\"nt\":%d,\"ny\":%d,\"nx\":%d,\"shard_frames\":%d}\n",
         nt,
         NGFF_READBACK_NY,
         NGFF_READBACK_NX,
         shard_frames);
  fflush(stdout);
  int frames = 0, target = -1;
  while (scanf("%d", &target) == 1 && target != 0) {
    CHECK(Done, target > frames && target <= nt + (buffered == 2));
    while (frames < target) {
      uint8_t* frame = src + frames * frame_bytes;
      CHECK(
        Done,
        writer_append_wait(writer, (struct slice){ frame, frame + frame_bytes })
            .error == 0);
      memset(frame, 0xFF, frame_bytes);
      ++frames;
    }
    printf("appended %d\n", frames);
    fflush(stdout);
  }
  CHECK(Done, target == 0 && frames == nt + (buffered == 2));
  CHECK(Done, writer_flush(writer).error == (buffered == 2));
  CHECK(Done, writer_close(writer).error == (buffered == 2));
  CHECK(Done, ngff_multiscale_flush(sink.ms) == 0);
  if (buffered == 2) {
    const struct buffered_writer_stats stats =
      buffered_writer_get_stats(adapter);
    CHECK(Done, stats.downstream_finished && stats.failed);
    CHECK(Done, stats.forwarded_bytes == nt * frame_bytes);
    CHECK(Done, stats.abandoned_bytes == frame_bytes);
  }
  err = 0;
Done:
  buffered_writer_destroy(adapter);
  stream_destroy(stream);
  test_zarr_multiscale_close(&sink);
  free(allocation);
#ifdef TEST_NGFF_READBACK_GPU
  if (context)
    cuCtxDestroy(context);
#endif
  return err;
}
