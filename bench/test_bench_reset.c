#include "bench_gpu.h"
#include "stream.cpu.h"
#include "stream/layouts.h"
#include "test_shard_sink.h"
#include "test_shard_verify.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// Use actual backends and readable output to verify that resetting counters
// neither finalizes a writer nor loses accepted warmup data.
int
main(int argc, char** argv)
{
  if (argc != 2)
    return 1;
  const int gpu = strcmp(argv[1], "gpu") == 0;
  if (gpu && bench_gpu_context_create())
    return 1;
  int rc = 1;
  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 0, 8, 8 });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 2, 2, 2 });
  dims[0].chunks_per_shard = 2;
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });
  dims_set_downsample_by_name(dims, 3, "tyx");
  const struct tile_stream_configuration config = {
    .dtype = dtype_u16,
    .dimensions = dims,
    .rank = 3,
    .buffer_capacity_bytes = 4096,
    .epochs_per_batch = 2,
    .codec = { .id = CODEC_ZSTD },
    .max_threads = 2,
  };
  struct test_shard_sink sink;
  test_sink_init_multi(&sink, 3, (int[]){ 16, 16, 16 }, 64 << 10);
  struct tile_stream_cpu* cpu = NULL;
  struct tile_stream_gpu* device = NULL;
  uint16_t* data = NULL;
  struct writer* writer;
  const struct tile_stream_layout* layout;
  if (gpu) {
    device = bench_gpu_create(&config, &sink.base);
    CHECK(Fail, device);
    writer = bench_gpu_writer(device);
    layout = bench_gpu_layout(device);
  } else {
    cpu = tile_stream_cpu_create(&config, &sink.base);
    CHECK(Fail, cpu);
    writer = tile_stream_cpu_writer(cpu);
    layout = tile_stream_cpu_layout(cpu);
  }
  const size_t warm_elements = 4 * layout->epoch_elements;
  data = malloc(2 * warm_elements * sizeof(*data));
  CHECK(Fail, data);
  for (size_t i = 0; i < 2 * warm_elements; ++i)
    data[i] = i < warm_elements ? 1 : 2;
  CHECK(Fail,
        writer_append_wait(writer, (struct slice){ data, data + 1 }).error ==
          0);
  CHECK(Fail,
        (gpu ? bench_gpu_reset_metrics(device)
             : tile_stream_cpu_reset_metrics(cpu)) == 1);
  CHECK(Fail,
        writer_append_wait(writer,
                           (struct slice){ data + 1, data + warm_elements / 2 })
            .error == 0);
  // One full batch still leaves the coarsest append-downsample level partial.
  CHECK(Fail,
        (gpu ? bench_gpu_reset_metrics(device)
             : tile_stream_cpu_reset_metrics(cpu)) == 1);
  CHECK(
    Fail,
    writer_append_wait(
      writer, (struct slice){ data + warm_elements / 2, data + warm_elements })
        .error == 0);
  CHECK(Fail,
        (gpu ? bench_gpu_reset_metrics(device)
             : tile_stream_cpu_reset_metrics(cpu)) == 0);
  struct stream_metrics m =
    gpu ? bench_gpu_get_metrics(device) : tile_stream_cpu_get_metrics(cpu);
  CHECK(Fail,
        m.compress.count == 0 && m.scatter.count == 0 && m.append_count == 0 &&
          m.max_append_ms == 0 && m.memcpy_bytes == 0 &&
          m.d2h_payload_copy_count == 0);
  CHECK(Fail,
        m.compress.name && m.compress.owner == METRIC_OWNER_COMPRESS &&
          m.compress.best_ms == 1e30f);
  CHECK(
    Fail,
    writer_append_wait(
      writer, (struct slice){ data + warm_elements, data + 2 * warm_elements })
        .error == 0);
  CHECK(Fail,
        writer_flush(writer).error == 0 && writer_close(writer).error == 0);
  CHECK(Fail,
        (gpu ? bench_gpu_reset_metrics(device)
             : tile_stream_cpu_reset_metrics(cpu)) == -1);
  CHECK(Fail,
        (gpu ? bench_gpu_cursor(device) : tile_stream_cpu_cursor(cpu)) ==
          2 * warm_elements);
  m = gpu ? bench_gpu_get_metrics(device) : tile_stream_cpu_get_metrics(cpu);
  CHECK(Fail, m.compress.count == 2);
  CHECK(Fail, m.scatter.input_bytes == warm_elements * sizeof(*data));
  if (gpu)
    CHECK(Fail,
          m.memcpy_bytes == warm_elements * sizeof(*data) &&
            m.h2d.input_bytes == warm_elements * sizeof(*data));
  // Four L0 shards, with two warmup and two measured generations.
  for (int i = 0; i < 4; ++i) {
    const struct test_shard_writer* shard = &sink.writers[0][i];
    uint64_t offsets[32], sizes[32];
    CHECK(Fail,
          shard_index_parse(shard->buf, shard->size, 32, offsets, sizes) == 0);
    CHECK(Fail, shard_index_check_crc(shard->buf, shard->size, 32) == 0);
    for (int c = 0; c < 32; ++c)
      CHECK(Fail,
            chunk_decompress_verify_u16(
              shard->buf + offsets[c], sizes[c], 16, 8, i < 2 ? 1 : 2) == 0);
  }
  rc = 0;
Fail:
  if (gpu)
    bench_gpu_destroy(device);
  else
    tile_stream_cpu_destroy(cpu);
  free(data);
  test_sink_free(&sink);
  if (gpu)
    bench_gpu_context_destroy();
  return rc;
}
