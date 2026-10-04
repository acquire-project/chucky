// Packed input must land at logical coordinates, leaving chunk edges zero.
// Compile the same regression for both stream backends.
#include "test_shard_sink.h"
#include "test_shard_verify.h"
#include "util/prelude.h"

#ifdef TEST_GPU
#include "multiarray.gpu.h"
#include "stream.gpu.h"
#include "test_runner.h"
#define stream_create tile_stream_gpu_create
#define stream_writer tile_stream_gpu_writer
#define stream_cursor tile_stream_gpu_cursor
#define stream_destroy tile_stream_gpu_destroy
#define stream_type tile_stream_gpu
#define stream_memory_info tile_stream_memory_info
#define stream_memory_estimate tile_stream_gpu_memory_estimate
#define multi_create multiarray_tile_stream_gpu_create
#define multi_writer multiarray_tile_stream_gpu_writer
#define multi_destroy multiarray_tile_stream_gpu_destroy
#define multi_type multiarray_tile_stream_gpu
#else
#include "multiarray.cpu.h"
#include "stream.cpu.h"
#define stream_create tile_stream_cpu_create
#define stream_writer tile_stream_cpu_writer
#define stream_cursor tile_stream_cpu_cursor
#define stream_destroy tile_stream_cpu_destroy
#define stream_type tile_stream_cpu
#define stream_memory_info tile_stream_cpu_memory_info
#define stream_memory_estimate tile_stream_cpu_memory_estimate
#define multi_create multiarray_tile_stream_cpu_create
#define multi_writer multiarray_tile_stream_cpu_writer
#define multi_destroy multiarray_tile_stream_cpu_destroy
#define multi_type multiarray_tile_stream_cpu
#endif

#include <stdlib.h>
#include <string.h>

static uint16_t
pixel(uint64_t t, uint64_t y, uint64_t x, uint64_t width)
{
  return (uint16_t)(1 + 1000 * t + y * width + x);
}

static int
run_case(uint64_t frames,
         uint64_t height,
         uint64_t width,
         uint64_t depth,
         size_t append_elements,
         int bounded,
         int transpose,
         int multiscale,
         uint64_t channels)
{
  int error = 1;
  struct stream_type* stream = NULL;
  struct test_shard_sink sink;
  log_info("packed: %llux%llux%llux%llu depth=%llu append=%zu bounded=%d "
           "transpose=%d lod=%d",
           (unsigned long long)frames,
           (unsigned long long)channels,
           (unsigned long long)height,
           (unsigned long long)width,
           (unsigned long long)depth,
           append_elements,
           bounded,
           transpose,
           multiscale);
  const int shard_counts[] = { 8, 8 };
  const int nlod = multiscale ? 2 : 1;
  test_sink_init_multi(&sink, nlod, shard_counts, 4 << 20);
  uint16_t* input =
    malloc((frames + 1) * channels * height * width * sizeof(*input));
  CHECK(Cleanup, input);
  for (uint64_t t = 0; t < (frames + 1) * channels; ++t)
    for (uint64_t y = 0; y < height; ++y)
      for (uint64_t x = 0; x < width; ++x)
        input[(t * height + y) * width + x] = pixel(t, y, x, width);

  struct dimension dims[4];
  const int rank = channels == 1 ? 3 : 4;
  if (rank == 3) {
    dims_create(
      dims, "tyx", (uint64_t[]){ bounded ? frames : 0, height, width });
    dims_set_chunk_sizes(dims, 3, (uint64_t[]){ depth, 8, 8 });
    dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });
  } else {
    dims_create(dims,
                "tcyx",
                (uint64_t[]){ bounded ? frames : 0, channels, height, width });
    dims_set_chunk_sizes(dims, 4, (uint64_t[]){ depth, 1, 8, 8 });
    dims_set_shard_counts(dims, 4, (uint64_t[]){ 0, 1, 1, 1 });
  }
  dims[0].chunks_per_shard = 4;
  dims[rank - 2].downsample = dims[rank - 1].downsample = multiscale;
  if (transpose) {
    dims[rank - 2].storage_position = rank - 1;
    dims[rank - 1].storage_position = rank - 2;
  }
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 510,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 2,
    .max_nlod = nlod,
    .max_threads = 4,
    .reduce_method = lod_reduce_max,
  };
  if (!multiscale) {
    struct stream_memory_info memory;
    const uint64_t rows = depth * height * (depth > 1 ? channels : 1);
    const size_t table_bytes =
      (height % 8 || width % 8) ? (rows + width) * sizeof(uint64_t) : 0;
    CHECK(Cleanup, !stream_memory_estimate(&config, 0, &memory));
    CHECK(Cleanup, memory.nlod == 1 && memory.lod_bytes == table_bytes);
  }
  stream = stream_create(&config, &sink.base);
  CHECK(Cleanup, stream);
  struct writer* writer = stream_writer(stream);
  const size_t total = frames * channels * height * width;
  for (size_t offset = 0; offset < total;) {
    size_t count = append_elements;
    if (count > total - offset)
      count = total - offset;
    const struct slice part = { input + offset, input + offset + count };
    const struct writer_result r = writer_append_wait(writer, part);
    CHECK(Cleanup, !r.error && r.rest.beg == r.rest.end);
    offset += count;
  }
  if (bounded) {
    const struct slice extra = { input + total,
                                 input + total + height * width };
    const struct writer_result r = writer_append_wait(writer, extra);
    CHECK(Cleanup, r.error == writer_error_finished && r.rest.beg == extra.beg);
  }
  CHECK(Cleanup, stream_cursor(stream) == total);
  CHECK(Cleanup, !writer_flush(writer).error);
  CHECK(Cleanup, !writer_close(writer).error);
  CHECK(Cleanup, sink.last_append_size0 == frames);

  for (int lv = 0; lv < nlod; ++lv) {
    const uint64_t scale = 1u << lv;
    const uint64_t ny = ceildiv(height, scale), nx = ceildiv(width, scale);
    const uint64_t cy = ceildiv(ny, 8), cx = ceildiv(nx, 8);
    const uint64_t ct =
      bounded && ceildiv(frames, depth) < 4 ? ceildiv(frames, depth) : 4;
    const size_t slots = ct * channels * cy * cx;
    uint64_t* offsets = malloc(slots * sizeof(*offsets));
    uint64_t* sizes = malloc(slots * sizeof(*sizes));
    if (!offsets || !sizes) {
      free(offsets);
      free(sizes);
      goto Cleanup;
    }
    int bad = 0;
    for (uint64_t sh = 0; sh < ceildiv(frames, ct * depth) && !bad; ++sh) {
      const struct test_shard_writer* w = &sink.writers[lv][sh];
      if (!w->finalized || shard_index_check_crc(w->buf, w->size, slots) ||
          shard_index_parse(w->buf, w->size, slots, offsets, sizes)) {
        bad = 1;
        break;
      }
      for (uint64_t slot = 0; slot < slots && !bad; ++slot) {
        const uint64_t tc = sh * ct + slot / (channels * cy * cx);
        const uint64_t channel = slot / (cy * cx) % channels;
        if (tc * depth >= frames) {
          bad = sizes[slot] != UINT64_MAX;
          continue;
        }
        if (sizes[slot] != depth * 8 * 8 * sizeof(uint16_t) ||
            offsets[slot] > w->size || sizes[slot] > w->size - offsets[slot]) {
          bad = 1;
          break;
        }
        const uint64_t yc = transpose ? slot % cy : (slot / cx) % cy;
        const uint64_t xc = transpose ? (slot / cy) % cx : slot % cx;
        for (uint64_t t = 0; t < depth; ++t)
          for (uint64_t y = 0; y < 8; ++y)
            for (uint64_t x = 0; x < 8; ++x) {
              uint64_t gt = tc * depth + t, gy = yc * 8 + y, gx = xc * 8 + x;
              uint16_t expected = 0, actual;
              if (gt < frames && gy < ny && gx < nx) {
                // Independent max-reduction reference, including odd edges.
                for (uint64_t iy = gy * scale;
                     iy < (gy + 1) * scale && iy < height;
                     ++iy)
                  for (uint64_t ix = gx * scale;
                       ix < (gx + 1) * scale && ix < width;
                       ++ix) {
                    uint16_t value =
                      pixel(gt * channels + channel, iy, ix, width);
                    if (value > expected)
                      expected = value;
                  }
              }
              const uint64_t pos = t * 64 + (transpose ? x * 8 + y : y * 8 + x);
              memcpy(&actual, w->buf + offsets[slot] + pos * 2, 2);
              if (actual != expected && !bad) {
                log_error("level %d pixel (%llu,%llu,%llu): %u != %u",
                          lv,
                          (unsigned long long)gt,
                          (unsigned long long)gy,
                          (unsigned long long)gx,
                          actual,
                          expected);
                bad = 1;
              }
            }
      }
    }
    free(offsets);
    free(sizes);
    CHECK(Cleanup, !bad);
  }
  error = 0;
Cleanup:
  stream_destroy(stream);
  test_sink_free(&sink);
  free(input);
  return error;
}

static int
test_multiarray_packed(void)
{
  int error = 1;
  struct multi_type* stream = NULL;
  struct test_shard_sink sink[3];
  struct shard_sink* sinks[3];
  struct dimension dims[3][3];
  struct tile_stream_configuration configs[3];
  const uint64_t heights[] = { 3, 5, 4 }, widths[] = { 5, 7, 8 };
  // Two different packed mappings and one aligned layout share the pools.
  for (int a = 0; a < 3; ++a) {
    test_sink_init(&sink[a], 4, 4096);
    sinks[a] = &sink[a].base;
    dims_create(dims[a], "tyx", (uint64_t[]){ 0, heights[a], widths[a] });
    dims_set_chunk_sizes(dims[a], 3, (uint64_t[]){ 1, 4, 4 });
    dims_set_shard_counts(dims[a], 3, (uint64_t[]){ 0, 1, 1 });
    dims[a][0].chunks_per_shard = 2;
    if (a == 1) {
      dims[a][1].storage_position = 2;
      dims[a][2].storage_position = 1;
    }
    configs[a] = (struct tile_stream_configuration){
      .buffer_capacity_bytes = 128,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = dims[a],
      .codec = { .id = CODEC_NONE },
      .epochs_per_batch = 2,
      .max_threads = 2,
    };
  }
  stream = multi_create(3, configs, sinks, 0);
  CHECK(Cleanup, stream);
  struct multiarray_writer* writer = multi_writer(stream);
  for (uint64_t t = 0; t < 5; ++t)
    for (int a = 0; a < 3; ++a) {
      uint16_t input[35];
      const size_t n = heights[a] * widths[a];
      for (uint64_t y = 0; y < heights[a]; ++y)
        for (uint64_t x = 0; x < widths[a]; ++x)
          input[y * widths[a] + x] = pixel(t + 10 * a, y, x, widths[a]);
      for (size_t begin = 0; begin < n;) {
        const size_t end = begin == 0 ? n / 2 : n;
        const struct multiarray_writer_result r = writer->update(
          writer, a, (struct slice){ input + begin, input + end });
        CHECK(Cleanup, !r.error && r.rest.beg == r.rest.end);
        begin = end;
      }
    }
  CHECK(Cleanup, !writer->flush(writer).error && !writer->close(writer).error);
  for (int a = 0; a < 3; ++a) {
    CHECK(Cleanup, sink[a].last_append_size0 == 5);
    const uint64_t cy = ceildiv(heights[a], 4), cx = ceildiv(widths[a], 4);
    const size_t slots = 2 * cy * cx;
    for (uint64_t sh = 0; sh < 3; ++sh) {
      uint64_t offsets[8], sizes[8];
      const struct test_shard_writer* w = &sink[a].writers[0][sh];
      CHECK(Cleanup, w->finalized);
      CHECK(Cleanup, !shard_index_check_crc(w->buf, w->size, slots));
      CHECK(Cleanup,
            !shard_index_parse(w->buf, w->size, slots, offsets, sizes));
      for (uint64_t t = 0; t < 2; ++t)
        for (uint64_t iy = 0; iy < cy; ++iy)
          for (uint64_t ix = 0; ix < cx; ++ix) {
            const uint64_t slot =
              t * cy * cx + (a == 1 ? ix * cy + iy : iy * cx + ix);
            if (sh * 2 + t >= 5) {
              CHECK(Cleanup, sizes[slot] == UINT64_MAX);
              continue;
            }
            CHECK(Cleanup,
                  sizes[slot] == 32 && offsets[slot] <= w->size &&
                    sizes[slot] <= w->size - offsets[slot]);
            for (uint64_t y = 0; y < 4; ++y)
              for (uint64_t x = 0; x < 4; ++x) {
                const uint64_t gy = iy * 4 + y, gx = ix * 4 + x;
                uint16_t expected =
                  gy < heights[a] && gx < widths[a]
                    ? pixel(sh * 2 + t + 10 * a, gy, gx, widths[a])
                    : 0;
                uint16_t actual;
                const uint64_t pos = a == 1 ? x * 4 + y : y * 4 + x;
                memcpy(&actual, w->buf + offsets[slot] + pos * 2, 2);
                CHECK(Cleanup, actual == expected);
              }
          }
    }
  }
  error = 0;
Cleanup:
  multi_destroy(stream);
  for (int a = 0; a < 3; ++a)
    test_sink_free(&sink[a]);
  return error;
}

static int
test_packed(void)
{
  // The reported reproducer, a divisible control, sub-chunk dimensions,
  // split rows/frames, reused batches/shards, bounded tails and storage order.
  CHECK(Fail, !run_case(3, 9, 13, 1, 117, 0, 0, 0, 1));
  CHECK(Fail, !run_case(3, 8, 16, 1, 128, 0, 0, 0, 1));
  CHECK(Fail, !run_case(3, 3, 5, 2, 7, 0, 0, 0, 1));
  for (int bounded = 0; bounded < 2; ++bounded)
    for (int transpose = 0; transpose < 2; ++transpose)
      for (int lod = 0; lod < 2; ++lod)
        CHECK(Fail, !run_case(11, 9, 13, 2, 131, bounded, transpose, lod, 1));
  CHECK(Fail, !run_case(3, 257, 263, 1, 70001, 0, 0, 0, 1));
  CHECK(Fail, !run_case(11, 8, 16, 2, 131, 1, 0, 0, 1));
  for (int lod = 0; lod < 2; ++lod) {
    CHECK(Fail, !run_case(3, 9, 13, 1, 131, 0, 1, lod, 3));
    CHECK(Fail, !run_case(3, 9, 13, 2, 131, 1, 1, lod, 3));
  }
  CHECK(Fail, !test_multiarray_packed());
  return 0;
Fail:
  return 1;
}

#ifdef TEST_GPU
RUN_GPU_TESTS({ "packed_input", test_packed })
#else
int
main(void)
{
  return test_packed();
}
#endif
