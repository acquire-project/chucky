#include "multiarray.gpu.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "test_runner.h"
#include "test_shard_sink.h"
#include "test_shard_verify.h"

#include <stdlib.h>

static int
run_copies(int full, int mixed, int finite)
{
  enum
  {
    CHUNK = 4096,
    CPS = 16,
    STAGING = 32768,
    BATCH = 12 * CHUNK
  };
  const size_t total = 2 * 1024 * 1024 + 123;
  uint8_t* input = malloc(total);
  struct test_shard_sink sink;
  struct tile_stream_gpu* stream = NULL;
  int result = 1;
  test_sink_init(&sink, 33, 128 * 1024);
  CHECK(Fail, input);
  for (size_t i = 0; i < total; ++i)
    input[i] = (uint8_t)((i * 137) ^ (i >> 9));
  struct dimension dim = { .size = finite ? total : 0,
                           .chunk_size = CHUNK,
                           .chunks_per_shard = CPS };
  const struct tile_stream_configuration cfg = {
    .buffer_capacity_bytes = STAGING,
    .dtype = dtype_u8,
    .rank = 1,
    .dimensions = &dim,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 12,
    .full_memcpy_timing = full,
  };
  stream = tile_stream_gpu_create(&cfg, &sink.base);
  CHECK(Fail, stream);
  CHECK(Fail, tile_stream_gpu_layout(stream)->epoch_elements == CHUNK);
  struct writer* w = tile_stream_gpu_writer(stream);
  const size_t offers[] = { 511, 8191, 8192, 8193, 131072 };
  uint64_t copies = 0, small = 0, timed = 0, timed_bytes = 0, appends = 0;
  size_t accepted = 0;
  while (accepted < total) {
    size_t offer = mixed ? offers[appends % countof(offers)] : 512;
    if (offer > total - accepted)
      offer = total - accepted;
    size_t remaining = offer;
    while (remaining) {
      const size_t batch_offset = accepted % BATCH;
      size_t n = STAGING - batch_offset % STAGING;
      if (n > BATCH - batch_offset)
        n = BATCH - batch_offset;
      if (n > remaining)
        n = remaining;
      copies++;
      int measure = full || n >= 8192;
      if (!measure) {
        measure = small % 64 == (small / 64) % 64;
        small++;
      }
      if (measure) {
        timed++;
        timed_bytes += n;
      }
      accepted += n;
      remaining -= n;
    }
    struct writer_result r = writer_append(
      w, (struct slice){ input + accepted - offer, input + accepted });
    CHECK(Fail, r.error == 0 && r.rest.beg == r.rest.end);
    appends++;
  }
  CHECK(Fail, writer_flush(w).error == 0);
  CHECK(Fail, writer_flush(w).error == 0);
  CHECK(Fail, writer_close(w).error == 0);
  const struct stream_metrics m = tile_stream_gpu_get_metrics(stream);
  CHECK(Fail, m.memcpy_calls == copies);
  CHECK(Fail, m.memcpy_bytes == total);
  CHECK(Fail, (uint64_t)m.memcpy.count == timed);
  CHECK(Fail, m.memcpy.input_bytes == (double)timed_bytes);
  CHECK(Fail, m.memcpy.output_bytes == (double)timed_bytes);
  CHECK(Fail, m.append_count == appends);
  CHECK(Fail, m.scatter_samples_lost == 0);
  CHECK(Fail, full || timed < copies);
  CHECK(Fail, m.memcpy.max_ms <= m.max_append_ms);
  const struct writer_result refused =
    writer_append(w, (struct slice){ input, input + 512 });
  CHECK(Fail, refused.error == writer_error_finished);
  CHECK(Fail, refused.rest.beg == input);
  CHECK(Fail, tile_stream_gpu_get_metrics(stream).memcpy_calls == copies);

  for (size_t sh = 0; sh < 33; ++sh) {
    const struct test_shard_writer* sw = &sink.writers[0][sh];
    uint64_t offsets[CPS], sizes[CPS];
    CHECK(Fail, sw->finalized && sw->size);
    CHECK(Fail, shard_index_parse(sw->buf, sw->size, CPS, offsets, sizes) == 0);
    CHECK(Fail, shard_index_check_crc(sw->buf, sw->size, CPS) == 0);
    for (size_t c = 0; c < CPS; ++c) {
      const size_t start = (sh * CPS + c) * CHUNK;
      if (start >= total)
        break;
      CHECK(Fail, sizes[c] == CHUNK);
      CHECK(Fail, offsets[c] <= sw->size - CHUNK);
      for (size_t j = 0; j < CHUNK; ++j)
        CHECK(Fail,
              sw->buf[offsets[c] + j] ==
                (start + j < total ? input[start + j] : 0));
    }
  }
  result = 0;
Fail:
  tile_stream_gpu_destroy(stream);
  test_sink_free(&sink);
  free(input);
  return result;
}

static int
test_copy_modes(void)
{
  for (int full = 0; full < 2; ++full)
    for (int mixed = 0; mixed < 2; ++mixed)
      for (int finite = 0; finite < 2; ++finite)
        if (run_copies(full, mixed, finite))
          return 1;
  return 0;
}

static int
test_per_array_phase(void)
{
  struct dimension dim = { .size = 0,
                           .chunk_size = 4096,
                           .chunks_per_shard = 16 };
  struct tile_stream_configuration cfg[3];
  struct test_shard_sink sinks[3];
  struct shard_sink* bases[3];
  struct multiarray_tile_stream_gpu* stream = NULL;
  int result = 1;
  for (int a = 0; a < 3; ++a) {
    test_sink_init(&sinks[a], 1, 128 * 1024);
    bases[a] = &sinks[a].base;
    cfg[a] = (struct tile_stream_configuration){
      .buffer_capacity_bytes = 32768,
      .dtype = dtype_u8,
      .rank = 1,
      .dimensions = &dim,
      .codec = { .id = CODEC_NONE },
      .epochs_per_batch = 12,
      .full_memcpy_timing = a == 2,
    };
  }
  stream = multiarray_tile_stream_gpu_create(3, cfg, bases, 0);
  CHECK(Fail, stream);
  struct multiarray_writer* w = multiarray_tile_stream_gpu_writer(stream);
  const uint8_t input[512] = { 0 };
  for (int round = 0; round < 2; ++round)
    for (int a = 0; a < 3; ++a)
      for (int i = 0; i < 8; ++i) {
        struct multiarray_writer_result r =
          w->update(w, a, (struct slice){ input, input + sizeof(input) });
        CHECK(Fail, r.error == 0 && r.rest.beg == r.rest.end);
      }
  CHECK(Fail, w->flush(w).error == 0);
  const struct stream_metrics m =
    multiarray_tile_stream_gpu_get_metrics(stream);
  CHECK(Fail, m.memcpy_calls == 48 && m.memcpy_bytes == 48 * 512);
  CHECK(Fail, m.memcpy.count == 18 && m.memcpy.input_bytes == 18 * 512);
  CHECK(Fail, m.append_count == 0);
  result = 0;
Fail:
  multiarray_tile_stream_gpu_destroy(stream);
  for (int a = 0; a < 3; ++a)
    test_sink_free(&sinks[a]);
  return result;
}

RUN_GPU_TESTS({ "copy modes and readback", test_copy_modes },
              { "per-array sampling phase", test_per_array_phase })
