// What the shape says after a flush, on the GPU backend (#193). A flush pads
// the chunk the cursor stopped in and closes its shard, so it finalizes the
// stream: anything appended afterwards would land at append positions the
// caller never asked for. Mirrors tests/test_shape_after_flush_cpu.c.

#include "stream.gpu.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include "test_runner.h"

#include <stdlib.h>

#define SHARD_CAP (1 << 20)
#define PLANE_ELEMS 16

struct shape_case
{
  uint64_t chunk_size; // along the append dim
  int planes;
  uint64_t expect_shape0;
};

static int
run_shape_case(const struct shape_case* c)
{
  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, SHARD_CAP);

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = c->chunk_size,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 4,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 1 },
    { .size = 4,
      .chunk_size = 4,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 2 },
  };

  // A long interval keeps the periodic update from firing, so every recorded
  // update is one the flush wrote.
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 1,
    .metadata_update_interval_s = 3600.0f,
  };

  uint16_t* data = NULL;
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &sink.base);
  CHECK(Fail, s);

  data = (uint16_t*)calloc(PLANE_ELEMS, sizeof(uint16_t));
  CHECK(Fail, data);

  struct writer* w = tile_stream_gpu_writer(s);
  CHECK(Fail, sink.update_append_count == 0);

  for (int i = 0; i < c->planes; ++i) {
    struct slice sl = { .beg = data, .end = data + PLANE_ELEMS };
    CHECK(Fail, writer_append(w, sl).error == 0);
  }

  CHECK(Fail, writer_flush(w).error == 0);
  // flush queues the writes; the extent is published once they have landed.
  CHECK(Fail, writer_close(w).error == 0);
  log_info("  updates=%d shape0=%llu",
           sink.update_append_count,
           (unsigned long long)sink.last_append_size0);
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == c->expect_shape0);

  // The flush finalized the stream: no more input, and the shape it published
  // is the last word.
  {
    struct slice sl = { .beg = data, .end = data + PLANE_ELEMS };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == writer_error_finished);
    CHECK(Fail, r.rest.beg == sl.beg && r.rest.end == sl.end);
  }
  CHECK(Fail, writer_flush(w).error == 0);
  CHECK(Fail, writer_close(w).error == 0);
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == c->expect_shape0);

  free(data);
  tile_stream_gpu_destroy(s);
  test_sink_free(&sink);
  return 0;

Fail:
  free(data);
  tile_stream_gpu_destroy(s);
  test_sink_free(&sink);
  return 1;
}

// One plane leaves the chunk two thirds full; the flush pads it and the shape
// reports the plane appended, not the padding.
static int
test_shape_exact_on_partial_chunk(void)
{
  const struct shape_case c = { .chunk_size = 3,
                                .planes = 1,
                                .expect_shape0 = 1 };
  return run_shape_case(&c);
}

// Whole chunks need no padding, and two of them fill the shard.
static int
test_shape_exact_on_whole_chunks(void)
{
  const struct shape_case c = { .chunk_size = 2,
                                .planes = 4,
                                .expect_shape0 = 4 };
  return run_shape_case(&c);
}

RUN_GPU_TESTS({ "shape_exact_on_partial_chunk",
                test_shape_exact_on_partial_chunk },
              { "shape_exact_on_whole_chunks",
                test_shape_exact_on_whole_chunks }, )
