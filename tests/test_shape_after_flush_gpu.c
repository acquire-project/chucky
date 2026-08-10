// What the shape says after a flush, on the GPU backend (#193). A flush pads
// the chunk it lands in and closes the shard, so chunks appended afterwards
// start past the padding and past the slots the flush left empty. The shape has
// to reach them. Mirrors tests/test_shape_after_flush_cpu.c.

#include "stream.gpu.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include "test_runner.h"

#include <stdlib.h>

#define SHARD_CAP (1 << 20)
#define MAX_ROUNDS 2

// One round of appends followed by a flush.
struct flush_round
{
  int planes; // handed over before this round's flush
  uint64_t expect_shape0;
};

struct shape_case
{
  uint64_t chunk_size; // along the append dim
  int n_rounds;
  struct flush_round rounds[MAX_ROUNDS];
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
  // update is one a flush wrote.
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

  const uint64_t plane_elems = 4 * 4;
  data = (uint16_t*)calloc(plane_elems, sizeof(uint16_t));
  CHECK(Fail, data);

  struct writer* w = tile_stream_gpu_writer(s);
  CHECK(Fail, sink.update_append_count == 0);

  for (int round = 0; round < c->n_rounds; ++round) {
    const struct flush_round* r = &c->rounds[round];
    for (int i = 0; i < r->planes; ++i) {
      struct slice sl = { .beg = data, .end = data + plane_elems };
      CHECK(Fail, writer_append(w, sl).error == 0);
    }

    CHECK(Fail, writer_flush(w).error == 0);
    log_info("  round %d: updates=%d shape0=%llu",
             round,
             sink.update_append_count,
             (unsigned long long)sink.last_append_size0);
    CHECK(Fail, sink.update_append_count == round + 1);
    CHECK(Fail, sink.last_append_size0 == r->expect_shape0);
  }

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

// The first flush closes shard 0 holding one chunk, leaving its second slot
// empty. The next two chunks go to shard 1, at append positions 4 through 7, so
// the shape has to reach 8 to cover them.
static int
test_shape_covers_chunks_after_flush(void)
{
  const struct shape_case c = { .chunk_size = 2,
                                .n_rounds = 2,
                                .rounds = { { .planes = 2,
                                              .expect_shape0 = 2 },
                                            { .planes = 4,
                                              .expect_shape0 = 8 } } };
  return run_shape_case(&c);
}

// The first flush pads a chunk as well as closing the shard. The plane appended
// after it lands in chunk 2, whose padding stops one element short of the
// chunk's end, so the shape is 8 rather than 9.
static int
test_shape_covers_padded_chunk_after_flush(void)
{
  const struct shape_case c = { .chunk_size = 3,
                                .n_rounds = 2,
                                .rounds = { { .planes = 1,
                                              .expect_shape0 = 1 },
                                            { .planes = 1,
                                              .expect_shape0 = 8 } } };
  return run_shape_case(&c);
}

RUN_GPU_TESTS({ "shape_covers_chunks_after_flush",
                test_shape_covers_chunks_after_flush },
              { "shape_covers_padded_chunk_after_flush",
                test_shape_covers_padded_chunk_after_flush }, )
