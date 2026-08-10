// What the shape says after a flush, on the CPU backend.
//
// #175: a failed flush leaves shards on disk that a reader cannot see unless
// the shape is written. The shape names finalized shards, so it is truthful on
// the failing path and worth writing.
//
// #193: a flush pads the chunk it lands in and closes the shard, so chunks
// appended afterwards start past the padding and past the slots the flush left
// empty. The shape has to reach them.

#include "stream.cpu.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include <stdlib.h>

#define SHARD_CAP (1 << 20)

// Two shards along the append dim can be opened; a third open fails, which is
// a delivery failure with no sink IO error behind it.
#define OPENABLE_SHARDS 2

#define MAX_ROUNDS 2

// One round of appends followed by a flush.
struct flush_round
{
  int planes; // handed over before this round's flush
  int expect_error;
  uint64_t expect_shape0;
};

struct shape_case
{
  const char* name;
  uint64_t chunk_size; // along the append dim
  uint32_t epochs_per_batch;
  int n_rounds;
  struct flush_round rounds[MAX_ROUNDS];
  // Optional: where the planes ended up, once every round has run.
  int (*verify)(const struct test_shard_sink* sink);
};

#define PLANE_ELEMS 16

// Each round fills its planes with its own value, so a check can say which
// round's plane it is looking at.
static uint16_t
plane_fill(int round)
{
  return (uint16_t)(0xa500 + round);
}

// Pins where the planes land, so a shape off by one in either direction fails
// here. Round 0's plane appears a second time at the chunk's first plane: the
// flush re-sends the chunk it padded, since nothing clears the pool behind it.
static int
verify_planes_in_padded_chunk(const struct test_shard_sink* sink)
{
  // With CODEC_NONE and no alignment, a shard's payload is its raw chunk data
  // starting at offset 0.
  const uint16_t* el = (const uint16_t*)sink->writers[0][1].buf;
  for (uint64_t i = 0; i < PLANE_ELEMS; ++i) {
    CHECK(Fail, el[i] == plane_fill(0));
    CHECK(Fail, el[PLANE_ELEMS + i] == plane_fill(1));
    CHECK(Fail, el[2 * PLANE_ELEMS + i] == 0); // padding
  }
  return 0;
Fail:
  log_error("  the planes are not where the shape says they are");
  return 1;
}

static int
run_shape_case(const struct shape_case* c)
{
  log_info("=== %s ===", c->name);

  struct test_shard_sink sink;
  test_sink_init(&sink, OPENABLE_SHARDS, SHARD_CAP);

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
    .epochs_per_batch = c->epochs_per_batch,
    .metadata_update_interval_s = 3600.0f,
  };

  uint16_t* data = NULL;
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  // One plane, so a case can stop partway through a chunk and make the flush
  // pad the rest.
  const size_t plane_bytes = PLANE_ELEMS * sizeof(uint16_t);
  data = (uint16_t*)malloc(plane_bytes);
  CHECK(Fail, data);

  struct writer* w = tile_stream_cpu_writer(s);
  CHECK(Fail, sink.update_append_count == 0);

  for (int round = 0; round < c->n_rounds; ++round) {
    const struct flush_round* r = &c->rounds[round];
    for (uint64_t i = 0; i < PLANE_ELEMS; ++i)
      data[i] = plane_fill(round);
    for (int i = 0; i < r->planes; ++i) {
      struct slice sl = { .beg = data, .end = (const char*)data + plane_bytes };
      CHECK(Fail, writer_append(w, sl).error == 0);
    }

    struct writer_result fr = writer_flush(w);
    log_info("  round %d: err=%d updates=%d shape0=%llu",
             round,
             fr.error,
             sink.update_append_count,
             (unsigned long long)sink.last_append_size0);
    CHECK(Fail, (fr.error != 0) == r->expect_error);
    CHECK(Fail, sink.update_append_count == round + 1);
    CHECK(Fail, sink.last_append_size0 == r->expect_shape0);
  }

  if (c->verify)
    CHECK(Fail, c->verify(&sink) == 0);

  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  const struct shape_case cases[] = {
    // One chunk per plane: four fill and close both openable shards, and a
    // fifth waits in the batch needing a third shard at flush, which cannot be
    // opened. The failing flush still names the four chunks that reached
    // closed-out shards, and does not claim the fifth.
    { .name = "shape_catches_up_after_flush_error",
      .chunk_size = 1,
      .epochs_per_batch = 2,
      .n_rounds = 1,
      .rounds = { { .planes = 5, .expect_error = 1, .expect_shape0 = 4 } } },
    // One plane leaves the chunk two thirds full and the flush pads the rest.
    // A clean stream still reports the plane appended, not the padding.
    { .name = "shape_exact_on_clean_flush",
      .chunk_size = 3,
      .epochs_per_batch = 1,
      .n_rounds = 1,
      .rounds = { { .planes = 1, .expect_error = 0, .expect_shape0 = 1 } } },
    // The first flush closes shard 0 holding one chunk, leaving its second slot
    // empty. The next two chunks go to shard 1, at append positions 4 through
    // 7, so the shape has to reach 8 to cover them.
    { .name = "shape_covers_chunks_after_flush",
      .chunk_size = 2,
      .epochs_per_batch = 1,
      .n_rounds = 2,
      .rounds = { { .planes = 2, .expect_error = 0, .expect_shape0 = 2 },
                  { .planes = 4, .expect_error = 0, .expect_shape0 = 8 } } },
    // The first flush pads a chunk as well as closing the shard. The plane
    // appended after it lands in chunk 2, one plane in, so its append position
    // is 7 and the shape has to reach 8. It stops there: the chunk's third
    // plane is padding, not data.
    { .name = "shape_covers_padded_chunk_after_flush",
      .chunk_size = 3,
      .epochs_per_batch = 1,
      .n_rounds = 2,
      .rounds = { { .planes = 1, .expect_error = 0, .expect_shape0 = 1 },
                  { .planes = 1, .expect_error = 0, .expect_shape0 = 8 } },
      .verify = verify_planes_in_padded_chunk },
  };

  int err = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    err |= run_shape_case(&cases[i]);
  return err;
}
