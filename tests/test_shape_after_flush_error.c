// Regression (#175): when a flush fails, the shards finalized since the last
// periodic metadata update are on disk but a reader cannot see them unless the
// shape is written. The shape is derived from finalized shards, not from the
// append cursor, so it is truthful on the failing path and worth writing.

#include "stream.cpu.h"
#include "stream/layouts.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define SHARD_CAP (1 << 20)

// Two shards along the append dim can be opened; a third open fails, which is
// a delivery failure with no sink IO error behind it.
#define OPENABLE_SHARDS 2

struct shape_case
{
  const char* name;
  uint64_t chunk_size; // along the append dim
  uint32_t epochs_per_batch;
  int planes; // planes handed over before the flush
  int expect_flush_error;
  uint64_t expect_shape0;
};

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

  // A long interval keeps the periodic update from firing, so any recorded
  // update is the one the final flush wrote.
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
  const uint64_t plane_elems = 4 * 4;
  const size_t plane_bytes = plane_elems * sizeof(uint16_t);
  data = (uint16_t*)calloc(plane_elems, sizeof(uint16_t));
  CHECK(Fail, data);

  struct writer* w = tile_stream_cpu_writer(s);
  for (int i = 0; i < c->planes; ++i) {
    struct slice sl = { .beg = data, .end = (const char*)data + plane_bytes };
    CHECK(Fail, writer_append(w, sl).error == 0);
  }
  CHECK(Fail, sink.update_append_count == 0);

  struct writer_result fr = writer_flush(w);
  CHECK(Fail, (fr.error != 0) == c->expect_flush_error);

  log_info("  updates=%d shape0=%llu",
           sink.update_append_count,
           (unsigned long long)sink.last_append_size0);
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == c->expect_shape0);

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
      .planes = 5,
      .expect_flush_error = 1,
      .expect_shape0 = 4 },
    // One plane leaves the chunk two thirds full and the flush pads the rest.
    // A clean stream still reports the plane appended, not the padding.
    { .name = "shape_exact_on_clean_flush",
      .chunk_size = 3,
      .epochs_per_batch = 1,
      .planes = 1,
      .expect_flush_error = 0,
      .expect_shape0 = 1 },
  };

  int err = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    err |= run_shape_case(&cases[i]);
  return err;
}
