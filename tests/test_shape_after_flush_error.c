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

// Two shards along the append dim can be opened; the third open fails, which
// is a delivery failure with no sink IO error behind it.
#define OPENABLE_SHARDS 2

static int
test_shape_catches_up_after_flush_error(void)
{
  log_info("=== test_shape_catches_up_after_flush_error ===");

  struct test_shard_sink sink;
  test_sink_init(&sink, OPENABLE_SHARDS, SHARD_CAP);

  // One chunk per epoch along dim0, two chunks per shard, so every second
  // epoch closes out a shard.
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
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
    .epochs_per_batch = 2,
    .metadata_update_interval_s = 3600.0f,
  };

  uint16_t* data = NULL;
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const uint64_t epoch_elems = tile_stream_cpu_layout(s)->epoch_elements;
  const size_t epoch_bytes = epoch_elems * sizeof(uint16_t);
  data = (uint16_t*)calloc(epoch_elems, sizeof(uint16_t));
  CHECK(Fail, data);

  struct writer* w = tile_stream_cpu_writer(s);

  // Four epochs fill and close out both openable shards.
  const int finalized_epochs = OPENABLE_SHARDS * 2;
  for (int i = 0; i < finalized_epochs; ++i) {
    struct slice sl = { .beg = data, .end = (const char*)data + epoch_bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
  }
  CHECK(Fail, sink.update_append_count == 0);

  // A fifth epoch stays in the batch until flush, where delivering it needs a
  // third shard that cannot be opened.
  {
    struct slice sl = { .beg = data, .end = (const char*)data + epoch_bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
  }

  struct writer_result fr = writer_flush(w);
  CHECK(Fail, fr.error != 0);

  // The failing flush still names the four chunks that reached closed-out
  // shards, and does not claim the fifth.
  log_info("  updates=%d shape0=%llu",
           sink.update_append_count,
           (unsigned long long)sink.last_append_size0);
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == (uint64_t)finalized_epochs);

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

// A clean stream must still report the exact element count, not the padded
// chunk count the shard counters alone would give.
static int
test_shape_exact_on_clean_flush(void)
{
  log_info("=== test_shape_exact_on_clean_flush ===");

  struct test_shard_sink sink;
  test_sink_init(&sink, OPENABLE_SHARDS, SHARD_CAP);

  // Three elements per chunk along dim0, so a single epoch leaves the chunk
  // two thirds full and the flush pads the rest. Two chunks per shard, so that
  // same flush also closes the shard with a slot to spare.
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 3,
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
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  // One epoch is one plane, a third of a chunk along dim0.
  const uint64_t plane_elems = 4 * 4;
  data = (uint16_t*)calloc(plane_elems, sizeof(uint16_t));
  CHECK(Fail, data);

  struct writer* w = tile_stream_cpu_writer(s);
  {
    struct slice sl = { .beg = data,
                        .end = (const char*)data +
                               plane_elems * sizeof(uint16_t) };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
  }

  struct writer_result fr = writer_flush(w);
  CHECK(Fail, fr.error == 0);

  log_info("  shape0=%llu", (unsigned long long)sink.last_append_size0);
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == 1);

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
  int err = 0;
  err |= test_shape_catches_up_after_flush_error();
  err |= test_shape_exact_on_clean_flush();
  return err;
}
