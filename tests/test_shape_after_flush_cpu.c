// The shape after a flush is checked here, on the CPU backend.
//
// #175: a failed flush leaves shards on disk that a reader cannot see unless
// the shape is written.
//
// #193: a flush pads the chunk the cursor stopped in and closes its shard.
// Anything appended afterwards would start past that padding and past the shard
// slots the close left empty, at append positions the caller never asked for.
// A flush therefore finalizes the stream, and every clean case here checks that
// the next append is refused and the shape stays put.

#include "stream.cpu.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include <stdlib.h>

#define SHARD_CAP (1 << 20)

// Two shards along the append dim can be opened; a third open fails, which is
// a delivery failure with no sink IO error behind it.
#define OPENABLE_SHARDS 2

#define PLANE_ELEMS 16
#define PLANE_FILL 0xa5a5

struct shape_case
{
  const char* name;
  uint64_t chunk_size; // along the append dim
  uint32_t epochs_per_batch;
  int planes; // handed over before the flush
  int expect_flush_error;
  uint64_t expect_shape0;
  // Padding the flush added past the planes, in planes. Checked against the
  // last shard's bytes, so the shape is pinned to the data rather than to
  // arithmetic.
  uint64_t expect_padding_planes;
};

// The flush pads with zeros, and the shape stops at the planes appended.
static int
verify_padding_is_zero(const struct test_shard_sink* sink,
                       int shard,
                       uint64_t planes,
                       uint64_t padding_planes)
{
  // With CODEC_NONE and no alignment, a shard's payload is its raw chunk data
  // starting at offset 0.
  const uint16_t* el = (const uint16_t*)sink->writers[0][shard].buf;
  for (uint64_t p = 0; p < planes; ++p)
    for (uint64_t i = 0; i < PLANE_ELEMS; ++i)
      CHECK(Fail, el[p * PLANE_ELEMS + i] == PLANE_FILL);
  for (uint64_t p = planes; p < planes + padding_planes; ++p)
    for (uint64_t i = 0; i < PLANE_ELEMS; ++i)
      CHECK(Fail, el[p * PLANE_ELEMS + i] == 0);
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
  // update is one the flush wrote.
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
  for (uint64_t i = 0; i < PLANE_ELEMS; ++i)
    data[i] = PLANE_FILL;

  struct writer* w = tile_stream_cpu_writer(s);
  CHECK(Fail, sink.update_append_count == 0);

  for (int i = 0; i < c->planes; ++i) {
    struct slice sl = { .beg = data, .end = (const char*)data + plane_bytes };
    CHECK(Fail, writer_append(w, sl).error == 0);
  }

  struct writer_result fr = writer_flush(w);
  // The extent is published at close, not at flush. Closing runs even after a
  // failed flush, which is what #175 is about.
  struct writer_result cr = writer_close(w);
  if (!fr.error)
    fr = cr;
  log_info("  err=%d updates=%d shape0=%llu",
           fr.error,
           sink.update_append_count,
           (unsigned long long)sink.last_append_size0);
  CHECK(Fail, (fr.error != 0) == c->expect_flush_error);
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == c->expect_shape0);

  // The flush finalized the stream: no more input, and the shape it published
  // is the last word.
  {
    struct slice sl = { .beg = data, .end = (const char*)data + plane_bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == writer_error_finished);
    CHECK(Fail, r.rest.beg == sl.beg && r.rest.end == sl.end);
  }
  // Finalizing again reports the same outcome without writing anything more.
  CHECK(Fail, sink.update_append_count == 1);
  CHECK(Fail, sink.last_append_size0 == c->expect_shape0);

  if (c->expect_padding_planes)
    CHECK(Fail,
          verify_padding_is_zero(
            &sink, 0, (uint64_t)c->planes, c->expect_padding_planes) == 0);

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
    // The shape reports the plane appended, not the padding, and the padding
    // really is zeros on disk.
    { .name = "shape_exact_on_clean_flush",
      .chunk_size = 3,
      .epochs_per_batch = 1,
      .planes = 1,
      .expect_flush_error = 0,
      .expect_shape0 = 1,
      .expect_padding_planes = 2 },
    // Whole chunks need no padding, and two of them fill the shard.
    { .name = "shape_exact_on_whole_chunks",
      .chunk_size = 2,
      .epochs_per_batch = 1,
      .planes = 4,
      .expect_flush_error = 0,
      .expect_shape0 = 4 },
  };

  int err = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    err |= run_shape_case(&cases[i]);
  return err;
}
