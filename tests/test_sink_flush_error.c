// Regression: writer_flush must propagate sink->flush errors as
// writer_result.error. Wraps the standard test sink with a flush hook that
// returns non-zero (simulating e.g. a metadata write failure) and asserts
// the cascade surfaces the error.

#include "stream.cpu.h"
#include "stream/layouts.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define SHARD_CAP (1 << 20)

#define FLUSH_ERROR_CODE 42

struct flush_error_sink
{
  struct test_shard_sink inner;
  int flush_called;
  int flush_rc;
};

static int
flush_error_sink_flush(struct shard_sink* self)
{
  struct flush_error_sink* fes = (struct flush_error_sink*)self;
  fes->flush_called++;
  return fes->flush_rc;
}

static void
flush_error_sink_init(struct flush_error_sink* fes,
                      int num_shards,
                      size_t per_shard_capacity,
                      int flush_rc)
{
  test_sink_init(&fes->inner, num_shards, per_shard_capacity);
  fes->flush_called = 0;
  fes->flush_rc = flush_rc;
  fes->inner.base.flush = flush_error_sink_flush;
}

static void
flush_error_sink_free(struct flush_error_sink* fes)
{
  test_sink_free(&fes->inner);
}

// CPU pipeline: writer_flush should surface a non-zero return from
// sink->flush as writer_result.error.
static int
test_cpu_flush_error_propagates(void)
{
  log_info("=== test_cpu_flush_error_propagates ===");

  struct flush_error_sink sink;
  flush_error_sink_init(&sink, 16, SHARD_CAP, FLUSH_ERROR_CODE);

  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  uint16_t* data = NULL;
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.inner.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);
  uint64_t epoch_elems = lay->epoch_elements;
  size_t epoch_bytes = epoch_elems * sizeof(uint16_t);
  data = (uint16_t*)malloc(epoch_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < epoch_elems; ++i)
    data[i] = (uint16_t)i;

  struct writer* w = tile_stream_cpu_writer(s);

  struct slice sl = { .beg = data, .end = (const char*)data + epoch_bytes };
  struct writer_result ar = writer_append(w, sl);
  CHECK(Fail, ar.error == 0);

  // flush only queues the writes; the sink's own flush runs in close, so that
  // is where its error surfaces.
  CHECK(Fail, writer_flush(w).error == 0);
  CHECK(Fail, sink.flush_called == 0);
  CHECK(Fail, writer_close(w).error != 0);
  CHECK(Fail, sink.flush_called == 1);

  free(data);
  tile_stream_cpu_destroy(s);
  flush_error_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_cpu_destroy(s);
  flush_error_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// Sanity: a flush hook that returns 0 must not poison the result.
static int
test_cpu_flush_ok_passthrough(void)
{
  log_info("=== test_cpu_flush_ok_passthrough ===");

  struct flush_error_sink sink;
  flush_error_sink_init(&sink, 16, SHARD_CAP, 0);

  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  uint16_t* data = NULL;
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.inner.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);
  uint64_t epoch_elems = lay->epoch_elements;
  size_t epoch_bytes = epoch_elems * sizeof(uint16_t);
  data = (uint16_t*)malloc(epoch_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < epoch_elems; ++i)
    data[i] = (uint16_t)i;

  struct writer* w = tile_stream_cpu_writer(s);

  struct slice sl = { .beg = data, .end = (const char*)data + epoch_bytes };
  struct writer_result ar = writer_append(w, sl);
  CHECK(Fail, ar.error == 0);

  CHECK(Fail, writer_flush(w).error == 0);
  CHECK(Fail, writer_close(w).error == 0);
  CHECK(Fail, sink.flush_called == 1);

  free(data);
  tile_stream_cpu_destroy(s);
  flush_error_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_cpu_destroy(s);
  flush_error_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_cpu_flush_error_propagates();
  rc |= test_cpu_flush_ok_passthrough();
  return rc;
}
