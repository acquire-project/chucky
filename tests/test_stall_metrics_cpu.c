// A count alone would pass a metric that brackets the wrong work, so each kind
// of work in the sink stage is made slow in turn: a slow fence must land in the
// wait metrics and a slow write must not (#232).

#include "platform/platform.h"
#include "stream.cpu.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include <stdlib.h>

#define HOLD_MS 20
#define SHARD_CAP (1 << 20)
#define SHARD_ALIGNMENT 4096
#define EPOCHS 12
// A shard index is never reused, so the sink needs one per closed generation.
#define SHARDS (EPOCHS / 2 + 2)
#define PLANE_ELEMS 16
#define PLANE_FILL 0xa5a5

struct hold_times
{
  int64_t fence_ns;
  int64_t write_ns;
  int64_t open_ns;
};

static struct hold_times holds;
static int fence_waits;
static uint64_t fence_seq;
static struct shard_writer* (*inner_open)(struct shard_sink*,
                                          uint8_t,
                                          uint64_t);
static int (*inner_write)(struct shard_writer*,
                          uint64_t,
                          const void*,
                          const void*);
static int (*inner_write_direct)(struct shard_writer*,
                                 uint64_t,
                                 const void*,
                                 const void*);
static int (*inner_write_from_output)(struct shard_writer*,
                                      uint64_t,
                                      const void*,
                                      const void*,
                                      struct host_output_group*);

static struct io_event
hold_record_fence(struct shard_sink* self)
{
  (void)self;
  return (struct io_event){ .seq = ++fence_seq };
}

static void
hold_wait_fence(struct shard_sink* self, struct io_event ev)
{
  (void)self;
  (void)ev;
  ++fence_waits;
  if (holds.fence_ns)
    platform_sleep_ns(holds.fence_ns);
}

static int
hold_write(struct shard_writer* self,
           uint64_t offset,
           const void* beg,
           const void* end)
{
  if (holds.write_ns)
    platform_sleep_ns(holds.write_ns);
  return inner_write(self, offset, beg, end);
}

static int
hold_write_direct(struct shard_writer* self,
                  uint64_t offset,
                  const void* beg,
                  const void* end)
{
  if (holds.write_ns)
    platform_sleep_ns(holds.write_ns);
  return inner_write_direct(self, offset, beg, end);
}

static int
hold_write_from_output(struct shard_writer* self,
                       uint64_t offset,
                       const void* beg,
                       const void* end,
                       struct host_output_group* group)
{
  if (holds.write_ns)
    platform_sleep_ns(holds.write_ns);
  return inner_write_from_output(self, offset, beg, end, group);
}

static struct shard_writer*
hold_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  if (holds.open_ns)
    platform_sleep_ns(holds.open_ns);
  struct shard_writer* w = inner_open(self, level, shard_index);
  if (w && w->write && w->write_direct && w->write_from_output &&
      w->write_direct != hold_write_direct) {
    inner_write = w->write;
    inner_write_direct = w->write_direct;
    inner_write_from_output = w->write_from_output;
    w->write = hold_write;
    w->write_direct = hold_write_direct;
    w->write_from_output = hold_write_from_output;
  }
  return w;
}

struct run_result
{
  struct stream_metrics streaming; // read before the flush
  struct stream_metrics final;
  int fence_waits_streaming;
  int write_count;
  int open_count;
};

static int
run_stream(struct hold_times h, struct run_result* out)
{
  holds = h;
  fence_waits = 0;
  fence_seq = 0;

  struct test_shard_sink sink;
  test_sink_init(&sink, SHARDS, SHARD_CAP);
  sink.shard_alignment = SHARD_ALIGNMENT;
  inner_open = sink.base.open;
  sink.base.open = hold_open;
  sink.base.record_fence = hold_record_fence;
  sink.base.wait_fence = hold_wait_fence;

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

  // One epoch per batch, and an interval of zero so the extent is published
  // every batch: both waits then happen many times in a short run.
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 1,
    .metadata_update_interval_s = 0.0f,
  };

  uint16_t* plane = NULL;
  int failed = 1;
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const size_t plane_bytes = PLANE_ELEMS * sizeof(uint16_t);
  plane = (uint16_t*)malloc(plane_bytes);
  CHECK(Fail, plane);
  for (uint64_t i = 0; i < PLANE_ELEMS; ++i)
    plane[i] = PLANE_FILL;

  struct writer* w = tile_stream_cpu_writer(s);
  for (int i = 0; i < EPOCHS; ++i) {
    struct slice sl = { .beg = plane, .end = (const char*)plane + plane_bytes };
    CHECK(Fail, writer_append(w, sl).error == 0);
  }

  out->streaming = tile_stream_cpu_get_metrics(s);
  out->fence_waits_streaming = fence_waits;

  CHECK(Fail, writer_flush(w).error == 0);
  CHECK(Fail, writer_close(w).error == 0);

  out->final = tile_stream_cpu_get_metrics(s);
  out->write_count = sink.write_count + sink.write_direct_count;
  out->open_count = sink.open_count;
  failed = 0;

Fail:
  free(plane);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  return failed;
}

// Every wait the sink was asked for while streaming is charged to exactly one
// metric. The flush's waits run after this is read, so they are left out.
static int
every_wait_is_charged(const struct run_result* r)
{
  const struct stream_metrics* m = &r->streaming;
  const int charged =
    m->footer_buffer_stall.count + m->append_extent_stall.count;
  if (charged != r->fence_waits_streaming) {
    log_error("  %d of %d waits charged to a metric",
              charged,
              r->fence_waits_streaming);
    return 0;
  }
  return 1;
}

static int
measured(const struct stream_metric* m)
{
  if (m->count <= 0) {
    log_error("  %s: no measurement arrived", m->name);
    return 0;
  }
  return 1;
}

static int
at_least_ms(const struct stream_metric* m, double ms)
{
  if (!measured(m))
    return 0;
  if (!((double)m->best_ms >= ms)) {
    log_error("  %s: fastest of %d waits was %g ms, wanted %g",
              m->name,
              m->count,
              (double)m->best_ms,
              ms);
    return 0;
  }
  return 1;
}

static int
under_ms(const struct stream_metric* m, double ms)
{
  if (!measured(m))
    return 0;
  if (!((double)m->ms < ms)) {
    log_error("  %s: %d waits totalling %g ms, wanted under %g",
              m->name,
              m->count,
              (double)m->ms,
              ms);
    return 0;
  }
  return 1;
}

// The fastest measurement is what is held to, because an average could hide a
// wait the timer missed.
static int
slow_fences_land_in_the_wait_metrics(void)
{
  log_info("=== slow fences ===");
  struct run_result r = { 0 };
  CHECK(Fail,
        run_stream((struct hold_times){ .fence_ns = HOLD_MS * 1000000LL },
                   &r) == 0);

  CHECK(Fail, every_wait_is_charged(&r));

  const double one_hold = HOLD_MS * 0.9;
  CHECK(Fail, at_least_ms(&r.streaming.footer_buffer_stall, one_hold));
  CHECK(Fail, at_least_ms(&r.streaming.append_extent_stall, one_hold));
  CHECK(Fail, at_least_ms(&r.final.flush_writes_stall, one_hold));
  return 0;
Fail:
  return 1;
}

// Writes and opens are the sink stage's other work, so a wait metric that
// brackets any of it picks up a hold here.
static int
slow_writes_stay_out_of_the_wait_metrics(void)
{
  log_info("=== slow writes ===");
  struct run_result r = { 0 };
  CHECK(Fail,
        run_stream((struct hold_times){ .write_ns = HOLD_MS * 1000000LL,
                                        .open_ns = HOLD_MS * 1000000LL },
                   &r) == 0);

  CHECK(Fail, every_wait_is_charged(&r));
  // Without this the run proves nothing: the holds have to land in the stage
  // the wait metrics are carved out of.
  const double held = (double)(r.write_count + r.open_count) * HOLD_MS;
  log_info("  %d writes and %d opens held, sink total %g ms",
           r.write_count,
           r.open_count,
           (double)r.final.sink.ms);
  CHECK(Fail, r.write_count > 0 && r.open_count > 0);
  CHECK(Fail, (double)r.final.sink.ms > held * 0.9);

  CHECK(Fail, under_ms(&r.final.footer_buffer_stall, HOLD_MS));
  CHECK(Fail, under_ms(&r.final.append_extent_stall, HOLD_MS));
  CHECK(Fail, under_ms(&r.final.flush_writes_stall, HOLD_MS));
  return 0;
Fail:
  return 1;
}

int
main(void)
{
  int failed = 0;
  failed |= slow_fences_land_in_the_wait_metrics();
  failed |= slow_writes_stay_out_of_the_wait_metrics();
  if (failed) {
    log_error("FAIL");
    return 1;
  }
  log_info("PASS");
  return 0;
}
