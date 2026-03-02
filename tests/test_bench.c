#include "compress.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_data.h"
#include "zarr_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- Throughput helpers ---

static double
gb_per_s(double bytes, double ms)
{
  if (ms <= 0)
    return 0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// --- Discard shard_sink for benchmarks ---

struct discard_shard_writer
{
  struct shard_writer base;
  struct discard_shard_sink* parent;
};

struct discard_shard_sink
{
  struct shard_sink base;
  struct discard_shard_writer writer;
  size_t total_bytes;
  size_t shards_finalized;
  struct stream_metric sink;
  struct platform_clock clock;
};

static int
discard_shard_write(struct shard_writer* self,
                    uint64_t offset,
                    const void* beg,
                    const void* end)
{
  (void)offset;
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  platform_toc(&w->parent->clock);
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  w->parent->total_bytes += nbytes;
  accumulate_metric_ms(&w->parent->sink,
                       (float)(platform_toc(&w->parent->clock) * 1000.0),
                       nbytes);
  return 0;
}

static int
discard_shard_finalize(struct shard_writer* self)
{
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  w->parent->shards_finalized++;
  return 0;
}

static struct shard_writer*
discard_shard_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct discard_shard_sink* s = (struct discard_shard_sink*)self;
  return &s->writer.base;
}

static void
discard_shard_sink_init(struct discard_shard_sink* s)
{
  *s = (struct discard_shard_sink){
    .base = { .open = discard_shard_open },
    .sink = { .name = "Sink", .best_ms = 1e30f },
  };
  s->writer = (struct discard_shard_writer){
    .base = { .write = discard_shard_write,
              .finalize = discard_shard_finalize },
    .parent = s,
  };
}

// --- Metering shard_sink wrapper ---
//
// Wraps any shard_sink, timing each write call and accumulating bytes.

#define METER_MAX_WRITERS 32

struct metering_writer
{
  struct shard_writer base;
  struct shard_writer* inner;
  struct metering_sink* parent;
  int in_use;
};

struct metering_sink
{
  struct shard_sink base;
  struct shard_sink* inner;
  struct metering_writer writers[METER_MAX_WRITERS];
  size_t total_bytes;
  struct stream_metric metric;
  struct platform_clock clock;
};

static int
metering_write(struct shard_writer* self,
               uint64_t offset,
               const void* beg,
               const void* end)
{
  struct metering_writer* w = (struct metering_writer*)self;
  platform_toc(&w->parent->clock);
  int rc = w->inner->write(w->inner, offset, beg, end);
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  w->parent->total_bytes += nbytes;
  accumulate_metric_ms(&w->parent->metric,
                       (float)(platform_toc(&w->parent->clock) * 1000.0),
                       nbytes);
  return rc;
}

static int
metering_finalize(struct shard_writer* self)
{
  struct metering_writer* w = (struct metering_writer*)self;
  int rc = w->inner->finalize(w->inner);
  w->in_use = 0;
  w->inner = NULL;
  return rc;
}

static struct shard_writer*
metering_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct metering_sink* ms = (struct metering_sink*)self;
  struct shard_writer* inner = ms->inner->open(ms->inner, level, shard_index);
  if (!inner)
    return NULL;
  for (int i = 0; i < METER_MAX_WRITERS; ++i) {
    if (!ms->writers[i].in_use) {
      ms->writers[i].in_use = 1;
      ms->writers[i].inner = inner;
      return &ms->writers[i].base;
    }
  }
  return NULL;
}

static void
metering_sink_init(struct metering_sink* ms, struct shard_sink* inner)
{
  *ms = (struct metering_sink){
    .base = { .open = metering_open },
    .inner = inner,
    .metric = { .name = "Sink", .best_ms = 1e30f },
  };
  for (int i = 0; i < METER_MAX_WRITERS; ++i) {
    ms->writers[i] = (struct metering_writer){
      .base = { .write = metering_write, .finalize = metering_finalize },
      .parent = ms,
    };
  }
}

// --- Small compressed+shard smoke test ---

static int
test_compressed_small(void)
{
  log_info("=== test_compressed_small ===");

  const struct dimension dims[] = {
    { .size = 40, .tile_size = 4, .tiles_per_shard = 5 },
    { .size = 2048, .tile_size = 256, .tiles_per_shard = 4 },
    { .size = 2048, .tile_size = 512, .tiles_per_shard = 2 },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3 },
  };
  const size_t total_elements = dim_total_elements(dims, 4);

  struct tile_stream_gpu s = { 0 };
  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 8 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = &dss.base,
  };

  CHECK(Fail, tile_stream_gpu_create(&config, &s));

  const size_t num_epochs =
    (total_elements + s.layout.epoch_elements - 1) / s.layout.epoch_elements;
  log_info("  total: %zu elements, %zu epochs", total_elements, num_epochs);

  CHECK(Fail, pump_data(&s.writer, total_elements, fill_zeros) == 0);

  CHECK(Fail, s.cursor == total_elements);
  log_info("  shards finalized: %zu, total bytes: %zu",
           dss.shards_finalized,
           dss.total_bytes);

  tile_stream_gpu_destroy(&s);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(&s);
  log_error("  FAIL");
  return 1;
}

// --- Report + pipeline helpers ---

static void
print_metric_row(const struct stream_metric* m)
{
  if (m->count <= 0)
    return;
  const int N = m->count;
  double avg_ms = (double)m->ms / N;
  double bytes_per = m->total_bytes / N;
  double avg_gbs = gb_per_s(m->total_bytes, (double)m->ms);
  int has_best = m->best_ms < 1e29f;

  if (has_best) {
    double best_gbs = gb_per_s(bytes_per, (double)m->best_ms);
    log_info("  %-12s %8.2f %8.2f %10.2f %10.2f",
             m->name,
             avg_gbs,
             best_gbs,
             avg_ms,
             (double)m->best_ms);
  } else {
    log_info(
      "  %-12s %8.2f %8s %10.2f %10s", m->name, avg_gbs, "-", avg_ms, "-");
  }
}

struct sink_stats
{
  size_t total_bytes;
  const struct stream_metric* sink;
};

static void
log_bench_header(const struct tile_stream_gpu* s,
                 size_t total_bytes,
                 size_t total_elements)
{
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;

  log_info("  total:       %.2f GiB (%zu elements, %zu epochs)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements,
           num_epochs);
  log_info(
    "  tile:        %lu elements = %lu KiB  (stride=%lu)",
    (unsigned long)s->layout.tile_elements,
    (unsigned long)(s->layout.tile_stride * s->config.bytes_per_element / 1024),
    (unsigned long)s->layout.tile_stride);
  log_info("  epoch:       %lu slots, %lu MiB pool",
           (unsigned long)s->layout.tiles_per_epoch,
           (unsigned long)(s->layout.tile_pool_bytes / (1024 * 1024)));
  if (s->config.codec != CODEC_NONE)
    log_info("  compress:    max_output=%zu comp_pool=%zu MiB",
             s->codec.max_output_size,
             (s->codec.batch_size * s->codec.max_output_size) / (1024 * 1024));
}

static void
print_bench_report(const struct tile_stream_gpu* s,
                   const struct sink_stats* ss,
                   size_t total_bytes,
                   size_t total_elements,
                   float wall_s)
{
  struct stream_metrics m = tile_stream_gpu_get_metrics(s);
  const size_t tile_bytes = s->layout.tile_stride * s->config.bytes_per_element;
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;
  const size_t total_tiles = num_epochs * s->layout.tiles_per_epoch;
  const size_t total_decompressed = total_tiles * tile_bytes;
  const double comp_ratio =
    total_decompressed > 0
      ? (double)ss->total_bytes / (double)total_decompressed
      : 0.0;

  log_info("");
  log_info("  --- Benchmark Results ---");
  log_info("  Input:        %.2f GiB (%zu elements)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements);
  log_info("  Compressed:   %.2f GiB (ratio: %.3f)",
           (double)ss->total_bytes / (1024.0 * 1024.0 * 1024.0),
           comp_ratio);
  log_info("  Tiles:        %zu (%zu/epoch x %zu epochs)",
           total_tiles,
           (size_t)s->layout.tiles_per_epoch,
           num_epochs);

  log_info("");
  log_info("  %-12s %8s %8s %10s %10s",
           "Stage",
           "avg GB/s",
           "best GB/s",
           "avg ms",
           "best ms");

  print_metric_row(&m.memcpy);
  print_metric_row(&m.h2d);
  print_metric_row(&m.scatter);
  print_metric_row(&m.lod_scatter);
  print_metric_row(&m.lod_reduce);
  print_metric_row(&m.lod_morton_tile);
  print_metric_row(&m.compress);
  print_metric_row(&m.aggregate);
  print_metric_row(&m.d2h);
  print_metric_row(ss->sink);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  log_info("");
  log_info("  Wall time:     %.3f s", wall_s);
  log_info("  Throughput:    %.2f GiB/s", throughput_gib);
}

// --- Reusable bench driver ---
//
// Runs a single benchmark with the given dimensions and fill function.
// When output_path is NULL, data is discarded. Otherwise:
//   - single-scale: writes to <output_path>/<array_name>/
//   - multiscale:   writes to <output_path>/multiscale/0/, .../1/, etc.

static int
run_bench(const char* label,
          const struct dimension* dims,
          uint8_t rank,
          fill_fn fill,
          const char* output_path,
          const char* array_name)
{
  log_info("=== %s ===", label);

  int is_multiscale = 0;
  for (uint8_t d = 0; d < rank; ++d)
    if (dims[d].downsample)
      is_multiscale = 1;

  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  struct zarr_sink* zsink = NULL;
  struct zarr_multiscale_sink* zmsink = NULL;
  struct metering_sink meter = { 0 };
  struct shard_sink* sink = &dss.base;

  if (output_path) {
    struct shard_sink* zarr_sink_ptr = NULL;
    if (is_multiscale) {
      struct zarr_multiscale_config zcfg = {
        .store_path = output_path,
        .array_name = array_name,
        .data_type = zarr_dtype_uint16,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
        .nlod = 0,
      };
      zmsink = zarr_multiscale_sink_create(&zcfg);
      CHECK(Fail, zmsink);
      zarr_sink_ptr = zarr_multiscale_sink_as_shard_sink(zmsink);
    } else {
      struct zarr_config zcfg = {
        .store_path = output_path,
        .array_name = array_name,
        .data_type = zarr_dtype_uint16,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
      };
      zsink = zarr_sink_create(&zcfg);
      CHECK(Fail, zsink);
      zarr_sink_ptr = zarr_sink_as_shard_sink(zsink);
    }
    metering_sink_init(&meter, zarr_sink_ptr);
    sink = &meter.base;
  }

  struct tile_stream_gpu s = { 0 };
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = rank,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = sink,
  };

  {
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(&config, &mem) == 0) {
      log_info("  GPU memory:  %.2f GiB device, %.2f GiB pinned",
               (double)mem.device_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)mem.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
      log_info("    staging:   %.2f MiB   tile_pool: %.2f GiB",
               (double)mem.staging_bytes / (1024.0 * 1024.0),
               (double)mem.tile_pool_bytes / (1024.0 * 1024.0 * 1024.0));
      log_info("    comp_pool: %.2f GiB   aggregate: %.2f GiB",
               (double)mem.compressed_pool_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)mem.aggregate_bytes / (1024.0 * 1024.0 * 1024.0));
      log_info("    lod:       %.2f MiB   codec:     %.2f MiB",
               (double)mem.lod_bytes / (1024.0 * 1024.0),
               (double)mem.codec_bytes / (1024.0 * 1024.0));
      log_info("    tiles:     %llu/epoch, %llu total (%d LOD levels)",
               (unsigned long long)mem.tiles_per_epoch,
               (unsigned long long)mem.total_tiles,
               mem.nlod);
    }
  }

  CHECK(Fail, tile_stream_gpu_create(&config, &s));
  log_bench_header(&s, total_bytes, total_elements);
  if (is_multiscale)
    log_info("  LOD levels:  %d", s.lod.plan.nlod);

  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  CHECK(Fail, pump_data(&s.writer, total_elements, fill) == 0);
  float wall_s = platform_toc(&clock);

  if (s.cursor != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)s.cursor,
              (ptrdiff_t)((int64_t)s.cursor - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct sink_stats ss =
      output_path
        ? (struct sink_stats){ .total_bytes = meter.total_bytes,
                               .sink = &meter.metric }
        : (struct sink_stats){ .total_bytes = dss.total_bytes,
                               .sink = &dss.sink };
    print_bench_report(&s, &ss, total_bytes, total_elements, wall_s);
  }

  tile_stream_gpu_destroy(&s);
  if (zsink) {
    zarr_sink_flush(zsink);
    zarr_sink_destroy(zsink);
  }
  if (zmsink) {
    zarr_multiscale_sink_flush(zmsink);
    zarr_multiscale_sink_destroy(zmsink);
  }
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(&s);
  if (zsink)
    zarr_sink_destroy(zsink);
  if (zmsink)
    zarr_multiscale_sink_destroy(zmsink);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  const char* output_path = (ac > 1) ? av[1] : NULL;

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_compressed_small();
  
  #if 0
      const int downsample[] = {1,2,3};
      struct dimension dims[] = {
        { .size = 1000, .tile_size = 2,  .tiles_per_shard = 32, .name = "t" },
        { .size = 256,  .tile_size = 16, .tiles_per_shard = 8, .name = "z" },
        { .size = 256,  .tile_size = 16, .tiles_per_shard = 8, .name = "y" },
        { .size = 256,  .tile_size = 16, .tiles_per_shard = 8, .name = "x" },
        { .size = 3,    .tile_size = 1,  .tiles_per_shard = 3, .name = "c" },
      };

      xor_pattern_init(dims, 5, 16);
  #else
      // Orca Quest 2, splitting the fov into two color channels along y
      const int downsample[] = { 2, 3 };
      struct dimension dims[] = {
        { .size = 10000,  .tile_size = 16,   .tiles_per_shard = 128,.name = "t" },
        { .size = 2,     .tile_size = 1,    .tiles_per_shard = 2, .name = "c" },
        { .size = 2048,  .tile_size = 128,  .tiles_per_shard = 9, .name = "y" },
        { .size = 2304,  .tile_size = 128,  .tiles_per_shard = 9, .name = "x" },
      };

      xor_pattern_init(dims, countof(dims), 16);
  #endif
  
  if (!ecode)
        ecode |=
      run_bench("bench", dims, countof(dims), fill_xor, output_path, "single");

  for (int i = 0; i < countof(downsample); ++i)
    dims[downsample[i]].downsample = 1;

  if (!ecode)
    ecode |= run_bench(
      "bench_multiscale", dims, countof(dims), fill_xor, output_path, "multiscale");

  xor_pattern_free();
  cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  cuCtxDestroy(ctx);
  return 1;
}
