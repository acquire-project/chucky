#include "bench_util.h"
#include "compress.h"
#include "dimension.h"
#include "lod.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "zarr_sink.h"

#include <stdio.h>
#include <string.h>

// --- Throughput helpers ---

double
gb_per_s(double bytes, double ms)
{
  if (ms <= 0)
    return 0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// --- Discard shard_sink for benchmarks ---

static int
discard_shard_write(struct shard_writer* self,
                    uint64_t offset,
                    const void* beg,
                    const void* end)
{
  (void)offset;
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  w->parent->total_bytes += nbytes;
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

void
discard_shard_sink_init(struct discard_shard_sink* s)
{
  *s = (struct discard_shard_sink){
    .base = { .open = discard_shard_open },
  };
  s->writer = (struct discard_shard_writer){
    .base = { .write = discard_shard_write,
              .finalize = discard_shard_finalize },
    .parent = s,
  };
}

// --- Metering shard_sink wrapper ---

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
metering_write_direct(struct shard_writer* self,
                      uint64_t offset,
                      const void* beg,
                      const void* end)
{
  struct metering_writer* w = (struct metering_writer*)self;
  platform_toc(&w->parent->clock);
  int rc = w->inner->write_direct(w->inner, offset, beg, end);
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
      ms->writers[i].base.write_direct =
        inner->write_direct ? metering_write_direct : NULL;
      return &ms->writers[i].base;
    }
  }
  return NULL;
}

void
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

// --- Report + pipeline helpers ---

void
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
    print_report("  %-12s %8.2f %8.2f %10.2f %10.2f",
                 m->name,
                 avg_gbs,
                 best_gbs,
                 avg_ms,
                 (double)m->best_ms);
  } else {
    print_report(
      "  %-12s %8.2f %8s %10.2f %10s", m->name, avg_gbs, "-", avg_ms, "-");
  }
}

void
log_bench_header(const struct tile_stream_gpu* s,
                 size_t total_bytes,
                 size_t total_elements)
{
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;

  print_report("  total:       %.2f GiB (%zu elements, %zu epochs)",
               (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
               total_elements,
               num_epochs);
  print_report(
    "  tile:        %lu elements = %lu KiB  (stride=%lu)",
    (unsigned long)s->layout.tile_elements,
    (unsigned long)(s->layout.tile_stride * s->config.bytes_per_element / 1024),
    (unsigned long)s->layout.tile_stride);
  print_report("  epoch:       %lu slots, %lu MiB pool",
               (unsigned long)s->layout.tiles_per_epoch,
               (unsigned long)(s->layout.tile_pool_bytes / (1024 * 1024)));
  if (s->config.codec != CODEC_NONE)
    print_report("  compress:    max_output=%zu comp_pool=%zu MiB",
                 s->codec.max_output_size,
                 (s->codec.batch_size * s->codec.max_output_size) /
                   (1024 * 1024));
}

void
print_bench_report(const struct tile_stream_gpu* s,
                   const struct sink_stats* ss,
                   size_t total_bytes,
                   size_t total_elements,
                   float wall_s,
                   float init_s,
                   float flush_s,
                   size_t flush_pending_bytes)
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

  print_report("");
  print_report("  --- Benchmark Results ---");
  print_report("  Input:        %.2f GiB (%zu elements)",
               (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
               total_elements);
  print_report("  Compressed:   %.2f GiB (ratio: %.3f)",
               (double)ss->total_bytes / (1024.0 * 1024.0 * 1024.0),
               comp_ratio);
  print_report("  Tiles:        %zu (%zu/epoch x %zu epochs)",
               total_tiles,
               (size_t)s->layout.tiles_per_epoch,
               num_epochs);

  print_report("");
  print_report("  %-12s %8s %8s %10s %10s",
               "Stage",
               "avg GB/s",
               "best GB/s",
               "avg ms",
               "best ms");

  print_metric_row(&m.memcpy);
  print_metric_row(&m.h2d);
  print_metric_row(&m.scatter);
  print_metric_row(&m.lod_gather);
  print_metric_row(&m.lod_reduce);
  print_metric_row(&m.lod_dim0_fold);
  print_metric_row(&m.lod_morton_tile);
  print_metric_row(&m.compress);
  print_metric_row(&m.aggregate);
  print_metric_row(&m.d2h);
  print_metric_row(&m.sink);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  print_report("");
  print_report("  Init time:     %.3f s", (double)init_s);
  if (flush_pending_bytes > 0 && flush_s > 0) {
    double flush_gib =
      ((double)flush_pending_bytes / (1024.0 * 1024.0 * 1024.0)) /
      (double)flush_s;
    print_report("  Flush time:    %.3f s (%.2f GiB/s)", (double)flush_s,
                 flush_gib);
  } else {
    print_report("  Flush time:    %.3f s", (double)flush_s);
  }
  print_report("  Wall time:     %.3f s", wall_s);
  print_report("  Throughput:    %.2f GiB/s", throughput_gib);
}

// --- Reusable bench driver ---
//
// Runs a single benchmark with the given dimensions and fill function.
// When output_path is NULL, data is discarded. Otherwise:
//   - single-scale: writes to <output_path>/<array_name>/
//   - multiscale:   writes to <output_path>/multiscale/0/, .../1/, etc.

int
run_bench(const struct bench_config* cfg)
{
  const char* label = cfg->label;
  const struct dimension* dims = cfg->dims;
  uint8_t rank = cfg->rank;
  fill_fn fill = cfg->fill;
  const char* output_path = cfg->output_path;
  const char* array_name = cfg->array_name;

  print_report("=== %s ===", label);

  int is_multiscale = 0;
  int dim0_downsample = 0;
  for (uint8_t d = 0; d < rank; ++d) {
    if (dims[d].downsample) {
      is_multiscale = 1;
      if (d == 0)
        dim0_downsample = 1;
    }
  }

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
        .exclude_dim0 = dim0_downsample,
        .unbuffered = 1,
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
        .unbuffered = 1,
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
    .codec = cfg->codec,
    .shard_sink = sink,
    .reduce_method = cfg->reduce_method,
    .dim0_reduce_method = cfg->dim0_reduce_method,
    .target_min_tiles = 2048,
    .shard_alignment = output_path ? platform_page_size() : 0,
  };

  {
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(&config, &mem) == 0) {
      print_report("  GPU memory:  %.2f GiB device, %.2f GiB pinned",
                   (double)mem.device_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)mem.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    staging:   %.2f MiB   tile_pool: %.2f GiB",
                   (double)mem.staging_bytes / (1024.0 * 1024.0),
                   (double)mem.tile_pool_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    comp_pool: %.2f GiB   aggregate: %.2f GiB",
                   (double)mem.compressed_pool_bytes /
                     (1024.0 * 1024.0 * 1024.0),
                   (double)mem.aggregate_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    lod:       %.2f MiB   codec:     %.2f MiB",
                   (double)mem.lod_bytes / (1024.0 * 1024.0),
                   (double)mem.codec_bytes / (1024.0 * 1024.0));
      print_report("    tiles:     %llu/epoch, %llu total (%d LOD levels, batch=%u)",
                   (unsigned long long)mem.tiles_per_epoch,
                   (unsigned long long)mem.total_tiles,
                   mem.nlod,
                   mem.epochs_per_batch);
    }
  }

  struct platform_clock init_clock = { 0 };
  platform_toc(&init_clock);
  CHECK(Fail, tile_stream_gpu_create(&config, &s) == 0);
  float init_s = platform_toc(&init_clock);

  log_bench_header(&s, total_bytes, total_elements);
  if (is_multiscale)
    print_report("  LOD levels:  %d", s.lod.plan.nlod);

  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  CHECK(Fail, pump_data(&s.writer, total_elements, fill) == 0);

  size_t pending_bytes = 0;
  if (zsink)
    pending_bytes = zarr_sink_pending_bytes(zsink);
  if (zmsink)
    pending_bytes = zarr_multiscale_sink_pending_bytes(zmsink);

  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  if (zsink)
    zarr_sink_flush(zsink);
  if (zmsink)
    zarr_multiscale_sink_flush(zmsink);
  float flush_s = platform_toc(&flush_clock);
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
      output_path ? (struct sink_stats){ .total_bytes = meter.total_bytes }
                  : (struct sink_stats){ .total_bytes = dss.total_bytes };
    print_bench_report(
      &s, &ss, total_bytes, total_elements, wall_s, init_s, flush_s,
      pending_bytes);
  }

  print_report("  PASS");
  int rc = 0;
  goto Cleanup;

Fail:
  print_report("  FAIL");
  rc = 1;

Cleanup:
  // Flush before destroying the stream — pending write_direct jobs
  // reference pinned host buffers owned by the stream.
  if (zsink)
    zarr_sink_flush(zsink);
  if (zmsink)
    zarr_multiscale_sink_flush(zmsink);
  tile_stream_gpu_destroy(&s);
  zarr_sink_destroy(zsink);
  zarr_multiscale_sink_destroy(zmsink);
  return rc;
}

// --- CLI parsing helpers ---

// Returns the index of the matching string, or n if no match.
static int
match_option(const char* s, const char* const* options, int n)
{
  for (int i = 0; i < n; ++i)
    if (strcmp(s, options[i]) == 0)
      return i;
  return n;
}

static fill_fn
parse_fill(const char* s)
{
  static const char* const names[] = { "xor", "zeros", "thirds" };
  static const fill_fn fns[] = { fill_xor, fill_zeros, fill_thirds };
  int i = match_option(s, names, 3);
  if (i < 3)
    return fns[i];
  fprintf(stderr, "Unknown fill: %s (expected xor, zeros, thirds)\n", s);
  return NULL;
}

static int
parse_codec(const char* s, enum compression_codec* out)
{
  static const char* const names[] = { "none", "lz4", "zstd" };
  static const enum compression_codec vals[] = { CODEC_NONE,
                                                 CODEC_LZ4,
                                                 CODEC_ZSTD };
  int i = match_option(s, names, 3);
  if (i < 3) {
    *out = vals[i];
    return 1;
  }
  fprintf(stderr, "Unknown codec: %s (expected none, lz4, zstd)\n", s);
  return 0;
}

static int
parse_reduce(const char* s, enum lod_reduce_method* out)
{
  static const char* const names[] = { "mean",   "min",     "max",
                                       "median", "max_sup", "min_sup" };
  static const enum lod_reduce_method vals[] = {
    lod_reduce_mean,
    lod_reduce_min,
    lod_reduce_max,
    lod_reduce_median,
    lod_reduce_max_suppressed,
    lod_reduce_min_suppressed,
  };
  int i = match_option(s, names, 6);
  if (i < 6) {
    *out = vals[i];
    return 1;
  }
  fprintf(stderr,
          "Unknown reduce: %s (expected mean, min, max, median, max_sup, "
          "min_sup)\n",
          s);
  return 0;
}

// --- CLI driver ---

int
bench_stream_main(int ac,
                  char* av[],
                  const char* label,
                  struct dimension* dims,
                  uint8_t rank)
{
  fill_fn fill = fill_xor;
  enum compression_codec codec = CODEC_ZSTD;
  enum lod_reduce_method reduce = lod_reduce_mean;
  const char* output_path = NULL;

  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--fill") == 0 && i + 1 < ac) {
      fill = parse_fill(av[++i]);
      if (!fill)
        return 1;
    } else if (strcmp(av[i], "--codec") == 0 && i + 1 < ac) {
      if (!parse_codec(av[++i], &codec))
        return 1;
    } else if (strcmp(av[i], "--reduce") == 0 && i + 1 < ac) {
      if (!parse_reduce(av[++i], &reduce))
        return 1;
    } else if (strcmp(av[i], "-o") == 0 && i + 1 < ac) {
      output_path = av[++i];
    } else {
      fprintf(stderr, "Unknown option: %s\n", av[i]);
      fprintf(stderr,
              "Usage: %s [--fill xor|zeros|thirds] [--codec none|lz4|zstd] "
              "[--reduce mean|min|max|median|max_sup|min_sup] [-o path]\n",
              av[0]);
      return 1;
    }
  }

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  int need_xor = (fill == fill_xor);
  if (need_xor)
    xor_pattern_init(dims, rank, 16);

  struct bench_config cfg = {
    .label = label,
    .dims = dims,
    .rank = rank,
    .fill = fill,
    .output_path = output_path,
    .array_name = label,
    .codec = codec,
    .reduce_method = reduce,
    .dim0_reduce_method = reduce == lod_reduce_median ? lod_reduce_max : reduce,
  };
  dims_print(dims, rank);
  ecode = run_bench(&cfg);

  if (need_xor)
    xor_pattern_free();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  cuCtxDestroy(ctx);
  return 1;
}
