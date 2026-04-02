#include "bench_util.h"
#include "dimension.h"
#include "gpu/compress.h"
#include "gpu/lod.h"
#include "gpu/prelude.cuda.h"
#include "stream.cpu.h"
#include "util/prelude.h"
#include "zarr/json_writer.h"
#include "zarr_fs_sink.h"
#include "zarr_s3_sink.h"

#include <stdio.h>
#include <stdlib.h>
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
                       nbytes,
                       0);
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
                       nbytes,
                       0);
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

// --- Backend dispatch helpers ---

struct bench_handle
{
  enum bench_backend backend;
  union
  {
    struct tile_stream_gpu* gpu;
    struct tile_stream_cpu* cpu;
  };
};

static int
bench_is_null(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? h->cpu == NULL : h->gpu == NULL;
}

static const struct tile_stream_layout*
bench_layout(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_layout(h->cpu)
                                 : tile_stream_gpu_layout(h->gpu);
}

static struct stream_metrics
bench_get_metrics(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_get_metrics(h->cpu)
                                 : tile_stream_gpu_get_metrics(h->gpu);
}

static struct writer*
bench_writer(struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_writer(h->cpu)
                                 : tile_stream_gpu_writer(h->gpu);
}

static uint64_t
bench_cursor(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_cursor(h->cpu)
                                 : tile_stream_gpu_cursor(h->gpu);
}

static void
bench_destroy(struct bench_handle* h)
{
  if (h->backend == BENCH_CPU)
    tile_stream_cpu_destroy(h->cpu);
  else
    tile_stream_gpu_destroy(h->gpu);
}

// --- Report + pipeline helpers ---

void
print_metric_row(const struct stream_metric* m)
{
  if (m->count <= 0)
    return;
  const int N = m->count;
  double avg_ms = (double)m->ms / N;
  double bytes_per = m->input_bytes / N;
  double avg_gbs = gb_per_s(m->input_bytes, (double)m->ms);
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
log_bench_header(const struct tile_stream_layout* layout,
                 enum dtype dtype,
                 struct codec_config codec,
                 size_t max_compressed_size,
                 size_t codec_batch_size,
                 size_t total_bytes,
                 size_t total_elements)
{
  const size_t num_epochs =
    (total_elements + layout->epoch_elements - 1) / layout->epoch_elements;

  print_report("  total:       %.2f GiB (%zu elements, %zu epochs)",
               (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
               total_elements,
               num_epochs);
  print_report("  chunk:       %lu elements = %lu KiB  (stride=%lu)",
               (unsigned long)layout->chunk_elements,
               (unsigned long)(layout->chunk_stride * dtype_bpe(dtype) / 1024),
               (unsigned long)layout->chunk_stride);
  print_report("  epoch:       %lu slots, %lu MiB pool",
               (unsigned long)layout->chunks_per_epoch,
               (unsigned long)(layout->chunk_pool_bytes / (1024 * 1024)));
  if (codec.id != CODEC_NONE && max_compressed_size > 0)
    print_report("  compress:    max_output=%zu comp_pool=%zu MiB",
                 max_compressed_size,
                 (codec_batch_size * max_compressed_size) / (1024 * 1024));
}

void
print_bench_report(const struct stream_metrics* metrics,
                   const struct tile_stream_layout* layout,
                   enum dtype dtype,
                   const struct sink_stats* ss,
                   size_t total_bytes,
                   size_t total_elements,
                   float wall_s,
                   float init_s,
                   float flush_s,
                   size_t flush_pending_bytes)
{
  const size_t chunk_bytes = layout->chunk_stride * dtype_bpe(dtype);
  const size_t num_epochs =
    (total_elements + layout->epoch_elements - 1) / layout->epoch_elements;
  const uint64_t chunks_per_epoch =
    ss->total_chunks ? ss->total_chunks : layout->chunks_per_epoch;
  const size_t total_chunks = num_epochs * chunks_per_epoch;
  const size_t total_decompressed = total_chunks * chunk_bytes;
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
  print_report("  Chunks:       %zu (%llu/epoch x %zu epochs)",
               total_chunks,
               (unsigned long long)chunks_per_epoch,
               num_epochs);

  print_report("");
  print_report("  %-12s %8s %8s %10s %10s",
               "Stage",
               "avg GB/s",
               "best GB/s",
               "avg ms",
               "best ms");

  print_metric_row(&metrics->memcpy);
  print_metric_row(&metrics->h2d);
  print_metric_row(&metrics->scatter);
  print_metric_row(&metrics->lod_gather);
  print_metric_row(&metrics->lod_reduce);
  print_metric_row(&metrics->lod_append_fold);
  print_metric_row(&metrics->lod_morton_chunk);
  print_metric_row(&metrics->compress);
  print_metric_row(&metrics->aggregate);
  print_metric_row(&metrics->d2h);
  print_metric_row(&metrics->sink);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  print_report("");
  print_report("  Init time:     %.3f s", (double)init_s);
  if (flush_pending_bytes > 0 && flush_s > 0) {
    double flush_gib =
      ((double)flush_pending_bytes / (1024.0 * 1024.0 * 1024.0)) /
      (double)flush_s;
    print_report(
      "  Flush time:    %.3f s (%.2f GiB/s)", (double)flush_s, flush_gib);
  } else {
    print_report("  Flush time:    %.3f s", (double)flush_s);
  }
  print_report("  Wall time:     %.3f s", wall_s);
  print_report("  Throughput:    %.2f GiB/s", throughput_gib);
}

// --- Byte-size parser: "256K", "1M", "8G" etc. ---

static size_t
parse_bytes(const char* s)
{
  char* end = NULL;
  size_t val = (size_t)strtoull(s, &end, 10);
  if (end && *end) {
    switch (*end) {
      case 'k':
      case 'K':
        val <<= 10;
        break;
      case 'm':
      case 'M':
        val <<= 20;
        break;
      case 'g':
      case 'G':
        val <<= 30;
        break;
    }
  }
  return val;
}

// (autofit adapter removed — now using
// tile_stream_{gpu,cpu}_advise_chunk_sizes)

// --- dtype helpers ---

static const char* const dtype_names[] = {
  "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "f32", "f64",
};
static const enum dtype dtype_vals[] = {
  dtype_u8,  dtype_u16, dtype_u32, dtype_u64, dtype_i8,  dtype_i16,
  dtype_i32, dtype_i64, dtype_f16, dtype_f32, dtype_f64,
};
#define NUM_DTYPES (sizeof(dtype_vals) / sizeof(dtype_vals[0]))

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
  struct dimension* dims = cfg->dims;
  uint8_t rank = cfg->rank;
  fill_fn fill = cfg->fill;
  const char* output_path = cfg->output_path;
  const char* array_name = cfg->array_name;

  print_report(
    "=== %s [%s] ===", label, cfg->backend == BENCH_CPU ? "cpu" : "gpu");

  int is_multiscale = 0;
  for (uint8_t d = 0; d < rank; ++d) {
    if (dims[d].downsample)
      is_multiscale = 1;
  }

  const enum dtype dtype = cfg->dtype ? cfg->dtype : dtype_u16;

  // --- Chunk sizing ---
  if (cfg->chunk_ratios) {
    size_t bytes_per_element = dtype_bpe(dtype);
    size_t target =
      cfg->target_chunk_bytes ? cfg->target_chunk_bytes : (1 << 20);
    size_t budget = cfg->memory_budget;

    // Auto-detect memory budget
    if (budget == 0) {
      if (cfg->backend == BENCH_GPU) {
        size_t free_mem = 0, total_mem = 0;
        if (cuMemGetInfo(&free_mem, &total_mem) == CUDA_SUCCESS &&
            free_mem > 0) {
          budget = (size_t)((double)free_mem * 0.8);
          print_report(
            "  auto-detect: %.2f GiB free GPU memory (restrict to <80%%)",
            (double)free_mem / (1024.0 * 1024.0 * 1024.0));
        }
      } else {
        size_t avail = platform_available_memory();
        if (avail > 0) {
          budget = (size_t)((double)avail * 0.8);
          print_report(
            "  auto-detect: %.2f GiB available RAM (restrict to <80%%)",
            (double)avail / (1024.0 * 1024.0 * 1024.0));
        }
      }
    }

    // Auto-fit: try shrinking chunks to fit budget
    int fitted = 0;
    if (budget > 0) {
      struct discard_shard_sink fit_dss;
      discard_shard_sink_init(&fit_dss);
      struct tile_stream_configuration fit_config = {
        .buffer_capacity_bytes = 128 << 20,
        .dtype = dtype,
        .rank = rank,
        .dimensions = dims,
        .codec = cfg->codec,
        .reduce_method = cfg->reduce_method,
        .append_reduce_method = cfg->append_reduce_method,
        .target_batch_chunks = 2048,
        .shard_alignment =
          (output_path || cfg->s3_bucket) ? platform_page_size() : 0,
      };
      int advise_ok;
      if (cfg->backend == BENCH_GPU) {
        advise_ok = tile_stream_gpu_advise_chunk_sizes(
          &fit_config, target, cfg->chunk_ratios, budget);
      } else {
        advise_ok = tile_stream_cpu_advise_chunk_sizes(
          &fit_config, target, cfg->chunk_ratios, budget);
      }
      if (advise_ok == 0) {
        fitted = 1;
        uint64_t vol = 1;
        for (uint8_t d = 0; d < rank; ++d)
          vol *= dims[d].chunk_size;
        print_report("  auto-fit: %zu bytes/chunk",
                     (size_t)(vol * bytes_per_element));
      } else {
        print_report("  auto-fit: WARNING — no chunk size fits in budget");
      }
    }

    // Fallback: just budget chunk sizes without memory constraint
    if (!fitted)
      dims_budget_chunk_bytes(
        dims, rank, target, bytes_per_element, cfg->chunk_ratios);

    // Set shard counts after chunk sizing
    if (cfg->shard_counts)
      dims_set_shard_counts(dims, rank, cfg->shard_counts);
  }

  dims_print(dims, rank);

  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * dtype_bpe(dtype);

  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  struct zarr_fs_sink* zsink = NULL;
  struct zarr_fs_multiscale_sink* zmsink = NULL;
  struct zarr_s3_sink* zs3sink = NULL;
  struct zarr_s3_multiscale_sink* zs3msink = NULL;
  struct metering_sink meter = { 0 };
  struct shard_sink* sink = &dss.base;
  struct bench_handle h = { .backend = cfg->backend };

  if (cfg->s3_bucket) {
    struct shard_sink* zarr_sink_ptr = NULL;
    if (is_multiscale) {
      struct zarr_s3_multiscale_config zcfg = {
        .bucket = cfg->s3_bucket,
        .prefix = cfg->s3_prefix ? cfg->s3_prefix : label,
        .array_name = array_name,
        .region = cfg->s3_region,
        .endpoint = cfg->s3_endpoint,
        .throughput_gbps = cfg->s3_throughput_gbps,
        .data_type = dtype,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
        .nlod = 0,
        .codec = cfg->codec,
      };
      zs3msink = zarr_s3_multiscale_sink_create(&zcfg);
      CHECK(Fail, zs3msink);
      zarr_sink_ptr = zarr_s3_multiscale_sink_as_shard_sink(zs3msink);
    } else {
      struct zarr_s3_config zcfg = {
        .bucket = cfg->s3_bucket,
        .prefix = cfg->s3_prefix ? cfg->s3_prefix : label,
        .array_name = array_name,
        .region = cfg->s3_region,
        .endpoint = cfg->s3_endpoint,
        .throughput_gbps = cfg->s3_throughput_gbps,
        .data_type = dtype,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
        .codec = cfg->codec,
      };
      zs3sink = zarr_s3_sink_create(&zcfg);
      CHECK(Fail, zs3sink);
      zarr_sink_ptr = zarr_s3_sink_as_shard_sink(zs3sink);
    }
    metering_sink_init(&meter, zarr_sink_ptr);
    sink = &meter.base;
  } else if (output_path) {
    struct shard_sink* zarr_fs_sink_ptr = NULL;
    if (is_multiscale) {
      struct zarr_multiscale_config zcfg = {
        .store_path = output_path,
        .array_name = array_name,
        .data_type = dtype,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
        .nlod = 0,
        .unbuffered = 1,
      };
      zmsink = zarr_fs_multiscale_sink_create(&zcfg);
      CHECK(Fail, zmsink);
      zarr_fs_sink_ptr = zarr_fs_multiscale_sink_as_shard_sink(zmsink);
    } else {
      struct zarr_config zcfg = {
        .store_path = output_path,
        .array_name = array_name,
        .data_type = dtype,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
        .unbuffered = 1,
      };
      zsink = zarr_fs_sink_create(&zcfg);
      CHECK(Fail, zsink);
      zarr_fs_sink_ptr = zarr_fs_sink_as_shard_sink(zsink);
    }
    metering_sink_init(&meter, zarr_fs_sink_ptr);
    sink = &meter.base;
  }

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .dtype = dtype,
    .rank = rank,
    .dimensions = dims,
    .codec = cfg->codec,
    .reduce_method = cfg->reduce_method,
    .append_reduce_method = cfg->append_reduce_method,
    .target_batch_chunks = 2048,
    .shard_alignment =
      (output_path || cfg->s3_bucket) ? platform_page_size() : 0,
  };

  uint64_t est_total_chunks = 0;

  if (cfg->backend == BENCH_GPU) {
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(&config, &mem) == 0) {
      est_total_chunks = mem.total_chunks;
      print_report("  GPU memory:  %.2f GiB device, %.2f GiB pinned",
                   (double)mem.device_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)mem.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    staging:   %.2f MiB   chunk_pool: %.2f GiB",
                   (double)mem.staging_bytes / (1024.0 * 1024.0),
                   (double)mem.chunk_pool_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    comp_pool: %.2f GiB   aggregate: %.2f GiB",
                   (double)mem.compressed_pool_bytes /
                     (1024.0 * 1024.0 * 1024.0),
                   (double)mem.aggregate_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    lod:       %.2f MiB   codec:     %.2f MiB",
                   (double)mem.lod_bytes / (1024.0 * 1024.0),
                   (double)mem.codec_bytes / (1024.0 * 1024.0));
      print_report(
        "    chunks:    %llu/epoch, %llu total (%d LOD levels, batch=%u)",
        (unsigned long long)mem.chunks_per_epoch,
        (unsigned long long)mem.total_chunks,
        mem.nlod,
        mem.epochs_per_batch);
    }
  }

  if (cfg->backend == BENCH_CPU) {
    struct tile_stream_cpu_memory_info mem;
    if (tile_stream_cpu_memory_estimate(&config, &mem) == 0) {
      est_total_chunks = mem.total_chunks;
      print_report("  CPU memory:  %.2f GiB heap",
                   (double)mem.heap_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report("    chunk_pool: %.2f GiB   comp_pool: %.2f GiB",
                   (double)mem.chunk_pool_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)mem.compressed_pool_bytes /
                     (1024.0 * 1024.0 * 1024.0));
      print_report("    comp_sizes: %.2f MiB   aggregate: %.2f MiB",
                   (double)mem.comp_sizes_bytes / (1024.0 * 1024.0),
                   (double)mem.aggregate_bytes / (1024.0 * 1024.0));
      print_report("    lod:       %.2f MiB   shards:    %.2f MiB",
                   (double)mem.lod_bytes / (1024.0 * 1024.0),
                   (double)mem.shard_bytes / (1024.0 * 1024.0));
      print_report(
        "    chunks:    %llu/epoch, %llu total (%d LOD levels, batch=%u)",
        (unsigned long long)mem.chunks_per_epoch,
        (unsigned long long)mem.total_chunks,
        mem.nlod,
        mem.epochs_per_batch);
    }
  }

  struct platform_clock init_clock = { 0 };
  platform_toc(&init_clock);

  if (cfg->backend == BENCH_CPU)
    h.cpu = tile_stream_cpu_create(&config, sink);
  else
    h.gpu = tile_stream_gpu_create(&config, sink);
  CHECK(Fail, !bench_is_null(&h));
  float init_s = platform_toc(&init_clock);

  const struct tile_stream_layout* layout = bench_layout(&h);
  size_t max_compressed_size = 0;
  size_t codec_batch_size = 0;
  int nlod = 0;
  if (cfg->backend == BENCH_GPU) {
    struct tile_stream_status st = tile_stream_gpu_status(h.gpu);
    max_compressed_size = st.max_compressed_size;
    codec_batch_size = st.codec_batch_size;
    nlod = st.nlod;
  }

  log_bench_header(layout,
                   config.dtype,
                   config.codec,
                   max_compressed_size,
                   codec_batch_size,
                   total_bytes,
                   total_elements);
  if (is_multiscale && nlod > 0)
    print_report("  LOD levels:  %d", nlod);

  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  CHECK(Fail, pump_data(bench_writer(&h), total_elements, fill) == 0);

  size_t pending_bytes = 0;
  if (zsink)
    pending_bytes = zarr_fs_sink_pending_bytes(zsink);
  if (zmsink)
    pending_bytes = zarr_fs_multiscale_sink_pending_bytes(zmsink);

  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  if (zsink)
    zarr_fs_sink_flush(zsink);
  if (zmsink)
    zarr_fs_multiscale_sink_flush(zmsink);
  if (zs3sink)
    zarr_s3_sink_flush(zs3sink);
  if (zs3msink)
    zarr_s3_multiscale_sink_flush(zs3msink);
  float flush_s = platform_toc(&flush_clock);
  float wall_s = platform_toc(&clock);

  if (bench_cursor(&h) != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)bench_cursor(&h),
              (ptrdiff_t)((int64_t)bench_cursor(&h) - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct stream_metrics m = bench_get_metrics(&h);
    int has_output = output_path || cfg->s3_bucket;
    struct sink_stats ss =
      has_output ? (struct sink_stats){ .total_bytes = meter.total_bytes,
                                        .total_chunks = est_total_chunks }
                 : (struct sink_stats){ .total_bytes = dss.total_bytes,
                                        .total_chunks = est_total_chunks };
    print_bench_report(&m,
                       layout,
                       config.dtype,
                       &ss,
                       total_bytes,
                       total_elements,
                       wall_s,
                       init_s,
                       flush_s,
                       pending_bytes);

    if (cfg->json_output) {
      const size_t chunk_bytes = layout->chunk_stride * dtype_bpe(dtype);
      const size_t num_epochs =
        (total_elements + layout->epoch_elements - 1) / layout->epoch_elements;
      const uint64_t chunks_per_epoch =
        ss.total_chunks ? ss.total_chunks : layout->chunks_per_epoch;
      const size_t total_chunks = num_epochs * chunks_per_epoch;
      const size_t total_decompressed = total_chunks * chunk_bytes;
      const double comp_fold =
        ss.total_bytes > 0 ? (double)total_decompressed / (double)ss.total_bytes
                           : 0.0;
      const double GIB = 1024.0 * 1024.0 * 1024.0;
      double input_gib = (double)total_bytes / GIB;
      double compressed_gib = (double)ss.total_bytes / GIB;
      double throughput_gib = wall_s > 0 ? input_gib / wall_s : 0.0;
      double throughput_out_gib = wall_s > 0 ? compressed_gib / wall_s : 0.0;

      char json_buf[8192];
      struct json_writer jw;
      jw_init(&jw, json_buf, sizeof(json_buf));

      jw_object_begin(&jw);
      jw_key(&jw, "status");
      jw_string(&jw, "pass");
      jw_key(&jw, "throughput_in_gibs");
      jw_float(&jw, throughput_gib);
      jw_key(&jw, "throughput_out_gibs");
      jw_float(&jw, throughput_out_gib);
      jw_key(&jw, "compression_fold");
      jw_float(&jw, comp_fold);
      jw_key(&jw, "input_gib");
      jw_float(&jw, input_gib);
      jw_key(&jw, "compressed_gib");
      jw_float(&jw, compressed_gib);
      jw_key(&jw, "total_chunks");
      jw_uint(&jw, total_chunks);
      jw_key(&jw, "chunks_per_epoch");
      jw_uint(&jw, chunks_per_epoch);
      jw_key(&jw, "wall_s");
      jw_float(&jw, (double)wall_s);
      jw_key(&jw, "init_s");
      jw_float(&jw, (double)init_s);
      jw_key(&jw, "flush_s");
      jw_float(&jw, (double)flush_s);

      // Per-stage metrics
      jw_key(&jw, "stages");
      jw_object_begin(&jw);
      const char* stage_names[] = {
        "memcpy",           "h2d",
        "scatter",          "lod_gather",
        "lod_reduce",       "lod_append_fold",
        "lod_morton_chunk", "compress",
        "aggregate",        "d2h",
      };
      const struct stream_metric* stage_ptrs[] = {
        &m.memcpy,           &m.h2d,
        &m.scatter,          &m.lod_gather,
        &m.lod_reduce,       &m.lod_append_fold,
        &m.lod_morton_chunk, &m.compress,
        &m.aggregate,        &m.d2h,
      };
      int nstages = sizeof(stage_ptrs) / sizeof(stage_ptrs[0]);
      for (int si = 0; si < nstages; ++si) {
        if (stage_ptrs[si]->count <= 0)
          continue;
        const struct stream_metric* sm = stage_ptrs[si];
        double avg_ms = (double)sm->ms / sm->count;
        double in_gibs = gb_per_s(sm->input_bytes, (double)sm->ms);
        double out_gibs = gb_per_s(sm->output_bytes, (double)sm->ms);
        jw_key(&jw, stage_names[si]);
        jw_object_begin(&jw);
        jw_key(&jw, "avg_ms");
        jw_float(&jw, avg_ms);
        if (sm->best_ms < 1e29f) {
          jw_key(&jw, "best_ms");
          jw_float(&jw, (double)sm->best_ms);
        }
        jw_key(&jw, "in_gibs");
        jw_float(&jw, in_gibs);
        jw_key(&jw, "out_gibs");
        jw_float(&jw, out_gibs);
        jw_object_end(&jw);
      }
      if (meter.metric.count > 0) {
        const struct stream_metric* sm = &meter.metric;
        double avg_ms = (double)sm->ms / sm->count;
        double in_gibs = gb_per_s(sm->input_bytes, (double)sm->ms);
        double out_gibs = gb_per_s(sm->output_bytes, (double)sm->ms);
        jw_key(&jw, "sink");
        jw_object_begin(&jw);
        jw_key(&jw, "avg_ms");
        jw_float(&jw, avg_ms);
        if (sm->best_ms < 1e29f) {
          jw_key(&jw, "best_ms");
          jw_float(&jw, (double)sm->best_ms);
        }
        jw_key(&jw, "in_gibs");
        jw_float(&jw, in_gibs);
        jw_key(&jw, "out_gibs");
        jw_float(&jw, out_gibs);
        jw_object_end(&jw);
      }
      jw_object_end(&jw); // stages

      jw_object_end(&jw); // root
      printf("%.*s\n", (int)jw_length(&jw), json_buf);
    }
  }

  print_report("  PASS");
  int rc = 0;
  goto Cleanup;

Fail:
  if (cfg->json_output) {
    char err_buf[64];
    struct json_writer ejw;
    jw_init(&ejw, err_buf, sizeof(err_buf));
    jw_object_begin(&ejw);
    jw_key(&ejw, "status");
    jw_string(&ejw, "error");
    jw_object_end(&ejw);
    printf("%.*s\n", (int)jw_length(&ejw), err_buf);
  }
  print_report("  FAIL");
  rc = 1;

Cleanup:
  // Flush before destroying the stream — pending write_direct jobs
  // reference pinned host buffers owned by the stream.
  if (zsink)
    zarr_fs_sink_flush(zsink);
  if (zmsink)
    zarr_fs_multiscale_sink_flush(zmsink);
  if (zs3sink)
    zarr_s3_sink_flush(zs3sink);
  if (zs3msink)
    zarr_s3_multiscale_sink_flush(zs3msink);
  bench_destroy(&h);
  zarr_fs_sink_destroy(zsink);
  zarr_fs_multiscale_sink_destroy(zmsink);
  zarr_s3_sink_destroy(zs3sink);
  zarr_s3_multiscale_sink_destroy(zs3msink);
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
  static const char* const names[] = { "xor", "zeros", "rand" };
  static const fill_fn fns[] = { fill_xor, fill_zeros, fill_rand };
  int i = match_option(s, names, 3);
  if (i < 3)
    return fns[i];
  fprintf(stderr, "Unknown fill: %s (expected xor, zeros, rand)\n", s);
  return NULL;
}

static int
parse_codec(const char* s, struct codec_config* out)
{
  static const char* const names[] = {
    "none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"
  };
  static const enum compression_codec vals[] = { CODEC_NONE,
                                                 CODEC_LZ4_NON_STANDARD,
                                                 CODEC_ZSTD,
                                                 CODEC_BLOSC_LZ4,
                                                 CODEC_BLOSC_ZSTD };
  int i = match_option(s, names, 5);
  if (i < 5) {
    out->id = vals[i];
    if (out->id == CODEC_LZ4_NON_STANDARD && out->level == 0)
      out->level = 1;
    if (codec_is_blosc(out->id) && out->level == 0)
      out->level = 3;
    return 1;
  }
  fprintf(stderr,
          "Unknown codec: %s (expected none, lz4, zstd, blosc-lz4, "
          "blosc-zstd)\n",
          s);
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

// --- CLI parsing: backend ---

static int
parse_backend(const char* s, enum bench_backend* out)
{
  static const char* const names[] = { "gpu", "cpu" };
  static const enum bench_backend vals[] = { BENCH_GPU, BENCH_CPU };
  int i = match_option(s, names, 2);
  if (i < 2) {
    *out = vals[i];
    return 1;
  }
  fprintf(stderr, "Unknown backend: %s (expected gpu, cpu)\n", s);
  return 0;
}

// --- CLI parsing: dtype ---

static int
parse_dtype(const char* s, enum dtype* out)
{
  int i = match_option(s, dtype_names, NUM_DTYPES);
  if (i < (int)NUM_DTYPES) {
    *out = dtype_vals[i];
    return 1;
  }
  fprintf(stderr,
          "Unknown dtype: %s (expected u8, u16, u32, u64, i8, i16, i32, i64, "
          "f16, f32, f64)\n",
          s);
  return 0;
}

// --- CLI driver ---

int
bench_stream_main(int ac,
                  char* av[],
                  const char* label,
                  struct dimension* dims,
                  uint8_t rank,
                  const uint8_t* chunk_ratios,
                  size_t default_chunk_bytes,
                  const uint64_t* shard_counts)
{
  fill_fn fill = fill_xor;
  struct codec_config codec = { .id = CODEC_ZSTD };
  enum lod_reduce_method reduce = lod_reduce_mean;
  const char* output_path = NULL;
  const char* s3_bucket = NULL;
  const char* s3_prefix = NULL;
  const char* s3_region = NULL;
  const char* s3_endpoint = NULL;
  double s3_throughput_gbps = 0;
  enum bench_backend backend = BENCH_GPU;
  enum dtype dtype = dtype_u16;
  size_t target_chunk_bytes = 0;
  size_t memory_budget = 0;
  uint64_t frames = 0;
  int json_output = 0;

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
    } else if (strcmp(av[i], "--backend") == 0 && i + 1 < ac) {
      if (!parse_backend(av[++i], &backend))
        return 1;
    } else if (strcmp(av[i], "--dtype") == 0 && i + 1 < ac) {
      if (!parse_dtype(av[++i], &dtype))
        return 1;
    } else if (strcmp(av[i], "--frames") == 0 && i + 1 < ac) {
      frames = (uint64_t)strtoull(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--json") == 0) {
      json_output = 1;
    } else if (strcmp(av[i], "--chunk-bytes") == 0 && i + 1 < ac) {
      target_chunk_bytes = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "--memory-budget") == 0 && i + 1 < ac) {
      memory_budget = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "-o") == 0 && i + 1 < ac) {
      output_path = av[++i];
    } else if (strcmp(av[i], "--s3-bucket") == 0 && i + 1 < ac) {
      s3_bucket = av[++i];
    } else if (strcmp(av[i], "--s3-prefix") == 0 && i + 1 < ac) {
      s3_prefix = av[++i];
    } else if (strcmp(av[i], "--s3-region") == 0 && i + 1 < ac) {
      s3_region = av[++i];
    } else if (strcmp(av[i], "--s3-endpoint") == 0 && i + 1 < ac) {
      s3_endpoint = av[++i];
    } else if (strcmp(av[i], "--s3-throughput-gbps") == 0 && i + 1 < ac) {
      s3_throughput_gbps = strtod(av[++i], NULL);
    } else {
      fprintf(stderr, "Unknown option: %s\n", av[i]);
      fprintf(stderr,
              "Usage: %s [--fill xor|zeros|rand] [--codec none|lz4|zstd] "
              "[--reduce mean|min|max|median|max_sup|min_sup] "
              "[--backend gpu|cpu] [--dtype u8|u16|...] [--frames N] "
              "[--json] [--chunk-bytes N] [--memory-budget N] [-o path] "
              "[--s3-bucket B --s3-region R --s3-endpoint E [--s3-prefix P] "
              "[--s3-throughput-gbps N]]\n",
              av[0]);
      return 1;
    }
  }

  // Override frame count (dim 0 size) if requested
  if (frames > 0)
    dims[0].size = frames;

  int ecode = 0;
  CUcontext ctx = 0;

  if (backend == BENCH_GPU) {
    CUdevice dev;
    CU(Fail, cuInit(0));
    CU(Fail, cuDeviceGet(&dev, 0));
    CU(Fail, cuCtxCreate(&ctx, 0, dev));
  }

  int need_xor = (fill == fill_xor);
  int need_rand = (fill == fill_rand);
  if (need_xor)
    xor_pattern_init(dims, rank, 16);
  if (need_rand)
    rand_pattern_init(dims, rank, 16);

  struct bench_config cfg = {
    .label = label,
    .dims = dims,
    .rank = rank,
    .fill = fill,
    .output_path = output_path,
    .array_name = label,
    .s3_bucket = s3_bucket,
    .s3_prefix = s3_prefix,
    .s3_region = s3_region,
    .s3_endpoint = s3_endpoint,
    .s3_throughput_gbps = s3_throughput_gbps,
    .codec = codec,
    .reduce_method = reduce,
    .append_reduce_method =
      reduce == lod_reduce_median ? lod_reduce_max : reduce,
    .backend = backend,
    .dtype = dtype,
    .chunk_ratios = chunk_ratios,
    .target_chunk_bytes =
      target_chunk_bytes ? target_chunk_bytes : default_chunk_bytes,
    .memory_budget = memory_budget,
    .shard_counts = shard_counts,
    .json_output = json_output,
  };
  ecode = run_bench(&cfg);

  if (need_xor)
    xor_pattern_free();
  if (need_rand)
    rand_pattern_free();

  if (backend == BENCH_GPU)
    cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  if (backend == BENCH_GPU)
    cuCtxDestroy(ctx);
  return 1;
}

// ---------------------------------------------------------------------------
// Two-stream benchmark: two GPU pipelines, interleaved append on one thread
// ---------------------------------------------------------------------------

// Interleaved pump: alternates writer_append between two writers in lockstep.
// Each call pushes one chunk of data to each writer before advancing.
static int
pump_data_interleaved(struct writer* w0,
                      struct writer* w1,
                      size_t total_elements,
                      fill_fn fill,
                      size_t bpe)
{
  const size_t nelements = 32 * 1024 * 1024; // 32M elements per chunk
  size_t alloc = nelements * (bpe > 2 ? bpe : 2);
  uint16_t* data = (uint16_t*)malloc(alloc);
  if (!data)
    return 1;

  size_t off0 = 0, off1 = 0;

  while (off0 < total_elements || off1 < total_elements) {
    // --- stream 0 ---
    if (off0 < total_elements) {
      size_t n = nelements;
      if (off0 + n > total_elements)
        n = total_elements - off0;
      fill(data, n, off0, total_elements);
      struct slice input = { .beg = data, .end = (char*)data + n * bpe };
      struct writer_result r = writer_append_wait(w0, input);
      if (r.error) {
        log_error("  stream-0 append failed at offset %zu", off0);
        free(data);
        return 1;
      }
      off0 += n;
    }

    // --- stream 1 ---
    if (off1 < total_elements) {
      size_t n = nelements;
      if (off1 + n > total_elements)
        n = total_elements - off1;
      fill(data, n, off1, total_elements);
      struct slice input = { .beg = data, .end = (char*)data + n * bpe };
      struct writer_result r = writer_append_wait(w1, input);
      if (r.error) {
        log_error("  stream-1 append failed at offset %zu", off1);
        free(data);
        return 1;
      }
      off1 += n;
    }
  }

  // flush both
  struct writer_result r0 = writer_flush(w0);
  struct writer_result r1 = writer_flush(w1);
  free(data);
  return r0.error || r1.error;
}

static int
run_bench_two_streams(const struct bench_config* cfg)
{
  struct dimension* dims = cfg->dims;
  uint8_t rank = cfg->rank;
  fill_fn fill = cfg->fill;
  const char* output_path = cfg->output_path;
  const char* array_name = cfg->array_name;

  print_report("=== %s [gpu x2] ===", cfg->label);

  const enum dtype dtype = cfg->dtype ? cfg->dtype : dtype_u16;
  const size_t bpe = dtype_bpe(dtype);

  // --- Chunk sizing (shared config, applied once) ---
  if (cfg->chunk_ratios) {
    size_t target =
      cfg->target_chunk_bytes ? cfg->target_chunk_bytes : (1 << 20);
    size_t budget = cfg->memory_budget;

    if (budget == 0) {
      size_t free_mem = 0, total_mem = 0;
      if (cuMemGetInfo(&free_mem, &total_mem) == CUDA_SUCCESS && free_mem > 0) {
        // Reserve half for each stream (80% total, split two ways)
        budget = (size_t)((double)free_mem * 0.4);
        print_report(
          "  auto-detect: %.2f GiB free GPU memory (2 streams, ~40%% each)",
          (double)free_mem / (1024.0 * 1024.0 * 1024.0));
      }
    }

    struct tile_stream_configuration fit_config = {
      .buffer_capacity_bytes = 128 << 20,
      .dtype = dtype,
      .rank = rank,
      .dimensions = dims,
      .codec = cfg->codec,
      .reduce_method = cfg->reduce_method,
      .append_reduce_method = cfg->append_reduce_method,
      .target_batch_chunks = 2048,
    };

    int fitted = 0;
    if (budget > 0) {
      fitted = tile_stream_gpu_advise_chunk_sizes(
                 &fit_config, target, cfg->chunk_ratios, budget) == 0;
      if (fitted) {
        uint64_t vol = 1;
        for (uint8_t d = 0; d < rank; ++d)
          vol *= dims[d].chunk_size;
        print_report("  auto-fit: %zu bytes/chunk", (size_t)(vol * bpe));
      } else {
        print_report("  auto-fit: WARNING — no chunk size fits in budget");
      }
    }
    if (!fitted)
      dims_budget_chunk_bytes(dims, rank, target, bpe, cfg->chunk_ratios);

    if (cfg->shard_counts)
      dims_set_shard_counts(dims, rank, cfg->shard_counts);
  }

  dims_print(dims, rank);

  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * bpe;

  // --- Sinks: zarr FS when -o given, discard otherwise ---
  struct discard_shard_sink dss[2];
  struct zarr_fs_sink* zsink[2] = { NULL, NULL };
  struct metering_sink meter[2] = { { 0 }, { 0 } };
  struct shard_sink* sink[2];

  if (output_path) {
    // Build per-stream paths: <output_path>/stream-0, <output_path>/stream-1
    char path0[1024], path1[1024];
    snprintf(path0, sizeof(path0), "%s/stream-0", output_path);
    snprintf(path1, sizeof(path1), "%s/stream-1", output_path);
    const char* paths[2] = { path0, path1 };

    for (int k = 0; k < 2; ++k) {
      struct zarr_config zcfg = {
        .store_path = paths[k],
        .array_name = array_name,
        .data_type = dtype,
        .fill_value = 0,
        .rank = rank,
        .dimensions = dims,
        .unbuffered = 1,
      };
      zsink[k] = zarr_fs_sink_create(&zcfg);
      CHECK(Fail, zsink[k]);
      metering_sink_init(&meter[k], zarr_fs_sink_as_shard_sink(zsink[k]));
      sink[k] = &meter[k].base;
    }
    print_report("  output-0: %s", path0);
    print_report("  output-1: %s", path1);
  } else {
    discard_shard_sink_init(&dss[0]);
    discard_shard_sink_init(&dss[1]);
    sink[0] = &dss[0].base;
    sink[1] = &dss[1].base;
  }

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .dtype = dtype,
    .rank = rank,
    .dimensions = dims,
    .codec = cfg->codec,
    .reduce_method = cfg->reduce_method,
    .append_reduce_method = cfg->append_reduce_method,
    .target_batch_chunks = 2048,
    .shard_alignment = output_path ? platform_page_size() : 0,
  };

  // Memory estimates
  {
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(&config, &mem) == 0) {
      print_report(
        "  GPU memory (per stream): %.2f GiB device, %.2f GiB pinned",
        (double)mem.device_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)mem.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
      print_report(
        "  GPU memory (total x2):   %.2f GiB device, %.2f GiB pinned",
        2.0 * (double)mem.device_bytes / (1024.0 * 1024.0 * 1024.0),
        2.0 * (double)mem.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
    }
  }

  // Create two GPU streams
  struct platform_clock init_clock = { 0 };
  platform_toc(&init_clock);

  struct tile_stream_gpu* s0 = NULL;
  struct tile_stream_gpu* s1 = NULL;
  s0 = tile_stream_gpu_create(&config, sink[0]);
  s1 = tile_stream_gpu_create(&config, sink[1]);
  CHECK(Fail, s0 && s1);
  float init_s = platform_toc(&init_clock);

  const struct tile_stream_layout* layout = tile_stream_gpu_layout(s0);
  log_bench_header(
    layout, dtype, cfg->codec, 0, 0, total_bytes, total_elements);

  struct writer* w0 = tile_stream_gpu_writer(s0);
  struct writer* w1 = tile_stream_gpu_writer(s1);

  // Interleaved pump
  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  CHECK(Fail, pump_data_interleaved(w0, w1, total_elements, fill, bpe) == 0);

  // Flush zarr sinks before measuring wall time
  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  for (int k = 0; k < 2; ++k) {
    if (zsink[k])
      zarr_fs_sink_flush(zsink[k]);
  }
  float flush_s = platform_toc(&flush_clock);
  float wall_s = platform_toc(&clock);

  // Verify cursors
  CHECK(Fail, tile_stream_gpu_cursor(s0) == total_elements);
  CHECK(Fail, tile_stream_gpu_cursor(s1) == total_elements);

  // Collect metrics
  struct stream_metrics m[2] = {
    tile_stream_gpu_get_metrics(s0),
    tile_stream_gpu_get_metrics(s1),
  };

  size_t sink_bytes[2] = {
    output_path ? meter[0].total_bytes : dss[0].total_bytes,
    output_path ? meter[1].total_bytes : dss[1].total_bytes,
  };

  const double GIB = 1024.0 * 1024.0 * 1024.0;
  double per_stream_gib = (double)total_bytes / GIB;
  double combined_gib = 2.0 * per_stream_gib;

  // --- Combined summary ---
  print_report("");
  print_report("  --- Combined ---");
  print_report("  Input:        %.2f GiB (%zu elements x 2 streams)",
               combined_gib,
               total_elements);
  print_report("  Compressed:   %.2f GiB",
               (double)(sink_bytes[0] + sink_bytes[1]) / GIB);
  print_report("  Init time:     %.3f s", (double)init_s);
  if (flush_s > 0)
    print_report("  Flush time:    %.3f s", (double)flush_s);
  print_report("  Wall time:     %.3f s", (double)wall_s);
  print_report("  Throughput:    %.2f GiB/s (combined)",
               wall_s > 0 ? combined_gib / wall_s : 0.0);

  // --- Per-stream reports ---
  for (int k = 0; k < 2; ++k) {
    print_report("");
    print_report("  --- stream-%d ---", k);
    print_report("  Throughput:    %.2f GiB/s",
                 wall_s > 0 ? per_stream_gib / wall_s : 0.0);
    print_report("  Compressed:    %.2f GiB", (double)sink_bytes[k] / GIB);
    print_report("");
    print_report("  %-12s %8s %8s %10s %10s",
                 "Stage",
                 "avg GB/s",
                 "best GB/s",
                 "avg ms",
                 "best ms");
    print_metric_row(&m[k].memcpy);
    print_metric_row(&m[k].h2d);
    print_metric_row(&m[k].scatter);
    print_metric_row(&m[k].lod_gather);
    print_metric_row(&m[k].lod_reduce);
    print_metric_row(&m[k].lod_append_fold);
    print_metric_row(&m[k].lod_morton_chunk);
    print_metric_row(&m[k].compress);
    print_metric_row(&m[k].aggregate);
    print_metric_row(&m[k].d2h);
    print_metric_row(&m[k].sink);
  }

  print_report("  PASS");
  tile_stream_gpu_destroy(s1);
  tile_stream_gpu_destroy(s0);
  zarr_fs_sink_destroy(zsink[0]);
  zarr_fs_sink_destroy(zsink[1]);
  return 0;

Fail:
  print_report("  FAIL");
  // Flush before destroying streams — pending write_direct jobs
  // reference pinned host buffers owned by the stream.
  for (int k = 0; k < 2; ++k) {
    if (zsink[k])
      zarr_fs_sink_flush(zsink[k]);
  }
  if (s1)
    tile_stream_gpu_destroy(s1);
  if (s0)
    tile_stream_gpu_destroy(s0);
  zarr_fs_sink_destroy(zsink[0]);
  zarr_fs_sink_destroy(zsink[1]);
  return 1;
}

int
bench_two_streams_main(int ac,
                       char* av[],
                       const char* label,
                       struct dimension* dims,
                       uint8_t rank,
                       const uint8_t* chunk_ratios,
                       size_t default_chunk_bytes,
                       const uint64_t* shard_counts)
{
  fill_fn fill = fill_xor;
  struct codec_config codec = { .id = CODEC_ZSTD };
  enum lod_reduce_method reduce = lod_reduce_mean;
  enum dtype dtype = dtype_u16;
  size_t target_chunk_bytes = 0;
  size_t memory_budget = 0;
  uint64_t frames = 0;
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
    } else if (strcmp(av[i], "--dtype") == 0 && i + 1 < ac) {
      if (!parse_dtype(av[++i], &dtype))
        return 1;
    } else if (strcmp(av[i], "--frames") == 0 && i + 1 < ac) {
      frames = (uint64_t)strtoull(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--chunk-bytes") == 0 && i + 1 < ac) {
      target_chunk_bytes = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "--memory-budget") == 0 && i + 1 < ac) {
      memory_budget = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "-o") == 0 && i + 1 < ac) {
      output_path = av[++i];
    } else {
      fprintf(stderr, "Unknown option: %s\n", av[i]);
      fprintf(stderr,
              "Usage: %s [--fill xor|zeros|rand] [--codec none|lz4|zstd] "
              "[--reduce mean|min|max|median|max_sup|min_sup] "
              "[--dtype u8|u16|...] [--frames N] "
              "[--chunk-bytes N] [--memory-budget N] [-o path]\n",
              av[0]);
      return 1;
    }
  }

  if (frames > 0)
    dims[0].size = frames;

  CUdevice dev;
  CUcontext ctx = 0;
  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  int need_xor = (fill == fill_xor);
  int need_rand = (fill == fill_rand);
  if (need_xor)
    xor_pattern_init(dims, rank, 16);
  if (need_rand)
    rand_pattern_init(dims, rank, 16);

  struct bench_config cfg = {
    .label = label,
    .dims = dims,
    .rank = rank,
    .fill = fill,
    .output_path = output_path,
    .array_name = label,
    .codec = codec,
    .reduce_method = reduce,
    .append_reduce_method =
      reduce == lod_reduce_median ? lod_reduce_max : reduce,
    .backend = BENCH_GPU,
    .dtype = dtype,
    .chunk_ratios = chunk_ratios,
    .target_chunk_bytes =
      target_chunk_bytes ? target_chunk_bytes : default_chunk_bytes,
    .memory_budget = memory_budget,
    .shard_counts = shard_counts,
  };
  int ecode = run_bench_two_streams(&cfg);

  if (need_xor)
    xor_pattern_free();
  if (need_rand)
    rand_pattern_free();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  cuCtxDestroy(ctx);
  return 1;
}
