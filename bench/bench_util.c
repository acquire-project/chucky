#include "bench_util.h"
#include "bench_parse.h"
#include "bench_report.h"
#include "bench_zarr.h"
#include "defs.limits.h"
#include "dimension.h"
#include "gpu/compress.h"
#include "gpu/lod.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "sink_discard.h"
#include "sink_metering.h"
#include "sink_throttled.h"
#include "stream.cpu.h"
#include "stream/layouts.h"
#include "util/format_bytes.h"
#include "util/metric.h"
#include "util/prelude.h"
#include "zarr/json_writer.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void
print_advise_failure(const struct advise_layout_diagnostic* diag,
                     size_t budget,
                     size_t min_shard_bytes);

// Resolve the chunk + shard geometry for cfg->dims using cfg->chunk_ratios.
// When cfg->memory_budget is 0, auto-detects from backend free memory using
// budget_fraction (e.g. 0.8 for single-stream, 0.4 per stream for the
// two-stream driver). auto_detect_suffix is appended to the auto-detect log
// message (e.g. "(restrict to <80%)" or "(2 streams, ~40% each)").
// On success returns 0 and writes the chosen epochs_per_batch (0 if no
// auto-fit ran) to *out_epb. On failure returns 1 (message already printed).
// No-op if cfg->chunk_ratios is NULL.
static int
resolve_chunk_sizing(const struct bench_config* cfg,
                     enum dtype dtype,
                     double budget_fraction,
                     const char* auto_detect_suffix,
                     uint32_t* out_epb)
{
  *out_epb = 0;
  if (!cfg->chunk_ratios)
    return 0;

  const size_t bytes_per_element = dtype_bpe(dtype);
  const size_t target =
    cfg->target_chunk_bytes ? cfg->target_chunk_bytes : (1 << 20);
  size_t budget = cfg->memory_budget;

  if (budget == 0) {
    char buf[32];
    if (cfg->backend == BENCH_GPU) {
      size_t free_mem = 0, total_mem = 0;
      if (cuMemGetInfo(&free_mem, &total_mem) == CUDA_SUCCESS && free_mem > 0) {
        budget = (size_t)((double)free_mem * budget_fraction);
        format_bytes(buf, sizeof(buf), free_mem);
        print_report(
          "  auto-detect: %s free GPU memory %s", buf, auto_detect_suffix);
      }
    } else {
      size_t avail = platform_available_memory();
      if (avail > 0) {
        budget = (size_t)((double)avail * budget_fraction);
        format_bytes(buf, sizeof(buf), avail);
        print_report(
          "  auto-detect: %s available RAM %s", buf, auto_detect_suffix);
      }
    }
  }

  struct dimension* dims = cfg->dims;
  const uint8_t rank = cfg->rank;

  if (budget > 0) {
    struct tile_stream_configuration fit_config = {
      .buffer_capacity_bytes = 128 << 20,
      .dtype = dtype,
      .rank = rank,
      .dimensions = dims,
      .codec = cfg->codec,
      .reduce_method = cfg->reduce_method,
      .append_reduce_method = cfg->append_reduce_method,
      .target_batch_bytes =
        cfg->target_batch_bytes ? cfg->target_batch_bytes : 512u << 20,
    };
    struct advise_layout_diagnostic diag = { 0 };
    int advise_ok;
    if (cfg->backend == BENCH_GPU) {
      advise_ok = tile_stream_gpu_advise_layout(&fit_config,
                                                target,
                                                cfg->min_chunk_bytes,
                                                cfg->chunk_ratios,
                                                budget,
                                                cfg->min_shard_bytes,
                                                cfg->target_concurrent_shards,
                                                cfg->min_append_shards,
                                                0,
                                                &diag);
    } else {
      advise_ok = tile_stream_cpu_advise_layout(&fit_config,
                                                target,
                                                cfg->min_chunk_bytes,
                                                cfg->chunk_ratios,
                                                budget,
                                                cfg->min_shard_bytes,
                                                cfg->target_concurrent_shards,
                                                cfg->min_append_shards,
                                                0,
                                                &diag);
    }
    if (advise_ok != 0) {
      print_advise_failure(&diag, budget, cfg->min_shard_bytes);
      return 1;
    }
    *out_epb = fit_config.epochs_per_batch;
    uint64_t vol = 1;
    for (uint8_t d = 0; d < rank; ++d)
      vol *= dims[d].chunk_size;
    print_report("  auto-fit: %zu bytes/chunk (batch=%u)",
                 (size_t)(vol * bytes_per_element),
                 (unsigned)*out_epb);
    return 0;
  }

  // No budget: apply chunk budget + shard geometry directly.
  if (dims_budget_chunk_bytes(
        dims, rank, target, bytes_per_element, cfg->chunk_ratios)) {
    print_report("  chunk budget: ERROR -- invalid input");
    return 1;
  }
  if (cfg->min_shard_bytes > 0 &&
      dims_set_shard_geometry(dims,
                              rank,
                              cfg->min_shard_bytes,
                              cfg->target_concurrent_shards,
                              cfg->min_append_shards,
                              bytes_per_element)) {
    print_report(
      "  shard geometry: ERROR -- min_shard_bytes is smaller than one chunk");
    return 1;
  }
  return 0;
}

// Emit a reason-specific explanation after advise_layout fails.
static void
print_advise_failure(const struct advise_layout_diagnostic* diag,
                     size_t budget,
                     size_t min_shard_bytes)
{
  char budget_buf[32], shard_buf[32], chunk_buf[32], dev_buf[32];
  format_bytes(budget_buf, sizeof(budget_buf), budget);
  format_bytes(shard_buf, sizeof(shard_buf), min_shard_bytes);
  format_bytes(chunk_buf, sizeof(chunk_buf), diag->chunk_bytes);
  format_bytes(dev_buf, sizeof(dev_buf), diag->device_bytes);

  switch (diag->reason) {
    case ADVISE_BUDGET_EXCEEDED:
      print_report(
        "  auto-fit: ERROR -- memory budget exceeded at floor chunk size");
      print_report("    needed %s at chunk=%s, K=%u; budget=%s",
                   dev_buf,
                   chunk_buf,
                   diag->epochs_per_batch,
                   budget_buf);
      print_report(
        "    fix: raise memory_budget, lower min_chunk_bytes, or simplify "
        "codec/LOD");
      break;
    case ADVISE_PARTS_LIMIT_EXCEEDED:
      print_report("  auto-fit: ERROR -- %llu chunks per shard exceeds backend "
                   "limit of %llu",
                   (unsigned long long)diag->chunks_per_shard_total,
                   (unsigned long long)diag->parts_limit);
      print_report("    at chunk=%s, min_shard_bytes=%s", chunk_buf, shard_buf);
      print_report(
        "    fix: lower min_shard_bytes, raise target_concurrent_shards, "
        "or lower min_chunk_bytes");
      break;
    case ADVISE_MIN_SHARD_TOO_SMALL:
      print_report(
        "  auto-fit: ERROR -- min_shard_bytes (%s) is smaller than one chunk "
        "(%s)",
        shard_buf,
        chunk_buf);
      print_report("    fix: raise min_shard_bytes or lower target chunk size");
      break;
    case ADVISE_INVALID_CONFIG:
      print_report(
        "  auto-fit: ERROR -- invalid configuration rejected by memory "
        "estimate or shard geometry");
      break;
    case ADVISE_CHUNK_BUDGET_INFEASIBLE:
      print_report("  auto-fit: ERROR -- chunk budget infeasible at chunk=%s",
                   chunk_buf);
      print_report("    fix: raise target_chunk_bytes, or reduce pinned dims");
      break;
    default:
      print_report("  auto-fit: ERROR -- unknown failure (reason=%d)",
                   (int)diag->reason);
      break;
  }
}

// Print aggregate shard geometry: uncompressed bytes per shard (append-outer
// close), total shards. Complements dims_print, which shows per-dim values.
static void
print_shard_summary(const struct dimension* dims,
                    uint8_t rank,
                    size_t bytes_per_element)
{
  uint64_t chunk_elements = 1;
  uint64_t cps_total = 1;
  uint64_t total_shards = 1;
  for (uint8_t d = 0; d < rank; ++d) {
    chunk_elements *= dims[d].chunk_size;
    uint64_t tc = ceildiv(dims[d].size, dims[d].chunk_size);
    uint64_t cps = dims[d].chunks_per_shard ? dims[d].chunks_per_shard : tc;
    cps_total *= cps;
    total_shards *= tc ? ceildiv(tc, cps) : 1;
  }
  const uint64_t chunk_bytes = chunk_elements * bytes_per_element;
  const uint64_t shard_bytes = chunk_bytes * cps_total;
  char buf[32];
  format_bytes(buf, sizeof(buf), shard_bytes);
  print_report("  shard:       %s uncompressed (%llu chunks), %llu total",
               buf,
               (unsigned long long)cps_total,
               (unsigned long long)total_shards);
}

// --- Fill-pattern init (deferred until after chunk-fit succeeds) ---
//
// Pattern buffers can be several GiB for large arrays; initializing them in
// the CLI driver would pay that cost even for runs that fail at auto-fit.

static void
init_fill_pattern(fill_fn fill, const struct dimension* dims, uint8_t rank)
{
  if (fill == fill_xor)
    xor_pattern_init(dims, rank, 16);
  else if (fill == fill_rand)
    rand_pattern_init(dims, rank, 16);
}

static void
free_fill_pattern(fill_fn fill)
{
  if (fill == fill_xor)
    xor_pattern_free();
  else if (fill == fill_rand)
    rand_pattern_free();
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
  uint32_t chosen_epochs_per_batch = 0; // 0 = auto; set by advise_layout on fit

  if (resolve_chunk_sizing(
        cfg, dtype, 0.8, "(restrict to <80%)", &chosen_epochs_per_batch))
    return 1;

  dims_print(dims, rank);
  print_shard_summary(dims, rank, dtype_bpe(dtype));

  init_fill_pattern(fill, dims, rank);

  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * dtype_bpe(dtype);

  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  struct bench_zarr_handle zarr = { 0 };
  struct metering_sink meter = { 0 };
  struct throttled_shard_sink tss = { 0 };
  int use_throttled = 0;
  struct shard_sink* sink = &dss.base;
  struct bench_handle h = { .backend = cfg->backend };

  if (cfg->s3_bucket) {
    CHECK(Fail,
          bench_zarr_open_s3(&zarr,
                             cfg->s3_bucket,
                             cfg->s3_prefix ? cfg->s3_prefix : label,
                             array_name,
                             cfg->s3_region,
                             cfg->s3_endpoint,
                             cfg->s3_throughput_gbps,
                             dims,
                             rank,
                             dtype,
                             0,
                             cfg->codec,
                             is_multiscale) == 0);
    metering_sink_init(&meter, bench_zarr_as_shard_sink(&zarr));
    sink = &meter.base;
  } else if (output_path) {
    CHECK(Fail,
          bench_zarr_open_fs(&zarr,
                             output_path,
                             array_name,
                             dims,
                             rank,
                             dtype,
                             0,
                             cfg->codec,
                             is_multiscale) == 0);
    metering_sink_init(&meter, bench_zarr_as_shard_sink(&zarr));
    sink = &meter.base;
  } else if (cfg->io_bw_mbps > 0 || cfg->io_latency_us > 0) {
    CHECK(Fail,
          throttled_shard_sink_init(
            &tss, cfg->io_bw_mbps, cfg->io_latency_us) == 0);
    sink = &tss.base;
    use_throttled = 1;
  }

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .dtype = dtype,
    .rank = rank,
    .dimensions = dims,
    .codec = cfg->codec,
    .reduce_method = cfg->reduce_method,
    .append_reduce_method = cfg->append_reduce_method,
    .epochs_per_batch = chosen_epochs_per_batch,
    .target_batch_bytes = 512u << 20,
    .backpressure_bytes = cfg->backpressure_bytes,
    .max_threads = cfg->max_threads,
  };

  uint64_t est_total_chunks = 0;
  size_t est_total_bytes = 0;
  size_t est_pinned_bytes = 0;

  if (cfg->backend == BENCH_GPU) {
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(&config, 0, &mem) == 0) {
      est_total_chunks = mem.total_chunks;
      est_total_bytes = mem.device_bytes;
      est_pinned_bytes = mem.host_pinned_bytes;
      char a[32], b[32];
      format_bytes(a, sizeof(a), mem.device_bytes);
      format_bytes(b, sizeof(b), mem.host_pinned_bytes);
      print_report("  GPU memory:  %s device, %s pinned", a, b);
      format_bytes(a, sizeof(a), mem.staging_bytes);
      format_bytes(b, sizeof(b), mem.chunk_pool_bytes);
      print_report("    staging:   %s   chunk_pool: %s", a, b);
      format_bytes(a, sizeof(a), mem.compressed_pool_bytes);
      format_bytes(b, sizeof(b), mem.aggregate_bytes);
      print_report("    comp_pool: %s   aggregate: %s", a, b);
      format_bytes(a, sizeof(a), mem.lod_bytes);
      format_bytes(b, sizeof(b), mem.codec_bytes);
      print_report("    lod:       %s   codec:     %s", a, b);
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
    if (tile_stream_cpu_memory_estimate(&config, 0, &mem) == 0) {
      est_total_chunks = mem.total_chunks;
      est_total_bytes = mem.heap_bytes;
      char a[32], b[32];
      format_bytes(a, sizeof(a), mem.heap_bytes);
      print_report("  CPU memory:  %s heap", a);
      format_bytes(a, sizeof(a), mem.chunk_pool_bytes);
      format_bytes(b, sizeof(b), mem.compressed_pool_bytes);
      print_report("    chunk_pool: %s   comp_pool: %s", a, b);
      format_bytes(a, sizeof(a), mem.comp_sizes_bytes);
      format_bytes(b, sizeof(b), mem.aggregate_bytes);
      print_report("    comp_sizes: %s   aggregate: %s", a, b);
      format_bytes(a, sizeof(a), mem.lod_bytes);
      format_bytes(b, sizeof(b), mem.shard_bytes);
      print_report("    lod:       %s   shards:    %s", a, b);
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
  double fill_s = 0.0;
  CHECK(Fail,
        pump_data_modal(bench_writer(&h),
                        total_elements,
                        fill,
                        dtype_bpe(dtype),
                        cfg->pump_mode,
                        &fill_s) == 0);

  size_t pending_bytes = bench_zarr_pending_bytes(&zarr);

  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  if (bench_zarr_flush(&zarr)) {
    log_error("  I/O error detected during flush");
    goto Fail;
  }
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
    size_t sink_total_bytes;
    if (has_output)
      sink_total_bytes = meter.total_bytes;
    else if (use_throttled)
      sink_total_bytes = (size_t)atomic_load(&tss.total_bytes);
    else
      sink_total_bytes = dss.total_bytes;
    struct sink_stats ss = { .total_bytes = sink_total_bytes,
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

    if (cfg->pump_mode != PUMP_CYCLE_BLOCKS || fill_s > 0)
      print_report("  Data gen time: %.3f s (%.1f%% of wall)",
                   fill_s,
                   wall_s > 0 ? 100.0 * fill_s / wall_s : 0.0);

    if (cfg->json_output) {
      const struct stream_metric* sink_metric =
        meter.metric.count > 0 ? &meter.metric : NULL;
      print_bench_json_pass(&m,
                            sink_metric,
                            layout,
                            config.dtype,
                            &ss,
                            total_bytes,
                            total_elements,
                            wall_s,
                            init_s,
                            flush_s,
                            (float)fill_s,
                            est_total_bytes,
                            est_pinned_bytes);
    }
  }

  print_report("  PASS");
  int rc = 0;
  goto Cleanup;

Fail:
  if (cfg->json_output)
    print_bench_json_error();
  print_report("  FAIL");
  rc = 1;

Cleanup:
  // Flush before destroying the stream — pending write_direct jobs
  // reference pinned host buffers owned by the stream.
  bench_zarr_flush(&zarr);
  bench_destroy(&h);
  bench_zarr_close(&zarr);
  if (use_throttled)
    throttled_shard_sink_teardown(&tss);
  free_fill_pattern(fill);
  return rc;
}

// --- CLI driver ---

struct bench_cli_args
{
  fill_fn fill;
  struct codec_config codec;
  enum lod_reduce_method reduce;
  enum bench_backend backend;
  enum dtype dtype;
  size_t target_chunk_bytes;
  size_t target_batch_bytes;
  size_t memory_budget;
  uint64_t frames;
  int json_output;
  const char* output_path;
  const char* s3_bucket;
  const char* s3_prefix;
  const char* s3_region;
  const char* s3_endpoint;
  double s3_throughput_gbps;
  uint64_t io_bw_mbps;
  uint64_t io_latency_us;
  size_t backpressure_bytes;
  int max_threads;
  enum pump_mode pump_mode;
};

// Parse the shared bench CLI flags into out. Unknown options print a usage
// string and return 1. Flags accepted:
//   --fill --codec --reduce --backend --dtype --frames --json --chunk-bytes
//   --memory-budget -o --s3-bucket --s3-prefix --s3-region --s3-endpoint
//   --s3-throughput-gbps --io-bw-mbps --io-latency-us --backpressure
//   --max-threads.
// Drivers that don't honor a given flag (e.g. two-streams ignores --backend)
// just don't read the corresponding field afterward.
static int
parse_bench_cli_args(int ac, char* av[], struct bench_cli_args* out)
{
  out->fill = fill_xor;
  out->codec = (struct codec_config){ .id = CODEC_ZSTD };
  out->reduce = lod_reduce_mean;
  out->backend = BENCH_GPU;
  out->dtype = dtype_u16;
  out->target_chunk_bytes = 0;
  out->target_batch_bytes = 0;
  out->memory_budget = 0;
  out->frames = 0;
  out->json_output = 0;
  out->output_path = NULL;
  out->s3_bucket = NULL;
  out->s3_prefix = NULL;
  out->s3_region = NULL;
  out->s3_endpoint = NULL;
  out->s3_throughput_gbps = 0;
  out->io_bw_mbps = 0;
  out->io_latency_us = 0;
  out->backpressure_bytes = 0;
  out->max_threads = 0;
  out->pump_mode = PUMP_CYCLE_BLOCKS;

  const char* pump_flag = NULL; // last pump-mode flag seen (conflict warning)
  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--fill") == 0 && i + 1 < ac) {
      out->fill = parse_fill(av[++i]);
      if (!out->fill)
        return 1;
    } else if (strcmp(av[i], "--codec") == 0 && i + 1 < ac) {
      if (!parse_codec(av[++i], &out->codec))
        return 1;
    } else if (strcmp(av[i], "--reduce") == 0 && i + 1 < ac) {
      if (!parse_reduce(av[++i], &out->reduce))
        return 1;
    } else if (strcmp(av[i], "--backend") == 0 && i + 1 < ac) {
      if (!parse_backend(av[++i], &out->backend))
        return 1;
    } else if (strcmp(av[i], "--dtype") == 0 && i + 1 < ac) {
      if (!parse_dtype(av[++i], &out->dtype))
        return 1;
    } else if (strcmp(av[i], "--frames") == 0 && i + 1 < ac) {
      out->frames = (uint64_t)strtoull(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--json") == 0) {
      out->json_output = 1;
    } else if (strcmp(av[i], "--chunk-bytes") == 0 && i + 1 < ac) {
      out->target_chunk_bytes = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "--batch-bytes") == 0 && i + 1 < ac) {
      out->target_batch_bytes = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "--memory-budget") == 0 && i + 1 < ac) {
      out->memory_budget = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "-o") == 0 && i + 1 < ac) {
      out->output_path = av[++i];
    } else if (strcmp(av[i], "--s3-bucket") == 0 && i + 1 < ac) {
      out->s3_bucket = av[++i];
    } else if (strcmp(av[i], "--s3-prefix") == 0 && i + 1 < ac) {
      out->s3_prefix = av[++i];
    } else if (strcmp(av[i], "--s3-region") == 0 && i + 1 < ac) {
      out->s3_region = av[++i];
    } else if (strcmp(av[i], "--s3-endpoint") == 0 && i + 1 < ac) {
      out->s3_endpoint = av[++i];
    } else if (strcmp(av[i], "--s3-throughput-gbps") == 0 && i + 1 < ac) {
      out->s3_throughput_gbps = strtod(av[++i], NULL);
    } else if (strcmp(av[i], "--io-bw-mbps") == 0 && i + 1 < ac) {
      out->io_bw_mbps = strtoull(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--io-latency-us") == 0 && i + 1 < ac) {
      out->io_latency_us = strtoull(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--backpressure") == 0 && i + 1 < ac) {
      out->backpressure_bytes = parse_bytes(av[++i]);
    } else if (strcmp(av[i], "--max-threads") == 0 && i + 1 < ac) {
      out->max_threads = (int)strtol(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--prefill") == 0) {
      if (pump_flag)
        fprintf(stderr, "Warning: %s overrides %s\n", av[i], pump_flag);
      out->pump_mode = PUMP_SINGLE_BLOCK;
      pump_flag = "--prefill";
    } else if (strcmp(av[i], "--busy-producer") == 0) {
      if (pump_flag)
        fprintf(stderr, "Warning: %s overrides %s\n", av[i], pump_flag);
      out->pump_mode = PUMP_BUSY_PRODUCER;
      pump_flag = "--busy-producer";
    } else {
      fprintf(stderr, "Unknown option: %s\n", av[i]);
      fprintf(stderr,
              "Usage: %s [--fill xor|zeros|rand] [--codec none|lz4|zstd] "
              "[--reduce mean|min|max|median|max_sup|min_sup] "
              "[--backend gpu|cpu] [--dtype u8|u16|...] [--frames N] "
              "[--json] [--chunk-bytes N] [--batch-bytes N] "
              "[--memory-budget N] [-o path] "
              "[--s3-bucket B --s3-region R --s3-endpoint E [--s3-prefix P] "
              "[--s3-throughput-gbps N]] "
              "[--io-bw-mbps N (MiB/s)] [--io-latency-us N] "
              "[--backpressure N (bytes, e.g. 256M)] "
              "[--max-threads N (0 = OpenMP default)] "
              "[--busy-producer (regenerate input every append)] "
              "[--prefill (one identical block, within-mode A/B only)]\n",
              av[0]);
      return 1;
    }
  }
  return 0;
}

int
bench_stream_main(int ac, char* av[], struct bench_spec spec)
{
  struct bench_cli_args a;
  if (parse_bench_cli_args(ac, av, &a))
    return 1;

  struct dimension* dims = spec.dims;
  if (a.frames > 0)
    dims[0].size = a.frames;

  int ecode = 0;
  CUcontext ctx = 0;

  if (a.backend == BENCH_GPU) {
    CUdevice dev;
    CU(Fail, cuInit(0));
    CU(Fail, cuDeviceGet(&dev, 0));
    CU(Fail, cu_ctx_create(&ctx, 0, dev));
  }

  struct bench_config cfg = {
    .label = spec.label,
    .dims = dims,
    .rank = spec.rank,
    .fill = a.fill,
    .output_path = a.output_path,
    .array_name = spec.label,
    .s3_bucket = a.s3_bucket,
    .s3_prefix = a.s3_prefix,
    .s3_region = a.s3_region,
    .s3_endpoint = a.s3_endpoint,
    .s3_throughput_gbps = a.s3_throughput_gbps,
    .codec = a.codec,
    .reduce_method = a.reduce,
    .append_reduce_method =
      a.reduce == lod_reduce_median ? lod_reduce_max : a.reduce,
    .backend = a.backend,
    .dtype = a.dtype,
    .chunk_ratios = spec.chunk_ratios,
    .target_chunk_bytes =
      a.target_chunk_bytes ? a.target_chunk_bytes : spec.target_chunk_bytes,
    .min_chunk_bytes = spec.min_chunk_bytes,
    .target_batch_bytes = a.target_batch_bytes,
    .memory_budget = a.memory_budget,
    .min_shard_bytes = spec.min_shard_bytes,
    .target_concurrent_shards = spec.target_concurrent_shards,
    .min_append_shards = spec.min_append_shards,
    .json_output = a.json_output,
    .io_bw_mbps = a.io_bw_mbps,
    .io_latency_us = a.io_latency_us,
    .backpressure_bytes = a.backpressure_bytes,
    .max_threads = a.max_threads,
    .pump_mode = a.pump_mode,
  };
  ecode = run_bench(&cfg);

  if (a.backend == BENCH_GPU)
    cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  if (a.backend == BENCH_GPU)
    cuCtxDestroy(ctx);
  return 1;
}

// ---------------------------------------------------------------------------
// Two-stream benchmark: two GPU pipelines, interleaved append on one thread
// ---------------------------------------------------------------------------

// Interleaved pump: alternates writer_append between two writers in lockstep.
// Each iteration pushes one chunk to each writer before advancing. The pump
// mode (see test_data.h) decides how the chunk contents are produced; the
// default cycles a handful of pre-generated distinct blocks with no in-loop
// fill. Fill seconds are written to *out_fill_s when non-NULL.
static int
pump_data_interleaved(struct writer* w0,
                      struct writer* w1,
                      size_t total_elements,
                      fill_fn fill,
                      size_t bpe,
                      enum pump_mode mode,
                      double* out_fill_s)
{
  const size_t nelements = PUMP_BLOCK_ELEMENTS;

  if (out_fill_s)
    *out_fill_s = 0.0;

  struct pump_blocks blocks;
  if (pump_blocks_alloc(
        &blocks, total_elements, fill, bpe, mode, out_fill_s != NULL))
    return 1;
  const size_t nblocks = blocks.count;

  struct platform_clock fill_clock = { 0 };
  double fill_s = blocks.fill_s;

  int err = 0;
  size_t off0 = 0, off1 = 0;
  size_t b0 = 0, b1 = 0;

  while (off0 < total_elements || off1 < total_elements) {
    // --- stream 0 ---
    if (off0 < total_elements) {
      size_t n = nelements;
      if (off0 + n > total_elements)
        n = total_elements - off0;
      uint16_t* data = blocks.block[b0];
      if (mode == PUMP_BUSY_PRODUCER) {
        if (out_fill_s)
          platform_toc(&fill_clock);
        fill(data, n, off0, total_elements);
        if (out_fill_s)
          fill_s += (double)platform_toc(&fill_clock);
      }
      b0 = (b0 + 1 == nblocks) ? 0 : b0 + 1;
      struct slice input = { .beg = data, .end = (char*)data + n * bpe };
      struct writer_result r = writer_append_wait(w0, input);
      if (r.error) {
        log_error("  stream-0 append failed at offset %zu", off0);
        err = 1;
        break;
      }
      off0 += n;
    }

    // --- stream 1 ---
    if (off1 < total_elements) {
      size_t n = nelements;
      if (off1 + n > total_elements)
        n = total_elements - off1;
      uint16_t* data = blocks.block[b1];
      if (mode == PUMP_BUSY_PRODUCER) {
        if (out_fill_s)
          platform_toc(&fill_clock);
        fill(data, n, off1, total_elements);
        if (out_fill_s)
          fill_s += (double)platform_toc(&fill_clock);
      }
      b1 = (b1 + 1 == nblocks) ? 0 : b1 + 1;
      struct slice input = { .beg = data, .end = (char*)data + n * bpe };
      struct writer_result r = writer_append_wait(w1, input);
      if (r.error) {
        log_error("  stream-1 append failed at offset %zu", off1);
        err = 1;
        break;
      }
      off1 += n;
    }
  }

  struct writer_result r0 = writer_flush(w0);
  struct writer_result r1 = writer_flush(w1);
  if (!err)
    err = r0.error || r1.error;
  pump_blocks_free(&blocks);
  if (out_fill_s)
    *out_fill_s = fill_s;
  return err;
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
  uint32_t chosen_epochs_per_batch = 0; // 0 = auto; set by advise_layout on fit

  if (resolve_chunk_sizing(
        cfg, dtype, 0.4, "(2 streams, ~40% each)", &chosen_epochs_per_batch))
    return 1;

  dims_print(dims, rank);
  print_shard_summary(dims, rank, dtype_bpe(dtype));

  init_fill_pattern(fill, dims, rank);

  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * bpe;

  // --- Sinks: zarr FS when -o given, throttled when flags set, discard else
  // ---
  struct discard_shard_sink dss[2];
  struct bench_zarr_handle zarr[2] = { { 0 }, { 0 } };
  struct metering_sink meter[2] = { 0 };
  struct throttled_shard_sink tss[2] = { 0 };
  int use_throttled = 0;
  struct shard_sink* sink[2];

  if (output_path) {
    // Build per-stream paths: <output_path>/stream-0, <output_path>/stream-1
    char path0[1024], path1[1024];
    snprintf(path0, sizeof(path0), "%s/stream-0", output_path);
    snprintf(path1, sizeof(path1), "%s/stream-1", output_path);
    const char* paths[2] = { path0, path1 };

    for (int k = 0; k < 2; ++k) {
      CHECK(Fail,
            bench_zarr_open_fs(&zarr[k],
                               paths[k],
                               array_name,
                               dims,
                               rank,
                               dtype,
                               0,
                               cfg->codec,
                               0 /* single array */) == 0);
      metering_sink_init(&meter[k], bench_zarr_as_shard_sink(&zarr[k]));
      sink[k] = &meter[k].base;
    }
    print_report("  output-0: %s", path0);
    print_report("  output-1: %s", path1);
  } else if (cfg->io_bw_mbps > 0 || cfg->io_latency_us > 0) {
    for (int k = 0; k < 2; ++k) {
      CHECK(Fail,
            throttled_shard_sink_init(
              &tss[k], cfg->io_bw_mbps, cfg->io_latency_us) == 0);
      sink[k] = &tss[k].base;
    }
    use_throttled = 1;
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
    .epochs_per_batch = chosen_epochs_per_batch,
    .target_batch_bytes = 512u << 20,
    .backpressure_bytes = cfg->backpressure_bytes,
    .max_threads = cfg->max_threads,
  };

  // Memory estimates
  {
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(&config, 0, &mem) == 0) {
      char a[32], b[32];
      format_bytes(a, sizeof(a), mem.device_bytes);
      format_bytes(b, sizeof(b), mem.host_pinned_bytes);
      print_report("  GPU memory (per stream): %s device, %s pinned", a, b);
      format_bytes(a, sizeof(a), 2 * mem.device_bytes);
      format_bytes(b, sizeof(b), 2 * mem.host_pinned_bytes);
      print_report("  GPU memory (total x2):   %s device, %s pinned", a, b);
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
  double fill_s = 0.0;
  CHECK(Fail,
        pump_data_interleaved(
          w0, w1, total_elements, fill, bpe, cfg->pump_mode, &fill_s) == 0);

  // Flush zarr sinks before measuring wall time
  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  for (int k = 0; k < 2; ++k)
    CHECK(Fail, bench_zarr_flush(&zarr[k]) == 0);
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

  size_t sink_bytes[2];
  for (int k = 0; k < 2; ++k) {
    if (output_path)
      sink_bytes[k] = meter[k].total_bytes;
    else if (use_throttled)
      sink_bytes[k] = (size_t)atomic_load(&tss[k].total_bytes);
    else
      sink_bytes[k] = dss[k].total_bytes;
  }

  const double GIB = 1024.0 * 1024.0 * 1024.0;
  double per_stream_gib = (double)total_bytes / GIB;
  double combined_gib = 2.0 * per_stream_gib;

  // --- Combined summary ---
  fputc('\n', stderr);
  print_report("  --- Combined ---");
  {
    char buf[32];
    format_bytes(buf, sizeof(buf), 2 * (uint64_t)total_bytes);
    print_report(
      "  Input:        %s (%zu elements x 2 streams)", buf, total_elements);
    format_bytes(buf, sizeof(buf), (uint64_t)(sink_bytes[0] + sink_bytes[1]));
    print_report("  Compressed:   %s", buf);
  }
  print_report("  Init time:     %.3f s", (double)init_s);
  if (flush_s > 0)
    print_report("  Flush time:    %.3f s", (double)flush_s);
  print_report("  Wall time:     %.3f s", (double)wall_s);
  if (cfg->pump_mode != PUMP_CYCLE_BLOCKS || fill_s > 0)
    print_report("  Data gen time: %.3f s (%.1f%% of wall)",
                 fill_s,
                 wall_s > 0 ? 100.0 * fill_s / wall_s : 0.0);
  print_report("  Throughput:    %.2f GiB/s (combined)",
               wall_s > 0 ? combined_gib / wall_s : 0.0);

  // --- Per-stream reports ---
  for (int k = 0; k < 2; ++k) {
    fputc('\n', stderr);
    print_report("  --- stream-%d ---", k);
    print_report("  Throughput:    %.2f GiB/s",
                 wall_s > 0 ? per_stream_gib / wall_s : 0.0);
    char cbuf[32];
    format_bytes(cbuf, sizeof(cbuf), (uint64_t)sink_bytes[k]);
    print_report("  Compressed:    %s", cbuf);
    fputc('\n', stderr);
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

    int have_stalls =
      m[k].flush_stall.count > 0 || m[k].kick_sync_stall.count > 0 ||
      m[k].io_fence_stall.count > 0 || m[k].backpressure.count > 0 ||
      m[k].max_append_ms > 0 || m[k].peak_pending_bytes > 0;
    if (have_stalls) {
      fputc('\n', stderr);
      print_report("  --- Stall stats (stream-%d) ---", k);
      print_metric_row(&m[k].flush_stall);
      print_metric_row(&m[k].kick_sync_stall);
      print_metric_row(&m[k].io_fence_stall);
      print_metric_row(&m[k].backpressure);
      print_report("  max append ms:   %.2f", (double)m[k].max_append_ms);
      char pbuf[32];
      format_bytes(pbuf, sizeof(pbuf), (uint64_t)m[k].peak_pending_bytes);
      print_report("  peak pending:    %s", pbuf);
    }
  }

  print_report("  PASS");
  tile_stream_gpu_destroy(s1);
  tile_stream_gpu_destroy(s0);
  bench_zarr_close(&zarr[0]);
  bench_zarr_close(&zarr[1]);
  if (use_throttled) {
    throttled_shard_sink_teardown(&tss[0]);
    throttled_shard_sink_teardown(&tss[1]);
  }
  free_fill_pattern(fill);
  return 0;

Fail:
  print_report("  FAIL");
  // Flush before destroying streams — pending write_direct jobs
  // reference pinned host buffers owned by the stream.
  for (int k = 0; k < 2; ++k)
    bench_zarr_flush(&zarr[k]);
  if (s1)
    tile_stream_gpu_destroy(s1);
  if (s0)
    tile_stream_gpu_destroy(s0);
  bench_zarr_close(&zarr[0]);
  bench_zarr_close(&zarr[1]);
  if (use_throttled) {
    throttled_shard_sink_teardown(&tss[0]);
    throttled_shard_sink_teardown(&tss[1]);
  }
  free_fill_pattern(fill);
  return 1;
}

int
bench_two_streams_main(int ac, char* av[], struct bench_spec spec)
{
  struct bench_cli_args a;
  if (parse_bench_cli_args(ac, av, &a))
    return 1;

  struct dimension* dims = spec.dims;
  if (a.frames > 0)
    dims[0].size = a.frames;

  CUdevice dev;
  CUcontext ctx = 0;
  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cu_ctx_create(&ctx, 0, dev));

  struct bench_config cfg = {
    .label = spec.label,
    .dims = dims,
    .rank = spec.rank,
    .fill = a.fill,
    .output_path = a.output_path,
    .array_name = spec.label,
    .codec = a.codec,
    .reduce_method = a.reduce,
    .append_reduce_method =
      a.reduce == lod_reduce_median ? lod_reduce_max : a.reduce,
    .backend = BENCH_GPU, // two-streams is GPU-only
    .dtype = a.dtype,
    .chunk_ratios = spec.chunk_ratios,
    .target_chunk_bytes =
      a.target_chunk_bytes ? a.target_chunk_bytes : spec.target_chunk_bytes,
    .min_chunk_bytes = spec.min_chunk_bytes,
    .target_batch_bytes = a.target_batch_bytes,
    .memory_budget = a.memory_budget,
    .min_shard_bytes = spec.min_shard_bytes,
    .target_concurrent_shards = spec.target_concurrent_shards,
    .min_append_shards = spec.min_append_shards,
    .io_bw_mbps = a.io_bw_mbps,
    .io_latency_us = a.io_latency_us,
    .backpressure_bytes = a.backpressure_bytes,
    .max_threads = a.max_threads,
    .pump_mode = a.pump_mode,
  };
  int ecode = run_bench_two_streams(&cfg);

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  cuCtxDestroy(ctx);
  return 1;
}
