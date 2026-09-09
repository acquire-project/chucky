#include "bench_util.h"
#include "bench_gpu.h"
#include "bench_parse.h"
#include "bench_report.h"
#include "bench_zarr.h"
#include "defs.limits.h"
#include "dimension.h"
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
#include "zarr/shard_pool_fs.h"

#include <errno.h>
#include <math.h>
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
                                 : bench_gpu_layout(h->gpu);
}

static struct stream_metrics
bench_get_metrics(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_get_metrics(h->cpu)
                                 : bench_gpu_get_metrics(h->gpu);
}

static struct writer*
bench_writer(struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_writer(h->cpu)
                                 : bench_gpu_writer(h->gpu);
}

static uint64_t
bench_cursor(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_cursor(h->cpu)
                                 : bench_gpu_cursor(h->gpu);
}

static int
bench_worker_threads(const struct bench_handle* h)
{
  return h->backend == BENCH_CPU ? tile_stream_cpu_worker_threads(h->cpu)
                                 : bench_gpu_worker_threads(h->gpu);
}

static void
bench_destroy(struct bench_handle* h)
{
  if (h->backend == BENCH_CPU)
    tile_stream_cpu_destroy(h->cpu);
  else
    bench_gpu_destroy(h->gpu);
}

static void
print_advise_failure(const struct advise_layout_diagnostic* diag,
                     size_t budget,
                     size_t min_shard_bytes);

static uint64_t
resolved_batch_bytes(const struct bench_config* cfg)
{
  return cfg->target_batch_bytes ? cfg->target_batch_bytes
                                 : (uint64_t)512 << 20;
}

static int
bench_failed(int json_output)
{
  if (json_output)
    print_bench_json_error();
  print_report("  FAIL");
  return 1;
}

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
      size_t free_mem = bench_gpu_free_memory();
      if (free_mem > 0) {
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
      .target_batch_bytes = resolved_batch_bytes(cfg),
    };
    struct advise_layout_diagnostic diag = { 0 };
    int advise_ok;
    if (cfg->backend == BENCH_GPU) {
      advise_ok = bench_gpu_advise_layout(&fit_config,
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
  print_report("  %-17s %s uncompressed (%llu chunks), %llu total",
               "Shard:",
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

static void
append_sample_add(struct bench_append_sample* sample, double ms)
{
  sample->calls++;
  sample->over_100ms += ms >= 100;
  sample->total_ms += ms;
  if (ms > sample->max_ms)
    sample->max_ms = ms;
}

// Called only after a reset/drain, when no delivery worker can update the sink.
static uint64_t
sink_bytes(const struct metering_sink* meter,
           const struct throttled_shard_sink* throttled,
           const struct discard_shard_sink* discard)
{
  if (meter->inner)
    return meter->total_bytes;
  if (throttled->scheduler)
    return atomic_load(&throttled->total_bytes);
  return atomic_load(&discard->total_bytes);
}

static int
pump_measurement(struct bench_handle* h,
                 const unsigned char* source,
                 const struct bench_config* cfg,
                 struct bench_measurement* run,
                 const struct metering_sink* meter,
                 const struct throttled_shard_sink* throttled,
                 const struct discard_shard_sink* discard,
                 size_t* total_bytes)
{
  uint64_t accepted = 0, checked = 0;
  uint64_t frame_bytes = dtype_bpe(cfg->dtype ? cfg->dtype : dtype_u16);
  for (uint8_t d = 1; d < cfg->rank; ++d)
    frame_bytes *= cfg->dims[d].size;
  if (cfg->frames > SIZE_MAX / frame_bytes)
    return 1;
  const uint64_t limit = cfg->frames * frame_bytes;
  uint64_t next[3];
  int following[3] = { 0 };
  memcpy(next, run->boundary_bytes, sizeof(next));
  const int64_t start = platform_monotonic_ns();
  int64_t measure_start = start;
  int measuring = cfg->warmup_s == 0;
  while (1) {
    const size_t offset = accepted % run->source_bytes;
    size_t offer = run->append_bytes;
    if (offer > run->source_bytes - offset)
      offer = run->source_bytes - offset;
    // A warmup checkpoint never finalizes a partial batch. Clip just this
    // offer to a batch boundary; append-downsample folds may need more batches.
    if (!measuring) {
      const uint64_t remaining =
        run->boundary_bytes[0] - accepted % run->boundary_bytes[0];
      if (offer > remaining)
        offer = remaining;
    } else if (limit && offer > limit - (accepted - run->warmup_bytes)) {
      offer = limit - (accepted - run->warmup_bytes);
    }
    if (accepted > SIZE_MAX - offer)
      return 1;
    const uint64_t end = accepted + offer;
    int crossing[3], active = 0;
    if (run->boundary_timing) {
      for (int i = 0; i < 3; ++i) {
        crossing[i] = end >= next[i];
        active |= crossing[i] || following[i];
      }
    }
    const int timed = active && measuring;
    const int64_t before = timed ? platform_monotonic_ns() : 0;
    const struct slice input = { source + offset, source + offset + offer };
    struct writer_result result = writer_append_wait(bench_writer(h), input);
    const double ms = timed ? (platform_monotonic_ns() - before) * 1e-6 : 0;
    if (result.error || result.rest.beg != result.rest.end)
      return 1;
    if (active) {
      for (int i = 0; i < 3; ++i) {
        if (timed && crossing[i])
          append_sample_add(&run->boundary[i], ms);
        if (timed && following[i])
          append_sample_add(&run->following[i], ms);
        if (following[i])
          --following[i];
        if (crossing[i]) {
          next[i] = (end / run->boundary_bytes[i] + 1) * run->boundary_bytes[i];
          following[i] = 16;
        }
      }
    }
    accepted = end;
    const int at_limit =
      measuring && limit && accepted - run->warmup_bytes == limit;
    // Avoid clock reads per tiny append, but always check a frame limit and
    // warmup batch boundary. Long blocking appends can overshoot a duration.
    if (!at_limit && accepted - checked < (4u << 20) &&
        (measuring || accepted % run->boundary_bytes[0]))
      continue;
    checked = accepted;
    int64_t now = platform_monotonic_ns();
    if (!measuring && (now - start) * 1e-9 >= cfg->warmup_s &&
        accepted % run->boundary_bytes[0] == 0) {
      const int rc = h->backend == BENCH_CPU
                       ? tile_stream_cpu_reset_metrics(h->cpu)
                       : bench_gpu_reset_metrics(h->gpu);
      if (rc < 0)
        return 1;
      if (rc == 0) {
        const int64_t drained = platform_monotonic_ns();
        run->warmup_drain_s = (drained - now) * 1e-9;
        run->warmup_s = (drained - start) * 1e-9;
        run->warmup_bytes = accepted;
        run->warmup_output_bytes = sink_bytes(meter, throttled, discard);
        measure_start = drained;
        now = drained;
        measuring = 1;
        memset(following, 0, sizeof(following));
      }
    }
    if (measuring && (at_limit || (!limit && (now - measure_start) * 1e-9 >=
                                               cfg->duration_s))) {
      run->start_ns = measure_start;
      run->append_s = (now - measure_start) * 1e-9;
      run->input_bytes = accepted - run->warmup_bytes;
      run->complete_batches = run->input_bytes / run->boundary_bytes[0];
      // Two batch buffers on GPU; this conservative lower bound also applies
      // to CPU. Counts describe input positions, not internal event timestamps.
      run->batch_reuses =
        run->complete_batches > 2 ? run->complete_batches - 2 : 0;
      run->generation_transitions = (accepted - 1) / run->boundary_bytes[1] -
                                    run->warmup_bytes / run->boundary_bytes[1];
      run->coverage_sufficient =
        run->append_s >= 0.25 && run->warmup_s >= 0.25 &&
        run->warmup_bytes / run->boundary_bytes[0] >= 2 &&
        run->complete_batches >= 4 && run->generation_transitions >= 2;
      *total_bytes = accepted;
      return 0;
    }
  }
}

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
  const size_t bpe = dtype_bpe(dtype);
  struct bench_measurement measurement = {
    .boundary_timing = !cfg->no_boundary_timing,
    .reference_frames = dims[0].size,
    .source_bytes = 64u << 20,
    .append_bytes =
      cfg->append_elements ? cfg->append_elements * bpe : 64u << 20,
  };
  if (!measurement.append_bytes ||
      cfg->append_elements > measurement.source_bytes / bpe ||
      measurement.source_bytes % measurement.append_bytes) {
    log_error("Append size must divide the 64 MiB source ring");
    return bench_failed(cfg->json_output);
  }
  uint32_t chosen_epochs_per_batch = 0; // 0 = auto; set by advise_layout on fit

  if (resolve_chunk_sizing(
        cfg, dtype, 0.8, "(restrict to <80%)", &chosen_epochs_per_batch))
    return 1;

  dims_print(dims, rank);
  print_shard_summary(dims, rank, dtype_bpe(dtype));

  const int64_t prep_start = platform_monotonic_ns();
  init_fill_pattern(fill, dims, rank);

  size_t total_elements = dim_total_elements(dims, rank);
  size_t total_bytes = total_elements * bpe;
  unsigned char* source = NULL;

  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  struct bench_zarr_handle zarr = { 0 };
  struct metering_sink meter = { 0 };
  struct throttled_shard_sink tss = { 0 };
  int use_throttled = 0;
  struct shard_sink* sink = &dss.base;
  struct bench_handle h = { .backend = cfg->backend };

  {
    const size_t elements = measurement.source_bytes / bpe;
    source = calloc(elements, bpe > 2 ? bpe : 2);
    CHECK(Fail, source);
    fill((uint16_t*)source, elements, 0, elements);
    free_fill_pattern(fill);
    measurement.prep_s = (platform_monotonic_ns() - prep_start) * 1e-9;
    measurement.boundary_bytes[1] =
      dims[0].chunk_size * dims[0].chunks_per_shard * bpe;
    for (uint8_t d = 1; d < rank; ++d)
      measurement.boundary_bytes[1] *= dims[d].size;
    measurement.rank = rank;
    memcpy(measurement.geometry, dims, rank * sizeof(*dims));
    measurement.requested_frames = cfg->frames;
    measurement.requested_warmup_s = cfg->warmup_s;
    measurement.requested_duration_s = cfg->duration_s;
    measurement.memory_budget = cfg->memory_budget;
    measurement.target_batch_bytes = resolved_batch_bytes(cfg);
    // Run length and append size must not change the fitted geometry.
    dims[0].size = 0;
  }

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
    .target_batch_bytes = resolved_batch_bytes(cfg),
    .backpressure_bytes = cfg->backpressure_bytes,
    .max_threads = cfg->max_threads,
    .full_memcpy_timing = cfg->full_memcpy_timing,
  };

  uint64_t est_total_chunks = 0;
  size_t est_total_bytes = 0;
  size_t est_pinned_bytes = 0;

  if (cfg->backend == BENCH_GPU)
    bench_gpu_report_memory(
      &config, &est_total_chunks, &est_total_bytes, &est_pinned_bytes);

  if (cfg->backend == BENCH_CPU) {
    struct tile_stream_cpu_memory_info mem;
    if (tile_stream_cpu_memory_estimate(&config, 0, &mem) == 0) {
      est_total_chunks = mem.total_chunks;
      est_total_bytes = mem.heap_bytes;
      chosen_epochs_per_batch = mem.epochs_per_batch;
      char a[32], b[32];
      format_bytes(a, sizeof(a), mem.heap_bytes);
      print_report("  %-17s %s heap", "CPU memory:", a);
      format_bytes(a, sizeof(a), mem.chunk_pool_bytes);
      format_bytes(b, sizeof(b), mem.compressed_pool_bytes);
      print_report(
        "    %-12s %12s   %-12s %12s", "Chunk pool:", a, "Compressed:", b);
      format_bytes(a, sizeof(a), mem.comp_sizes_bytes);
      format_bytes(b, sizeof(b), mem.aggregate_bytes);
      print_report(
        "    %-12s %12s   %-12s %12s", "Comp. sizes:", a, "Aggregate:", b);
      format_bytes(a, sizeof(a), mem.host_output_pool_bytes);
      print_report("    %-12s %12s", "Host output:", a);
      format_bytes(a, sizeof(a), mem.lod_bytes);
      format_bytes(b, sizeof(b), mem.shard_bytes);
      print_report("    %-12s %12s   %-12s %12s", "LOD:", a, "Shards:", b);
      print_report(
        "    Chunks:      %llu/epoch, %llu total (%d LOD levels, batch=%u)",
        (unsigned long long)mem.chunks_per_epoch,
        (unsigned long long)mem.total_chunks,
        mem.nlod,
        mem.epochs_per_batch);
    }
  }

  struct bench_memory mem_used = {
    .estimate_total_bytes = est_total_bytes,
    .estimate_pinned_bytes = est_pinned_bytes,
  };
  if (platform_resident_memory(&mem_used.host_baseline_bytes) != 0) {
    log_warn("  host memory reading unavailable");
    mem_used.host_reading_failed = 1;
  }
  const size_t device_free_at_rest =
    cfg->backend == BENCH_GPU ? bench_gpu_free_memory() : 0;

  struct platform_clock init_clock = { 0 };
  platform_toc(&init_clock);

  if (cfg->backend == BENCH_CPU)
    h.cpu = tile_stream_cpu_create(&config, sink);
  else
    h.gpu = bench_gpu_create(&config, sink);
  CHECK(Fail, !bench_is_null(&h));
  float init_s = platform_toc(&init_clock);

  const struct tile_stream_layout* layout = bench_layout(&h);
  if (cfg->backend == BENCH_GPU)
    chosen_epochs_per_batch = bench_gpu_status(h.gpu).epochs_per_batch;
  {
    measurement.epoch_bytes = layout->epoch_elements * bpe;
    measurement.epochs_per_batch = chosen_epochs_per_batch;
    measurement.staging_bytes = config.buffer_capacity_bytes;
    measurement.boundary_bytes[0] =
      layout->epoch_elements * bpe * chosen_epochs_per_batch;
    uint64_t a = measurement.boundary_bytes[0],
             b = config.buffer_capacity_bytes;
    while (b) {
      const uint64_t remainder = a % b;
      a = b;
      b = remainder;
    }
    measurement.boundary_bytes[2] = a;
  }
  size_t max_compressed_size = 0;
  size_t codec_batch_size = 0;
  int nlod = 0;
  if (cfg->backend == BENCH_GPU) {
    struct tile_stream_status st = bench_gpu_status(h.gpu);
    max_compressed_size = st.max_compressed_size;
    codec_batch_size = st.codec_batch_size;
    nlod = st.nlod;
  }

  print_report("  Reference extent below fits geometry, not run length:");
  log_bench_header(layout,
                   config.dtype,
                   config.codec,
                   max_compressed_size,
                   codec_batch_size,
                   total_bytes,
                   total_elements);
  if (is_multiscale && nlod > 0)
    print_report("  %-17s %d", "LOD levels:", nlod);

  if (cfg->append_elements > 0) {
    char abuf[32];
    format_bytes(
      abuf, sizeof(abuf), (uint64_t)(cfg->append_elements * dtype_bpe(dtype)));
    print_report(
      "  %-17s %zu elements = %s", "Append size:", cfg->append_elements, abuf);
  }

  CHECK(Fail,
        pump_measurement(
          &h, source, cfg, &measurement, &meter, &tss, &dss, &total_bytes) ==
          0);
  total_elements = total_bytes / bpe;

  const int64_t drain_start = platform_monotonic_ns();
  measurement.append_s = (drain_start - measurement.start_ns) * 1e-9;
  CHECK(Fail, writer_flush(bench_writer(&h)).error == 0);

  // Final metadata publication belongs to the measured operation too.
  CHECK(Fail, writer_close(bench_writer(&h)).error == 0);

  uint64_t pending_bytes = bench_zarr_pending_bytes(&zarr);

  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  if (bench_zarr_flush(&zarr)) {
    log_error("  I/O error detected during flush");
    goto Fail;
  }
  float flush_s = platform_toc(&flush_clock);
  measurement.drain_s = (platform_monotonic_ns() - drain_start) * 1e-9;
  measurement.elapsed_s = measurement.append_s + measurement.drain_s;
  // A screening budget, not a steady-state precision guarantee: a large
  // endpoint drain means too much of this sample was still in flight.
  measurement.coverage_sufficient &=
    measurement.drain_s <= 0.1 * measurement.elapsed_s;
  const float wall_s = (float)measurement.elapsed_s;
  measurement.output_bytes =
    sink_bytes(&meter, &tss, &dss) - measurement.warmup_output_bytes;

  if (platform_peak_resident_memory(&mem_used.host_peak_bytes) != 0) {
    log_warn("  host peak memory reading unavailable");
    mem_used.host_reading_failed = 1;
  }
  // A 0 baseline is the page counter not started yet, not an empty process.
  const int baseline_is_usable = mem_used.host_baseline_bytes > 0;
  if (cfg->backend == BENCH_GPU) {
    bench_memory_record_device(
      &mem_used, device_free_at_rest, bench_gpu_free_memory());
  } else if (baseline_is_usable &&
             mem_used.host_peak_bytes > mem_used.host_baseline_bytes) {
    mem_used.measured_bytes =
      mem_used.host_peak_bytes - mem_used.host_baseline_bytes;
  }

  if (bench_cursor(&h) != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)bench_cursor(&h),
              (ptrdiff_t)((int64_t)bench_cursor(&h) - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct stream_metrics m = bench_get_metrics(&h);
    const struct sink_stats ss = { .total_bytes = measurement.output_bytes,
                                   .total_chunks = est_total_chunks };
    print_measurement_report(&measurement);
    print_bench_report(&m,
                       layout,
                       config.dtype,
                       &ss,
                       measurement.input_bytes,
                       measurement.input_bytes / bpe,
                       wall_s,
                       init_s,
                       flush_s,
                       pending_bytes);
    print_memory_report(&mem_used);

    if (cfg->json_output) {
      // Pipeline sink metrics share the reset boundary; the wrapper's
      // lifetime metric includes warmup and must not replace them.
      const struct stream_metric* sink_metric = NULL;
      print_bench_json_pass(&m,
                            sink_metric,
                            layout,
                            config.dtype,
                            config.codec,
                            &ss,
                            measurement.input_bytes,
                            measurement.input_bytes / bpe,
                            wall_s,
                            init_s,
                            flush_s,
                            &mem_used,
                            bench_worker_threads(&h),
                            &measurement);
    }
  }

  print_report("  PASS");
  int rc = 0;
  goto Cleanup;

Fail:
  rc = bench_failed(cfg->json_output);

Cleanup:
  // Flush before destroying the stream — pending write_direct jobs
  // reference pinned host buffers owned by the stream.
  bench_zarr_flush(&zarr);
  bench_destroy(&h);
  bench_zarr_close(&zarr);
  if (use_throttled)
    throttled_shard_sink_teardown(&tss);
  free_fill_pattern(fill);
  free(source);
  dims[0].size = measurement.reference_frames;
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
  uint64_t target_chunk_bytes;
  uint64_t target_batch_bytes;
  uint64_t memory_budget;
  uint64_t frames;
  uint64_t geometry_frames;
  int frames_set, duration_set, warmup_set;
  size_t append_elements; // 0 = default block
  int json_output;
  const char* output_path;
  const char* s3_bucket;
  const char* s3_prefix;
  const char* s3_region;
  const char* s3_endpoint;
  double s3_throughput_gbps;
  uint64_t io_bw_mbps;
  uint64_t io_latency_us;
  uint64_t backpressure_bytes;
  int max_threads;
  int full_memcpy_timing;
  double warmup_s;
  double duration_s;
  int no_boundary_timing;
};

static int
read_size(const char* flag, const char* text, uint64_t* out)
{
  if (parse_bytes(text, out))
    return 1;
  fprintf(stderr, "%s: cannot read \"%s\" as a size\n", flag, text);
  return 0;
}

static int
parse_bench_cli_args(int ac, char* av[], struct bench_cli_args* out)
{
  out->fill = fill_xor;
  out->codec = (struct codec_config){ .id = CODEC_ZSTD };
  out->reduce = lod_reduce_mean;
  out->backend = bench_gpu_enabled() ? BENCH_GPU : BENCH_CPU;
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
  out->full_memcpy_timing = 0;

  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--fill") == 0 && i + 1 < ac) {
      out->fill = parse_fill(av[++i]);
      if (!out->fill)
        return 1;
    } else if (strcmp(av[i], "--codec") == 0 && i + 1 < ac) {
      if (!parse_codec(av[++i], &out->codec))
        return 1;
    } else if (strcmp(av[i], "--blosc-block-bytes") == 0 && i + 1 < ac) {
      uint64_t block_bytes = 0;
      if (!read_size(av[i], av[i + 1], &block_bytes) || block_bytes == 0 ||
          block_bytes > UINT32_MAX) {
        fprintf(stderr, "Invalid --blosc-block-bytes: %s\n", av[i + 1]);
        return 1;
      }
      out->codec.blosc_block_bytes = (uint32_t)block_bytes;
      ++i;
    } else if (strcmp(av[i], "--blosc-shuffle") == 0 && i + 1 < ac) {
      const char* shuffle = av[++i];
      if (strcmp(shuffle, "none") == 0)
        out->codec.shuffle = CODEC_SHUFFLE_NONE;
      else if (strcmp(shuffle, "byte") == 0)
        out->codec.shuffle = CODEC_SHUFFLE_BYTE;
      else if (strcmp(shuffle, "bit") == 0)
        out->codec.shuffle = CODEC_SHUFFLE_BIT;
      else {
        fprintf(stderr,
                "Invalid --blosc-shuffle: %s (expected none|byte|bit)\n",
                shuffle);
        return 1;
      }
    } else if (strcmp(av[i], "--reduce") == 0 && i + 1 < ac) {
      if (!parse_reduce(av[++i], &out->reduce))
        return 1;
    } else if (strcmp(av[i], "--backend") == 0 && i + 1 < ac) {
      if (!parse_backend(av[++i], &out->backend))
        return 1;
    } else if (strcmp(av[i], "--dtype") == 0 && i + 1 < ac) {
      if (!parse_dtype(av[++i], &out->dtype))
        return 1;
    } else if ((strcmp(av[i], "--frames") == 0 ||
                strcmp(av[i], "--geometry-frames") == 0) &&
               i + 1 < ac) {
      const int geometry = strcmp(av[i], "--geometry-frames") == 0;
      const char* value = av[++i];
      char* end;
      errno = 0;
      const uint64_t frames = strtoull(value, &end, 10);
      if (end == value || *end || *value == '-' || errno ||
          (geometry && frames == 0)) {
        fprintf(stderr, "Invalid frame count: %s\n", value);
        return 1;
      }
      if (geometry)
        out->geometry_frames = frames;
      else {
        out->frames = frames;
        out->frames_set = 1;
      }
    } else if (strcmp(av[i], "--append-elements") == 0 && i + 1 < ac) {
      char* end;
      const char* value = av[++i];
      errno = 0;
      out->append_elements = (size_t)strtoull(value, &end, 10);
      if (end == value || *end || *value == '-' || errno) {
        fprintf(stderr, "Invalid --append-elements: %s\n", value);
        return 1;
      }
    } else if ((strcmp(av[i], "--duration") == 0 ||
                strcmp(av[i], "--warmup") == 0) &&
               i + 1 < ac) {
      char* end;
      const int duration = strcmp(av[i], "--duration") == 0;
      double seconds = strtod(av[i + 1], &end);
      if (end == av[i + 1] || *end || !isfinite(seconds) || seconds < 0 ||
          (duration && seconds == 0)) {
        fprintf(stderr, "Invalid %s: %s\n", av[i], av[i + 1]);
        return 1;
      }
      *(duration ? &out->duration_s : &out->warmup_s) = seconds;
      *(duration ? &out->duration_set : &out->warmup_set) = 1;
      ++i;
    } else if (strcmp(av[i], "--json") == 0) {
      out->json_output = 1;
    } else if (strcmp(av[i], "--chunk-bytes") == 0 && i + 1 < ac) {
      if (!read_size(av[i], av[i + 1], &out->target_chunk_bytes))
        return 1;
      ++i;
    } else if (strcmp(av[i], "--batch-bytes") == 0 && i + 1 < ac) {
      if (!read_size(av[i], av[i + 1], &out->target_batch_bytes))
        return 1;
      ++i;
    } else if (strcmp(av[i], "--memory-budget") == 0 && i + 1 < ac) {
      if (!read_size(av[i], av[i + 1], &out->memory_budget))
        return 1;
      ++i;
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
      if (!read_size(av[i], av[i + 1], &out->backpressure_bytes))
        return 1;
      ++i;
    } else if (strcmp(av[i], "--max-threads") == 0 && i + 1 < ac) {
      out->max_threads = (int)strtol(av[++i], NULL, 10);
    } else if (strcmp(av[i], "--full-memcpy-timing") == 0) {
      out->full_memcpy_timing = 1;
    } else if (strcmp(av[i], "--no-boundary-timing") == 0) {
      out->no_boundary_timing = 1;
    } else {
      fprintf(stderr, "Unknown option: %s\n", av[i]);
      fprintf(stderr,
              "Usage: %s [--fill xor|zeros|rand] [--codec "
              "none|lz4|zstd|blosc-lz4|blosc-zstd] "
              "[--blosc-block-bytes N (required for Blosc, e.g. 16K)] "
              "[--blosc-shuffle none|byte|bit] "
              "[--reduce mean|min|max|median|max_sup|min_sup] "
              "[--backend gpu|cpu] [--dtype u8|u16|...] "
              "[--geometry-frames N] [--frames N | --duration S] "
              "[--json] [--append-elements N] [--warmup S] "
              "[--no-boundary-timing] "
              "[--chunk-bytes N] [--batch-bytes N] "
              "[--memory-budget N] [-o path] "
              "[--s3-bucket B --s3-region R --s3-endpoint E [--s3-prefix P] "
              "[--s3-throughput-gbps N]] "
              "[--io-bw-mbps N (MiB/s)] [--io-latency-us N] "
              "[--backpressure N (bytes, e.g. 256M)] "
              "[--max-threads N (0 = OpenMP default)] "
              "[--full-memcpy-timing (GPU profiling)]\n",
              av[0]);
      return 1;
    }
  }
  if (out->frames > 0 && out->duration_set) {
    fprintf(stderr, "Use --frames or --duration; fit with --geometry-frames\n");
    return 1;
  }
  return codec_config_validate_blosc(out->codec);
}

int
bench_stream_main(int ac, char* av[], struct bench_spec spec)
{
  struct bench_cli_args a = { 0 };
  if (parse_bench_cli_args(ac, av, &a))
    return 1;

  struct dimension* dims = spec.dims;
  if (a.geometry_frames)
    dims[0].size = a.geometry_frames;
  if (!a.duration_set && !a.frames)
    a.duration_s = 1.0;
  if (!a.warmup_set)
    a.warmup_s = 0.25;

  if (a.backend == BENCH_GPU && bench_gpu_context_create())
    return bench_failed(a.json_output);

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
    .append_elements = a.append_elements,
    .json_output = a.json_output,
    .io_bw_mbps = a.io_bw_mbps,
    .io_latency_us = a.io_latency_us,
    .backpressure_bytes = a.backpressure_bytes,
    .max_threads = a.max_threads,
    .full_memcpy_timing = a.full_memcpy_timing,
    .frames = a.frames,
    .warmup_s = a.warmup_s,
    .duration_s = a.duration_s,
    .no_boundary_timing = a.no_boundary_timing,
  };
  int ecode = run_bench(&cfg);

  if (a.backend == BENCH_GPU)
    bench_gpu_context_destroy();
  return ecode;
}

// ---------------------------------------------------------------------------
// Two-stream benchmark: two GPU pipelines, interleaved append on one thread
// ---------------------------------------------------------------------------

// Interleaved pump: fill one block, then append it to each writer in turn until
// both are full.
static int
pump_data_interleaved(struct writer* w0,
                      struct writer* w1,
                      size_t total_elements,
                      fill_fn fill,
                      size_t bpe,
                      size_t block_elements)
{
  const size_t nelements =
    block_elements > 0 ? block_elements : (size_t)32 * 1024 * 1024;
  size_t alloc = nelements * (bpe > 2 ? bpe : 2);
  uint16_t* data = (uint16_t*)calloc(1, alloc);
  if (!data)
    return 1;
  fill(data,
       nelements < total_elements ? nelements : total_elements,
       0,
       total_elements);

  int err = 0;
  size_t off0 = 0, off1 = 0;

  while (off0 < total_elements || off1 < total_elements) {
    if (off0 < total_elements) {
      size_t n = nelements;
      if (off0 + n > total_elements)
        n = total_elements - off0;
      struct slice input = { .beg = data, .end = (char*)data + n * bpe };
      struct writer_result r = writer_append_wait(w0, input);
      if (r.error) {
        log_error("  stream-0 append failed at offset %zu", off0);
        err = 1;
        break;
      }
      off0 += n;
    }

    if (off1 < total_elements) {
      size_t n = nelements;
      if (off1 + n > total_elements)
        n = total_elements - off1;
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
  // flush only queues the writes; close waits for them, so the measured time
  // covers the drain and a write error is not lost.
  struct writer_result c0 = writer_close(w0);
  struct writer_result c1 = writer_close(w1);
  if (!err)
    err = r0.error || r1.error || c0.error || c1.error;
  free(data);
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
  // Declared before the first goto Fail: the cleanup path destroys them.
  struct tile_stream_gpu* s0 = NULL;
  struct tile_stream_gpu* s1 = NULL;

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
    .target_batch_bytes = resolved_batch_bytes(cfg),
    .backpressure_bytes = cfg->backpressure_bytes,
    .max_threads = cfg->max_threads,
    .full_memcpy_timing = cfg->full_memcpy_timing,
  };

  bench_gpu_report_memory_pair(&config);

  // Create two GPU streams
  struct platform_clock init_clock = { 0 };
  platform_toc(&init_clock);

  s0 = bench_gpu_create(&config, sink[0]);
  s1 = bench_gpu_create(&config, sink[1]);
  CHECK(Fail, s0 && s1);
  float init_s = platform_toc(&init_clock);

  const struct tile_stream_layout* layout = bench_gpu_layout(s0);
  log_bench_header(
    layout, dtype, cfg->codec, 0, 0, total_bytes, total_elements);

  struct writer* w0 = bench_gpu_writer(s0);
  struct writer* w1 = bench_gpu_writer(s1);

  // Interleaved pump
  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  CHECK(Fail,
        pump_data_interleaved(
          w0, w1, total_elements, fill, bpe, cfg->append_elements) == 0);

  // Flush zarr sinks before measuring wall time
  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  for (int k = 0; k < 2; ++k)
    CHECK(Fail, bench_zarr_flush(&zarr[k]) == 0);
  float flush_s = platform_toc(&flush_clock);
  float wall_s = platform_toc(&clock);

  // Verify cursors
  CHECK(Fail, bench_gpu_cursor(s0) == total_elements);
  CHECK(Fail, bench_gpu_cursor(s1) == total_elements);

  // Collect metrics
  struct stream_metrics m[2] = {
    bench_gpu_get_metrics(s0),
    bench_gpu_get_metrics(s1),
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
      "  %-17s %s (%zu elements x 2 streams)", "Input:", buf, total_elements);
    format_bytes(buf, sizeof(buf), (uint64_t)(sink_bytes[0] + sink_bytes[1]));
    print_report("  %-17s %s", "Output:", buf);
  }
  print_report("  %-17s %.3f s", "Init time:", (double)init_s);
  if (flush_s > 0)
    print_report("  %-17s %.3f s", "Flush time:", (double)flush_s);
  print_report("  %-17s %.3f s", "Wall time:", (double)wall_s);
  print_report("  %-17s %.2f GiB/s (combined)",
               "Throughput:",
               wall_s > 0 ? combined_gib / wall_s : 0.0);

  // --- Per-stream reports ---
  for (int k = 0; k < 2; ++k) {
    fputc('\n', stderr);
    print_report("  --- Stream %d ---", k);
    print_report("  %-17s %.2f GiB/s",
                 "Throughput:",
                 wall_s > 0 ? per_stream_gib / wall_s : 0.0);
    char cbuf[32];
    format_bytes(cbuf, sizeof(cbuf), (uint64_t)sink_bytes[k]);
    print_report("  %-17s %s", "Output:", cbuf);
    fputc('\n', stderr);
    print_stage_report(&m[k]);

    print_diagnostics_report(&m[k], wall_s);
  }

  print_report("  PASS");
  bench_gpu_destroy(s1);
  bench_gpu_destroy(s0);
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
    bench_gpu_destroy(s1);
  if (s0)
    bench_gpu_destroy(s0);
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
  struct bench_cli_args a = { 0 };
  if (parse_bench_cli_args(ac, av, &a))
    return 1;
  if (a.duration_set || a.warmup_set || (a.frames_set && !a.frames)) {
    print_report("--duration requires a single-stream benchmark");
    return bench_failed(a.json_output);
  }

  struct dimension* dims = spec.dims;
  if (a.frames > 0)
    dims[0].size = a.frames;

  // This driver reports to stderr only, so an error document would be the
  // only JSON a caller ever saw.
  if (a.json_output)
    print_report("  note: the two-stream benchmark does not write JSON");

  if (bench_gpu_context_create())
    return bench_failed(0);

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
    .append_elements = a.append_elements,
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
    .full_memcpy_timing = a.full_memcpy_timing,
  };
  int ecode = run_bench_two_streams(&cfg);

  bench_gpu_context_destroy();
  return ecode;
}
