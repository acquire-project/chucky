#include "bench_report.h"

#include "util/format_bytes.h"
#include "util/metric.h"
#include "zarr/json_writer.h"

#include <math.h>
#include <string.h>

enum diagnostic_section
{
  DIAGNOSTIC_HOST_BLOCK,
  DIAGNOSTIC_HOST_OVERHEAD,
  DIAGNOSTIC_DEVICE_WORK,
};

struct diagnostic_entry
{
  const char* id;    // stable machine-readable identity
  const char* label; // condition or work visible to a person
  const char* kind;  // distinguishes waits from host and device work
  enum diagnostic_section section;
  const struct stream_metric* metric;
};

#define DIAGNOSTIC_COUNT 12

static void
diagnostic_entries(const struct stream_metrics* m,
                   struct diagnostic_entry out[DIAGNOSTIC_COUNT])
{
  out[0] = (struct diagnostic_entry){ "batch_drain",
                                      "Batch delivery (wait/work)",
                                      "host_block",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->flush_stall };
  out[1] = (struct diagnostic_entry){ "d2h_dispatch",
                                      "D2H dispatch work",
                                      "host_overhead",
                                      DIAGNOSTIC_HOST_OVERHEAD,
                                      &m->delivery_dispatch };
  out[2] = (struct diagnostic_entry){ "footer_buffer_io",
                                      "Footer-buffer write",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->footer_buffer_stall };
  out[3] = (struct diagnostic_entry){ "append_extent_io",
                                      "Closed-shard writes",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->append_extent_stall };
  out[4] = (struct diagnostic_entry){ "final_io",
                                      "Final queued writes",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->flush_writes_stall };
  out[5] = (struct diagnostic_entry){ "sink_backpressure",
                                      "Sink queue below limit",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->backpressure };
  out[6] = (struct diagnostic_entry){ "staging_reuse",
                                      "Staging-buffer reuse",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->edge_stall[0] };
  out[7] = (struct diagnostic_entry){ "chunk_metadata_d2h",
                                      "Chunk metadata ready (inclusive)",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->edge_stall[1] };
  out[8] = (struct diagnostic_entry){ "indexed_aggregate_wait",
                                      "Aggregate dependency before metadata",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->indexed_aggregate_wait };
  out[9] = (struct diagnostic_entry){ "chunk_metadata_wait",
                                      "Metadata ready after aggregate",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->chunk_metadata_wait };
  out[10] = (struct diagnostic_entry){ "chunk_metadata_copy",
                                       "Chunk offsets/sizes D2H copies",
                                       "device_work",
                                       DIAGNOSTIC_DEVICE_WORK,
                                       &m->chunk_metadata_copy };
  out[11] = (struct diagnostic_entry){ "payload_d2h",
                                       "Payload D2H",
                                       "host_wait",
                                       DIAGNOSTIC_HOST_BLOCK,
                                       &m->edge_stall[2] };
}

// --- Throughput helpers ---

double
gb_per_s(double bytes, double ms)
{
  if (ms <= 0)
    return 0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// --- Report + pipeline helpers ---

// Keep table cells bounded without rounding a nonzero measurement to zero.
// Units stay in the headers, including when scientific notation is needed.
static void
format_measurement(char buf[32], double value, int decimals)
{
  int n = snprintf(buf, 32, "%.*f", decimals, value);
  if (n > 8 || (value != 0 && fabs(value) < (decimals == 3 ? 0.001 : 0.01)))
    snprintf(buf, 32, "%.2e", value);
}

static void
format_count(char buf[32], uint64_t count)
{
  // Only diagnostic table counts are compacted; copy-work and append totals
  // retain exact decimal counts in their own, wider summaries.
  int n = snprintf(buf, 32, "%llu", (unsigned long long)count);
  if (n > 10)
    snprintf(buf, 32, "%.3e", (double)count);
}

void
print_append_latency(const struct stream_metrics* m)
{
  if (m->append_count == 0) {
    char max_ms[32];
    format_measurement(max_ms, m->max_append_ms, 3);
    print_report("  %-17s %s ms", "Max append:", max_ms);
    return;
  }
  char p50[32], p90[32], p99[32], p999[32], max_ms[32];
  format_measurement(p50, append_ms_at(m, 0.50), 3);
  format_measurement(p90, append_ms_at(m, 0.90), 3);
  format_measurement(p99, append_ms_at(m, 0.99), 3);
  format_measurement(p999, append_ms_at(m, 0.999), 3);
  format_measurement(max_ms, m->max_append_ms, 3);
  print_report("  %-17s %llu", "Appends:", (unsigned long long)m->append_count);
  print_report("  %9s %9s %9s %9s %9s",
               "p50 ms",
               "p90 ms",
               "p99 ms",
               "p99.9 ms",
               "max ms");
  print_report("  %9s %9s %9s %9s %9s", p50, p90, p99, p999, max_ms);
}

void
print_memory_report(const struct bench_memory* mem)
{
  char a[32], b[32];
  fputc('\n', stderr);
  if (!mem->host_reading_failed) {
    format_bytes(a, sizeof(a), mem->host_baseline_bytes);
    format_bytes(b, sizeof(b), mem->host_peak_bytes);
    print_report("  %-17s %s at rest, %s peak", "Host memory:", a, b);
  } else {
    print_report("  %-17s unavailable", "Host memory:");
  }
  if (mem->device_used_bytes) {
    format_bytes(a, sizeof(a), mem->device_used_bytes);
    print_report("  %-17s %s", "Device memory:", a);
  }
  if (mem->device_overhead_valid) {
    const int negative = mem->device_overhead_bytes < 0;
    const uint64_t magnitude = negative
                                 ? 0 - (uint64_t)mem->device_overhead_bytes
                                 : (uint64_t)mem->device_overhead_bytes;
    format_bytes(a, sizeof(a), magnitude);
    print_report("  %-17s %c%s (observed minus estimated)",
                 "Device overhead:",
                 negative ? '-' : '+',
                 a);
  }
  if (mem->measured_bytes && mem->estimate_total_bytes) {
    format_bytes(a, sizeof(a), mem->estimate_total_bytes);
    print_report("  %-17s %s (%.2fx measured)",
                 "Estimate:",
                 a,
                 (double)mem->estimate_total_bytes /
                   (double)mem->measured_bytes);
  }
}

static void
print_metric_row(const char* name, const struct stream_metric* m)
{
  if (m->count <= 0)
    return;
  const int N = m->count;
  double avg_ms = (double)m->ms / N;
  double avg_gbs = gb_per_s(m->input_bytes, (double)m->ms);
  int has_best = m->best_ms < 1e29f;
  char avg_rate[32], best_rate[32] = "-", avg_time[32], best_time[32] = "-";
  format_measurement(avg_rate, avg_gbs, 2);
  format_measurement(avg_time, avg_ms, 3);

  if (has_best) {
    // Use the bytes recorded for that exact call so partial-batch tails
    // don't inflate "best" via an average-bytes / min-time fudge.
    double best_gbs = gb_per_s(m->best_input_bytes, (double)m->best_ms);
    format_measurement(best_rate, best_gbs, 2);
    format_measurement(best_time, m->best_ms, 3);
  }
  print_report("  %-14s %10s %10s %9s %9s",
               name,
               avg_rate,
               best_rate,
               avg_time,
               best_time);
}

void
print_stage_report(const struct stream_metrics* m)
{
  const int sampled = m->memcpy_calls > (uint64_t)m->memcpy.count;
  print_report("  %-14s %10s %10s %9s %9s",
               "Stage",
               "avg GiB/s",
               "best GiB/s",
               "avg ms",
               "best ms");
  print_metric_row(sampled ? "Memcpy[smp]" : "Memcpy", &m->memcpy);
  print_metric_row("H2D", &m->h2d);
  print_metric_row(m->scatter.name && strcmp(m->scatter.name, "Copy") == 0
                     ? "Copy"
                     : "Scatter",
                   &m->scatter);
  print_metric_row("LOD gather", &m->lod_gather);
  print_metric_row("LOD reduce", &m->lod_reduce);
  print_metric_row("Append fold", &m->lod_append_fold);
  print_metric_row("LOD to chunks", &m->lod_morton_chunk);
  print_metric_row("Compress", &m->compress);
  print_metric_row("Aggregate", &m->aggregate);
  print_metric_row("D2H", &m->d2h);
  print_metric_row("Sink", &m->sink);

  if (m->memcpy_calls) {
    fputc('\n', stderr);
    print_report("  Memcpy work:");
    print_report("  %-14s %20s %20s", "Coverage", "copies", "bytes");
    print_report("  %-14s %20llu %20llu",
                 "Total",
                 (unsigned long long)m->memcpy_calls,
                 (unsigned long long)m->memcpy_bytes);
    print_report(
      "  %-14s %20d %20.0f", "Timed", m->memcpy.count, m->memcpy.input_bytes);
    print_report("  %s",
                 sampled ? "Sample times/rates/extrema only; not extrapolated."
                         : "All copies timed.");
  }
}

static int
diagnostic_measured(const struct stream_metric* m)
{
  return m->count > 0 || m->wait_calls > 0;
}

static void
print_diagnostic_row(const struct diagnostic_entry* d, float wall_s)
{
  const struct stream_metric* m = d->metric;
  const double wall_pct = wall_s > 0 ? (double)m->ms / (wall_s * 10.0) : 0.0;
  char wait_calls[32] = "-", avg_ms[32] = "-", max_ms[32] = "-", pct[32];
  if (m->wait_calls > 0)
    format_count(wait_calls, m->wait_calls);
  format_measurement(pct, wall_pct, 2);
  if (m->count > 0) {
    format_measurement(avg_ms, (double)m->ms / m->count, 3);
    format_measurement(max_ms, m->max_ms, 3);
  }
  // Long conditions get a continuation row, never truncated labels or shifted
  // numeric columns. Owners are grouped above the rows to leave room for data.
  const char* label = d->label;
  if (strlen(label) > 26) {
    print_report("  %s", label);
    label = "";
  }
  print_report("  %-26s %10d %10s %9s %9s %8s",
               label,
               m->count,
               wait_calls,
               avg_ms,
               max_ms,
               pct);
}

static void
print_diagnostic_section(const struct diagnostic_entry entries[],
                         enum diagnostic_section section,
                         const char* title,
                         const char* interval_label,
                         float wall_s)
{
  int have_section = 0;
  for (size_t i = 0; i < DIAGNOSTIC_COUNT; ++i)
    if (entries[i].section == section && diagnostic_measured(entries[i].metric))
      have_section = 1;
  if (!have_section)
    return;

  fputc('\n', stderr);
  print_report("  --- %s ---", title);
  print_report("  %-26s %10s %10s %9s %9s %8s",
               interval_label,
               "samples",
               "waits",
               "avg ms",
               "max ms",
               "% wall");
  for (enum metric_owner owner = METRIC_OWNER_NONE; owner <= METRIC_OWNER_D2H;
       ++owner) {
    int have_owner = 0;
    for (size_t i = 0; i < DIAGNOSTIC_COUNT; ++i) {
      if (entries[i].section != section || entries[i].metric->owner != owner ||
          !diagnostic_measured(entries[i].metric))
        continue;
      if (!have_owner) {
        print_report("  [%s timeline]", metric_owner_name(owner));
        have_owner = 1;
      }
      print_diagnostic_row(&entries[i], wall_s);
    }
  }
}

static int
delivery_timing_measured(const struct delivery_timing* timing)
{
  return timing->submitted_to_start.count > 0 ||
         timing->start_to_payload_ready.count > 0 ||
         timing->payload_ready_to_writes_posted.count > 0 ||
         timing->submitted_to_slot_reuse.count > 0;
}

static void
print_duration_row(const char* label, const struct duration_stats* stats)
{
  if (stats->count == 0)
    return;
  char count[32], avg_ms[32], min_ms[32], max_ms[32];
  format_count(count, stats->count);
  format_measurement(avg_ms, (double)stats->total_ms / stats->count, 3);
  format_measurement(min_ms, stats->min_ms, 3);
  format_measurement(max_ms, stats->max_ms, 3);
  print_report(
    "  %-34s %10s %9s %9s %9s", label, count, avg_ms, min_ms, max_ms);
}

static void
print_delivery_timing(const struct delivery_timing* timing)
{
  if (!delivery_timing_measured(timing))
    return;

  fputc('\n', stderr);
  print_report("  --- Delivery latency ---");
  print_report("  %-34s %10s %9s %9s %9s",
               "Interval",
               "samples",
               "avg ms",
               "min ms",
               "max ms");
  print_duration_row("Submitted to worker start", &timing->submitted_to_start);
  print_duration_row("Delivery start to payload ready",
                     &timing->start_to_payload_ready);
  print_duration_row("Payload ready to writes posted",
                     &timing->payload_ready_to_writes_posted);
  print_duration_row("Submitted to slot reuse",
                     &timing->submitted_to_slot_reuse);
}

void
print_diagnostics_report(const struct stream_metrics* m, float wall_s)
{
  struct diagnostic_entry entries[DIAGNOSTIC_COUNT];
  diagnostic_entries(m, entries);
  print_diagnostic_section(entries,
                           DIAGNOSTIC_HOST_BLOCK,
                           "Host blocking",
                           "Awaited condition",
                           wall_s);
  print_diagnostic_section(entries,
                           DIAGNOSTIC_HOST_OVERHEAD,
                           "Host overhead",
                           "Measured work",
                           wall_s);
  print_diagnostic_section(
    entries, DIAGNOSTIC_DEVICE_WORK, "Device work", "Measured work", wall_s);
  print_delivery_timing(&m->delivery);

  if (m->scatter_samples_lost || m->lod_samples_lost) {
    fputc('\n', stderr);
    print_report("  TIMING SAMPLES LOST (stage totals under-report)");
    print_report(
      "  %-17s %llu", "Scatter:", (unsigned long long)m->scatter_samples_lost);
    print_report(
      "  %-17s %llu", "LOD:", (unsigned long long)m->lod_samples_lost);
  }
  if (m->append_count > 0 || m->max_append_ms > 0) {
    fputc('\n', stderr);
    print_report("  --- Append latency ---");
    print_append_latency(m);
  }
  if (m->peak_pending_bytes > 0) {
    fputc('\n', stderr);
    print_report("  --- Queue pressure ---");
    char pbuf[32];
    format_bytes(pbuf, sizeof(pbuf), m->peak_pending_bytes);
    print_report("  %-17s %s", "Peak pending:", pbuf);
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

  char buf[32];
  format_bytes(buf, sizeof(buf), (uint64_t)total_bytes);
  print_report("  %-17s %s (%zu elements, %zu epochs)",
               "Total:",
               buf,
               total_elements,
               num_epochs);
  format_bytes(
    buf, sizeof(buf), (uint64_t)(layout->chunk_stride * dtype_bpe(dtype)));
  print_report("  %-17s %lu elements = %s (stride=%lu)",
               "Chunk:",
               (unsigned long)layout->chunk_elements,
               buf,
               (unsigned long)layout->chunk_stride);
  format_bytes(buf, sizeof(buf), (uint64_t)layout->chunk_pool_bytes);
  print_report("  %-17s %lu slots, %s pool",
               "Epoch:",
               (unsigned long)layout->chunks_per_epoch,
               buf);
  if (codec.id != CODEC_NONE && max_compressed_size > 0) {
    format_bytes(
      buf, sizeof(buf), (uint64_t)(codec_batch_size * max_compressed_size));
    print_report("  %-17s %zu bytes/chunk max, %s pool",
                 "Compression:",
                 max_compressed_size,
                 buf);
  }
  if (codec_is_blosc(codec.id))
    print_report(
      "  %-17s %u bytes (requested)", "Blosc block:", codec.blosc_block_bytes);
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
                   uint64_t flush_pending_bytes)
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

  fputc('\n', stderr);
  print_report("  --- Benchmark results ---");
  char fbuf[32];
  format_bytes(fbuf, sizeof(fbuf), (uint64_t)total_bytes);
  print_report("  %-17s %s (%zu elements)", "Input:", fbuf, total_elements);
  format_bytes(fbuf, sizeof(fbuf), (uint64_t)ss->total_bytes);
  print_report("  %-17s %s (ratio: %.3f)", "Output:", fbuf, comp_ratio);
  print_report("  %-17s %zu (%llu/epoch x %zu epochs)",
               "Chunks:",
               total_chunks,
               (unsigned long long)chunks_per_epoch,
               num_epochs);

  fputc('\n', stderr);
  print_stage_report(metrics);

  if (metrics->d2h_payload_bytes_transferred ||
      metrics->d2h_metadata_bytes_transferred ||
      metrics->d2h_payload_copy_count) {
    char payload[32], metadata[32];
    format_bytes(
      payload, sizeof(payload), metrics->d2h_payload_bytes_transferred);
    format_bytes(
      metadata, sizeof(metadata), metrics->d2h_metadata_bytes_transferred);
    fputc('\n', stderr);
    print_report("  --- D2H transfer ---");
    print_report("  %-17s %s", "Payload:", payload);
    print_report("  %-17s %s", "Metadata:", metadata);
    print_report("  %-17s %llu",
                 "Payload copies:",
                 (unsigned long long)metrics->d2h_payload_copy_count);
  }
  if (metrics->shard_padding_logical_payload_bytes ||
      metrics->shard_padding_internal_bytes ||
      metrics->shard_padding_physical_update_count) {
    const uint64_t physical = metrics->shard_padding_logical_payload_bytes +
                              metrics->shard_padding_internal_bytes;
    const double ratio =
      physical > 0
        ? (double)metrics->shard_padding_internal_bytes / (double)physical
        : 0.0;
    char logical[32], padding[32], physical_buf[32];
    format_bytes(
      logical, sizeof(logical), metrics->shard_padding_logical_payload_bytes);
    format_bytes(
      padding, sizeof(padding), metrics->shard_padding_internal_bytes);
    format_bytes(physical_buf, sizeof(physical_buf), physical);
    fputc('\n', stderr);
    print_report("  --- Shard layout ---");
    print_report("  %-17s %s", "Logical payload:", logical);
    print_report(
      "  %-17s %s (%.2f%% of physical)", "Padding:", padding, ratio * 100.0);
    print_report("  %-17s %s", "Physical payload:", physical_buf);
    print_report(
      "  %-17s %llu / %llu",
      "Padded updates:",
      (unsigned long long)metrics->shard_padding_padded_update_count,
      (unsigned long long)metrics->shard_padding_physical_update_count);
  }

  print_diagnostics_report(metrics, wall_s);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  fputc('\n', stderr);
  print_report("  %-17s %.3f s", "Init time:", (double)init_s);
  if (flush_pending_bytes > 0 && flush_s > 0) {
    double flush_gib =
      ((double)flush_pending_bytes / (1024.0 * 1024.0 * 1024.0)) /
      (double)flush_s;
    print_report(
      "  %-17s %.3f s (%.2f GiB/s)", "Flush time:", (double)flush_s, flush_gib);
  } else {
    print_report("  %-17s %.3f s", "Flush time:", (double)flush_s);
  }
  print_report("  %-17s %.3f s", "Wall time:", wall_s);
  print_report("  %-17s %.2f GiB/s", "Throughput:", throughput_gib);
}

static void
json_stage_metric(struct json_writer* jw,
                  const char* name,
                  const struct stream_metric* sm)
{
  if (sm->count <= 0)
    return;
  double avg_ms = (double)sm->ms / sm->count;
  double in_gibs = gb_per_s(sm->input_bytes, (double)sm->ms);
  double out_gibs = gb_per_s(sm->output_bytes, (double)sm->ms);
  jw_key(jw, name);
  jw_object_begin(jw);
  // The measurement belongs to this timeline. Times may be summed only within
  // one owner.
  jw_key(jw, "owner");
  jw_string(jw, metric_owner_name(sm->owner));
  jw_key(jw, "total_ms");
  jw_float(jw, (double)sm->ms);
  jw_key(jw, "count");
  jw_uint(jw, (uint64_t)sm->count);
  jw_key(jw, "in_bytes");
  jw_uint(jw, (uint64_t)sm->input_bytes);
  jw_key(jw, "out_bytes");
  jw_uint(jw, (uint64_t)sm->output_bytes);
  jw_key(jw, "avg_ms");
  jw_float(jw, avg_ms);
  if (sm->best_ms < 1e29f) {
    jw_key(jw, "best_ms");
    jw_float(jw, (double)sm->best_ms);
    jw_key(jw, "best_in_gibs");
    jw_float(jw, gb_per_s(sm->best_input_bytes, (double)sm->best_ms));
    jw_key(jw, "best_out_gibs");
    jw_float(jw, gb_per_s(sm->best_output_bytes, (double)sm->best_ms));
  }
  jw_key(jw, "in_gibs");
  jw_float(jw, in_gibs);
  jw_key(jw, "out_gibs");
  jw_float(jw, out_gibs);
  jw_object_end(jw);
}

static void
json_diagnostic_metric(struct json_writer* jw,
                       const struct diagnostic_entry* d,
                       float wall_s)
{
  const struct stream_metric* m = d->metric;
  if (!diagnostic_measured(m))
    return;

  jw_key(jw, d->id);
  jw_object_begin(jw);
  jw_key(jw, "label");
  jw_string(jw, d->label);
  jw_key(jw, "kind");
  jw_string(jw, d->kind);
  jw_key(jw, "owner");
  jw_string(jw, metric_owner_name(m->owner));
  jw_key(jw, "total_ms");
  jw_float(jw, (double)m->ms);
  jw_key(jw, "samples");
  jw_uint(jw, (uint64_t)m->count);
  if (m->wait_calls > 0) {
    jw_key(jw, "wait_calls");
    jw_uint(jw, m->wait_calls);
  }
  if (m->count > 0) {
    jw_key(jw, "avg_ms");
    jw_float(jw, (double)m->ms / m->count);
    if (m->best_ms < 1e29f) {
      jw_key(jw, "min_ms");
      jw_float(jw, (double)m->best_ms);
    }
    jw_key(jw, "max_ms");
    jw_float(jw, (double)m->max_ms);
  }
  if (wall_s > 0) {
    jw_key(jw, "wall_pct");
    jw_float(jw, (double)m->ms / (wall_s * 10.0));
  }
  jw_object_end(jw);
}

static void
json_diagnostics(struct json_writer* jw,
                 const struct stream_metrics* m,
                 float wall_s)
{
  struct diagnostic_entry entries[DIAGNOSTIC_COUNT];
  diagnostic_entries(m, entries);

  jw_key(jw, "diagnostics");
  jw_object_begin(jw);
  for (size_t i = 0; i < DIAGNOSTIC_COUNT; ++i)
    json_diagnostic_metric(jw, &entries[i], wall_s);
  jw_object_end(jw);
}

static void
json_duration_stats(struct json_writer* jw,
                    const char* id,
                    const char* label,
                    const struct duration_stats* stats)
{
  if (stats->count == 0)
    return;
  jw_key(jw, id);
  jw_object_begin(jw);
  jw_key(jw, "label");
  jw_string(jw, label);
  jw_key(jw, "total_ms");
  jw_float(jw, (double)stats->total_ms);
  jw_key(jw, "samples");
  jw_uint(jw, stats->count);
  jw_key(jw, "avg_ms");
  jw_float(jw, (double)stats->total_ms / stats->count);
  jw_key(jw, "min_ms");
  jw_float(jw, (double)stats->min_ms);
  jw_key(jw, "max_ms");
  jw_float(jw, (double)stats->max_ms);
  jw_object_end(jw);
}

static void
json_delivery_timing(struct json_writer* jw,
                     const struct delivery_timing* timing)
{
  if (!delivery_timing_measured(timing))
    return;
  jw_key(jw, "delivery_timing");
  jw_object_begin(jw);
  json_duration_stats(jw,
                      "submitted_to_start",
                      "Submitted to worker start",
                      &timing->submitted_to_start);
  json_duration_stats(jw,
                      "start_to_payload_ready",
                      "Delivery start to payload ready",
                      &timing->start_to_payload_ready);
  json_duration_stats(jw,
                      "payload_ready_to_writes_posted",
                      "Payload ready to writes posted",
                      &timing->payload_ready_to_writes_posted);
  json_duration_stats(jw,
                      "submitted_to_slot_reuse",
                      "Submitted to slot reuse",
                      &timing->submitted_to_slot_reuse);
  jw_object_end(jw);
}

void
print_bench_json_pass(const struct stream_metrics* m,
                      const struct stream_metric* sink_metric,
                      const struct tile_stream_layout* layout,
                      enum dtype dtype,
                      struct codec_config codec,
                      const struct sink_stats* ss,
                      size_t total_bytes,
                      size_t total_elements,
                      float wall_s,
                      float init_s,
                      float flush_s,
                      const struct bench_memory* mem,
                      int worker_threads)
{
  const size_t chunk_bytes = layout->chunk_stride * dtype_bpe(dtype);
  const size_t num_epochs =
    (total_elements + layout->epoch_elements - 1) / layout->epoch_elements;
  const uint64_t chunks_per_epoch =
    ss->total_chunks ? ss->total_chunks : layout->chunks_per_epoch;
  const size_t total_chunks = num_epochs * chunks_per_epoch;
  const size_t total_decompressed = total_chunks * chunk_bytes;
  const double comp_fold =
    ss->total_bytes > 0 ? (double)total_decompressed / (double)ss->total_bytes
                        : 0.0;
  const double GIB = 1024.0 * 1024.0 * 1024.0;
  const double input_gib = (double)total_bytes / GIB;
  const double compressed_gib = (double)ss->total_bytes / GIB;
  const double throughput_gib = wall_s > 0 ? input_gib / wall_s : 0.0;
  const double throughput_out_gib = wall_s > 0 ? compressed_gib / wall_s : 0.0;

  struct strbuf json_buf = { 0 };
  struct json_writer jw;
  jw_init(&jw, &json_buf);

  jw_object_begin(&jw);
  jw_key(&jw, "status");
  jw_string(&jw, "pass");
  if (codec_is_blosc(codec.id)) {
    jw_key(&jw, "blosc_block_bytes");
    jw_uint(&jw, codec.blosc_block_bytes);
    jw_key(&jw, "blosc_shuffle");
    jw_string(&jw,
              codec.shuffle == CODEC_SHUFFLE_BIT    ? "bit"
              : codec.shuffle == CODEC_SHUFFLE_BYTE ? "byte"
                                                    : "none");
    jw_key(&jw, "blosc_level");
    jw_uint(&jw, codec.level);
  }
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
  jw_key(&jw, "memory_estimate_total_bytes");
  jw_uint(&jw, mem->estimate_total_bytes);
  jw_key(&jw, "memory_estimate_pinned_bytes");
  jw_uint(&jw, mem->estimate_pinned_bytes);
  jw_key(&jw, "memory_host_baseline_bytes");
  jw_uint(&jw, mem->host_baseline_bytes);
  jw_key(&jw, "memory_host_peak_bytes");
  jw_uint(&jw, mem->host_peak_bytes);
  jw_key(&jw, "memory_host_reading_failed");
  jw_bool(&jw, mem->host_reading_failed);
  jw_key(&jw, "memory_device_used_bytes");
  jw_uint(&jw, mem->device_used_bytes);
  jw_key(&jw, "memory_device_overhead_bytes");
  if (mem->device_overhead_valid)
    jw_int(&jw, mem->device_overhead_bytes);
  else
    jw_null(&jw);
  jw_key(&jw, "memory_measured_bytes");
  jw_uint(&jw, mem->measured_bytes);
  jw_key(&jw, "worker_threads");
  jw_uint(&jw, (uint64_t)worker_threads);

  if (m->d2h_payload_bytes_transferred || m->d2h_metadata_bytes_transferred ||
      m->d2h_payload_copy_count) {
    jw_key(&jw, "d2h_transfer");
    jw_object_begin(&jw);
    jw_key(&jw, "payload_bytes_transferred");
    jw_uint(&jw, m->d2h_payload_bytes_transferred);
    jw_key(&jw, "metadata_bytes_transferred");
    jw_uint(&jw, m->d2h_metadata_bytes_transferred);
    jw_key(&jw, "payload_copy_count");
    jw_uint(&jw, m->d2h_payload_copy_count);
    jw_object_end(&jw);
  }

  if (m->shard_padding_logical_payload_bytes ||
      m->shard_padding_internal_bytes ||
      m->shard_padding_physical_update_count) {
    jw_key(&jw, "shard_padding");
    jw_object_begin(&jw);
    jw_key(&jw, "logical_payload_bytes");
    jw_uint(&jw, m->shard_padding_logical_payload_bytes);
    jw_key(&jw, "internal_padding_bytes");
    jw_uint(&jw, m->shard_padding_internal_bytes);
    jw_key(&jw, "physical_shard_update_count");
    jw_uint(&jw, m->shard_padding_physical_update_count);
    jw_key(&jw, "padded_update_count");
    jw_uint(&jw, m->shard_padding_padded_update_count);
    jw_object_end(&jw);
  }

  if (m->memcpy_calls) {
    jw_key(&jw, "memcpy_work");
    jw_object_begin(&jw);
    jw_key(&jw, "calls");
    jw_uint(&jw, m->memcpy_calls);
    jw_key(&jw, "bytes");
    jw_uint(&jw, m->memcpy_bytes);
    jw_key(&jw, "timing_scope");
    jw_string(&jw,
              m->memcpy_calls > (uint64_t)m->memcpy.count ? "sampled" : "full");
    jw_object_end(&jw);
  }

  jw_key(&jw, "stages");
  jw_object_begin(&jw);
  // A separate key prevents existing consumers from treating a sampled time
  // sum as the full stage duration. All fields in this row refer to samples.
  json_stage_metric(
    &jw,
    m->memcpy_calls > (uint64_t)m->memcpy.count ? "memcpy_sample" : "memcpy",
    &m->memcpy);
  json_stage_metric(&jw, "h2d", &m->h2d);
  json_stage_metric(&jw, "scatter", &m->scatter);
  json_stage_metric(&jw, "lod_gather", &m->lod_gather);
  json_stage_metric(&jw, "lod_reduce", &m->lod_reduce);
  json_stage_metric(&jw, "lod_append_fold", &m->lod_append_fold);
  json_stage_metric(&jw, "lod_morton_chunk", &m->lod_morton_chunk);
  json_stage_metric(&jw, "compress", &m->compress);
  json_stage_metric(&jw, "aggregate", &m->aggregate);
  json_stage_metric(&jw, "d2h", &m->d2h);
  if (sink_metric)
    json_stage_metric(&jw, "sink", sink_metric);
  jw_object_end(&jw);

  jw_key(&jw, "stalls");
  jw_object_begin(&jw);
  jw_key(&jw, "flush_stall_ms");
  jw_float(&jw, (double)m->flush_stall.ms);
  jw_key(&jw, "flush_stall_count");
  jw_uint(&jw, (uint64_t)m->flush_stall.count);
  jw_key(&jw, "drain_dispatch_ms");
  jw_float(&jw, (double)m->delivery_dispatch.ms);
  jw_key(&jw, "drain_dispatch_count");
  jw_uint(&jw, (uint64_t)m->delivery_dispatch.count);
  jw_key(&jw, "footer_buffer_ms");
  jw_float(&jw, (double)m->footer_buffer_stall.ms);
  jw_key(&jw, "footer_buffer_count");
  jw_uint(&jw, (uint64_t)m->footer_buffer_stall.count);
  jw_key(&jw, "append_extent_ms");
  jw_float(&jw, (double)m->append_extent_stall.ms);
  jw_key(&jw, "append_extent_count");
  jw_uint(&jw, (uint64_t)m->append_extent_stall.count);
  jw_key(&jw, "flush_writes_ms");
  jw_float(&jw, (double)m->flush_writes_stall.ms);
  jw_key(&jw, "flush_writes_count");
  jw_uint(&jw, (uint64_t)m->flush_writes_stall.count);
  jw_key(&jw, "backpressure_ms");
  jw_float(&jw, (double)m->backpressure.ms);
  jw_key(&jw, "backpressure_count");
  jw_uint(&jw, (uint64_t)m->backpressure.count);
  // Non-zero means the stage totals above are under-reported.
  jw_key(&jw, "scatter_samples_lost");
  jw_uint(&jw, m->scatter_samples_lost);
  jw_key(&jw, "lod_samples_lost");
  jw_uint(&jw, m->lod_samples_lost);
  // Keyed by metric name.
  jw_key(&jw, "owners");
  jw_object_begin(&jw);
  jw_key(&jw, "flush_stall");
  jw_string(&jw, metric_owner_name(m->flush_stall.owner));
  jw_key(&jw, "drain_dispatch");
  jw_string(&jw, metric_owner_name(m->delivery_dispatch.owner));
  jw_key(&jw, "footer_buffer");
  jw_string(&jw, metric_owner_name(m->footer_buffer_stall.owner));
  jw_key(&jw, "append_extent");
  jw_string(&jw, metric_owner_name(m->append_extent_stall.owner));
  jw_key(&jw, "flush_writes");
  jw_string(&jw, metric_owner_name(m->flush_writes_stall.owner));
  jw_key(&jw, "backpressure");
  jw_string(&jw, metric_owner_name(m->backpressure.owner));
  jw_object_end(&jw);
  jw_key(&jw, "edge_stalls");
  jw_object_begin(&jw);
  // These keys shipped before display names and stable metric IDs were
  // separated. Keep them frozen for existing JSON consumers.
  static const char* legacy_edge_names[] = { "StagingFree",
                                             "ChunkIndex",
                                             "D2HDone" };
  for (size_t i = 0; i < sizeof(m->edge_stall) / sizeof(m->edge_stall[0]);
       ++i) {
    const struct stream_metric* es = &m->edge_stall[i];
    if (es->count <= 0 || !es->name)
      continue;
    jw_key(&jw, legacy_edge_names[i]);
    jw_object_begin(&jw);
    jw_key(&jw, "owner");
    jw_string(&jw, metric_owner_name(es->owner));
    jw_key(&jw, "total_ms");
    jw_float(&jw, (double)es->ms);
    jw_key(&jw, "count");
    jw_uint(&jw, (uint64_t)es->count);
    jw_object_end(&jw);
  }
  jw_object_end(&jw);
  if (m->append_count > 0) {
    jw_key(&jw, "append_ms_p50");
    jw_float(&jw, (double)append_ms_at(m, 0.50));
    jw_key(&jw, "append_ms_p90");
    jw_float(&jw, (double)append_ms_at(m, 0.90));
    jw_key(&jw, "append_ms_p99");
    jw_float(&jw, (double)append_ms_at(m, 0.99));
    jw_key(&jw, "append_ms_p999");
    jw_float(&jw, (double)append_ms_at(m, 0.999));
  }
  // The buckets themselves, so a reader can ask their own question — such as
  // how many appends missed their frame budget. No single percentile answers
  // that, because where the slow tail starts depends on the append size.
  jw_key(&jw, "append_ms_histogram");
  jw_array_begin(&jw);
  for (int i = 0; i < APPEND_LATENCY_BUCKETS; ++i) {
    if (m->append_ms_buckets[i] == 0)
      continue;
    jw_object_begin(&jw);
    jw_key(&jw, "upto_ms");
    jw_float(&jw, (double)append_bucket_ms(m, i));
    jw_key(&jw, "n");
    jw_uint(&jw, m->append_ms_buckets[i]);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);
  jw_key(&jw, "append_count");
  jw_uint(&jw, m->append_count);
  jw_key(&jw, "max_append_ms");
  jw_float(&jw, (double)m->max_append_ms);
  jw_key(&jw, "peak_pending_mib");
  jw_float(&jw, (double)m->peak_pending_bytes / (1024.0 * 1024.0));
  jw_object_end(&jw);

  json_diagnostics(&jw, m, wall_s);
  json_delivery_timing(&jw, &m->delivery);
  jw_object_end(&jw);
  printf("%s\n", strbuf_cstr(&json_buf));
  strbuf_free(&json_buf);
}

void
print_bench_json_error(void)
{
  struct strbuf buf = { 0 };
  struct json_writer jw;
  jw_init(&jw, &buf);
  jw_object_begin(&jw);
  jw_key(&jw, "status");
  jw_string(&jw, "error");
  jw_object_end(&jw);
  printf("%s\n", strbuf_cstr(&buf));
  strbuf_free(&buf);
}
