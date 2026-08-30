#include "bench_report.h"

#include "util/format_bytes.h"
#include "util/metric.h"
#include "zarr/json_writer.h"

enum diagnostic_section
{
  DIAGNOSTIC_HOST_BLOCK,
  DIAGNOSTIC_PIPELINE_GAP,
  DIAGNOSTIC_HOST_OVERHEAD,
};

struct diagnostic_entry
{
  const char* id;    // stable machine-readable identity
  const char* label; // condition or work visible to a person
  const char* kind;  // distinguishes waits from gaps and host work
  enum diagnostic_section section;
  const struct stream_metric* metric;
};

#define DIAGNOSTIC_COUNT 11

static void
diagnostic_entries(const struct stream_metrics* m,
                   struct diagnostic_entry out[DIAGNOSTIC_COUNT])
{
  out[0] = (struct diagnostic_entry){ "batch_drain",
                                      "Batch drain (wait/work)",
                                      "host_block",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->flush_stall };
  out[1] = (struct diagnostic_entry){ "d2h_dispatch",
                                      "D2H dispatch work",
                                      "host_overhead",
                                      DIAGNOSTIC_HOST_OVERHEAD,
                                      &m->drain_dispatch };
  out[2] = (struct diagnostic_entry){ "output_slot_io",
                                      "Output-slot writes",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->io_fence_stall };
  out[3] = (struct diagnostic_entry){ "footer_buffer_io",
                                      "Footer-buffer write",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->footer_buffer_stall };
  out[4] = (struct diagnostic_entry){ "append_extent_io",
                                      "Closed-shard writes",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->append_extent_stall };
  out[5] = (struct diagnostic_entry){ "final_io",
                                      "Final queued writes",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->flush_writes_stall };
  out[6] = (struct diagnostic_entry){ "sink_backpressure",
                                      "Sink queue below limit",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->backpressure };
  out[7] = (struct diagnostic_entry){ "prior_tail_state",
                                      "Prior tail state",
                                      "pipeline_gap",
                                      DIAGNOSTIC_PIPELINE_GAP,
                                      &m->tail_gate };
  out[8] = (struct diagnostic_entry){ "staging_reuse",
                                      "Staging-buffer reuse",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->edge_stall[0] };
  out[9] = (struct diagnostic_entry){ "chunk_metadata_d2h",
                                      "Chunk offsets/sizes D2H",
                                      "host_wait",
                                      DIAGNOSTIC_HOST_BLOCK,
                                      &m->edge_stall[1] };
  out[10] = (struct diagnostic_entry){ "payload_d2h",
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

void
print_append_latency(const struct stream_metrics* m)
{
  if (m->append_count == 0) {
    print_report("  max append ms:   %.2f", (double)m->max_append_ms);
    return;
  }
  print_report("  append ms:       p50 %.3f  p90 %.3f  p99 %.3f  p99.9 %.3f"
               "  max %.3f  (%llu appends)",
               (double)append_ms_at(m, 0.50),
               (double)append_ms_at(m, 0.90),
               (double)append_ms_at(m, 0.99),
               (double)append_ms_at(m, 0.999),
               (double)m->max_append_ms,
               (unsigned long long)m->append_count);
}

void
print_memory_report(const struct bench_memory* mem)
{
  char a[32], b[32];
  fputc('\n', stderr);
  if (!mem->host_reading_failed) {
    format_bytes(a, sizeof(a), mem->host_baseline_bytes);
    format_bytes(b, sizeof(b), mem->host_peak_bytes);
    print_report("  Host memory:   %s at rest, %s peak", a, b);
  } else {
    print_report("  Host memory:   unavailable");
  }
  if (mem->device_used_bytes) {
    format_bytes(a, sizeof(a), mem->device_used_bytes);
    print_report("  Device memory: %s", a);
  }
  if (mem->measured_bytes && mem->estimate_total_bytes) {
    format_bytes(a, sizeof(a), mem->estimate_total_bytes);
    print_report("  Estimate:      %s (%.2fx measured)",
                 a,
                 (double)mem->estimate_total_bytes /
                   (double)mem->measured_bytes);
  }
}

void
print_metric_row(const struct stream_metric* m)
{
  if (m->count <= 0)
    return;
  const int N = m->count;
  double avg_ms = (double)m->ms / N;
  double avg_gbs = gb_per_s(m->input_bytes, (double)m->ms);
  int has_best = m->best_ms < 1e29f;

  if (has_best) {
    // Use the bytes recorded for that exact call so partial-batch tails
    // don't inflate "best" via an average-bytes / min-time fudge.
    double best_gbs = gb_per_s(m->best_input_bytes, (double)m->best_ms);
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
  char wait_calls[24];
  if (m->wait_calls > 0)
    snprintf(wait_calls,
             sizeof(wait_calls),
             "%llu",
             (unsigned long long)m->wait_calls);
  else
    snprintf(wait_calls, sizeof(wait_calls), "-");

  if (m->count > 0) {
    print_report("  %-10s %-25s %8d %10s %9.3f %9.3f %7.2f",
                 metric_owner_name(m->owner),
                 d->label,
                 m->count,
                 wait_calls,
                 (double)m->ms / m->count,
                 (double)m->max_ms,
                 wall_pct);
  } else {
    print_report("  %-10s %-25s %8d %10s %9s %9s %7.2f",
                 metric_owner_name(m->owner),
                 d->label,
                 0,
                 wait_calls,
                 "-",
                 "-",
                 wall_pct);
  }
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
  print_report("  %-10s %-25s %8s %10s %9s %9s %7s",
               "Timeline",
               interval_label,
               "samples",
               "waits",
               "avg ms",
               "max ms",
               "% wall");
  for (size_t i = 0; i < DIAGNOSTIC_COUNT; ++i)
    if (entries[i].section == section && diagnostic_measured(entries[i].metric))
      print_diagnostic_row(&entries[i], wall_s);
}

void
print_diagnostics_report(const struct stream_metrics* m, float wall_s)
{
  struct diagnostic_entry entries[DIAGNOSTIC_COUNT];
  diagnostic_entries(m, entries);
  print_diagnostic_section(entries,
                           DIAGNOSTIC_HOST_BLOCK,
                           "Host blocking",
                           "Reason / awaited condition",
                           wall_s);
  print_diagnostic_section(entries,
                           DIAGNOSTIC_PIPELINE_GAP,
                           "Pipeline gaps",
                           "Awaited condition",
                           wall_s);
  print_diagnostic_section(entries,
                           DIAGNOSTIC_HOST_OVERHEAD,
                           "Host overhead",
                           "Measured work",
                           wall_s);

  if (m->scatter_samples_lost || m->lod_samples_lost) {
    fputc('\n', stderr);
    print_report("  TIMING SAMPLES LOST: scatter=%llu lod=%llu "
                 "(stage totals under-report)",
                 (unsigned long long)m->scatter_samples_lost,
                 (unsigned long long)m->lod_samples_lost);
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
    print_report("  peak pending:    %s", pbuf);
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
  print_report("  total:       %s (%zu elements, %zu epochs)",
               buf,
               total_elements,
               num_epochs);
  format_bytes(
    buf, sizeof(buf), (uint64_t)(layout->chunk_stride * dtype_bpe(dtype)));
  print_report("  chunk:       %lu elements = %s  (stride=%lu)",
               (unsigned long)layout->chunk_elements,
               buf,
               (unsigned long)layout->chunk_stride);
  format_bytes(buf, sizeof(buf), (uint64_t)layout->chunk_pool_bytes);
  print_report("  epoch:       %lu slots, %s pool",
               (unsigned long)layout->chunks_per_epoch,
               buf);
  if (codec.id != CODEC_NONE && max_compressed_size > 0) {
    format_bytes(
      buf, sizeof(buf), (uint64_t)(codec_batch_size * max_compressed_size));
    print_report(
      "  compress:    max_output=%zu comp_pool=%s", max_compressed_size, buf);
  }
}

// Writes, and the room for running several at once.
static void
print_write_report(const struct shard_pool_io_stats* io)
{
  if (!io || io->queue.writes == 0)
    return;

  fputc('\n', stderr);
  print_report("  --- Writes ---");
  print_report("  files waiting:   %.2f avg, %llu peak",
               io->queue.files_waiting_mean,
               (unsigned long long)io->queue.files_waiting_peak);
  print_report("  files opened:    %llu (%llu open at once)",
               (unsigned long long)io->files_opened,
               (unsigned long long)io->files_open_peak);

  char buf[32];
  format_bytes(buf, sizeof(buf), io->queue.bytes_waiting_peak);
  // Two high-water marks, not one moment; the running job is in the bytes.
  print_report("  queued peak:     %s, %llu jobs",
               buf,
               (unsigned long long)io->queue.jobs_waiting_peak);

  format_bytes(buf, sizeof(buf), io->queue.bytes_borrowed);
  char cbuf[32];
  format_bytes(cbuf, sizeof(cbuf), io->queue.bytes_copied);
  print_report("  writes:          %llu (%s borrowed, %s copied)",
               (unsigned long long)io->queue.writes,
               buf,
               cbuf);
  print_report("  wait per write:  %.3f ms avg, %.3f ms max",
               io->queue.wait_ms_mean,
               io->queue.wait_ms_max);
  print_report("  run per write:   %.3f ms avg, %.3f ms max",
               io->queue.run_ms_mean,
               io->queue.run_ms_max);

  for (uint64_t i = 0; i < IO_SIZE_BUCKETS; ++i) {
    if (io->queue.size_buckets[i] == 0)
      continue;
    format_bytes(buf, sizeof(buf), (uint64_t)1 << i);
    print_report(
      "    >= %-10s %llu", buf, (unsigned long long)io->queue.size_buckets[i]);
  }
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
                   uint64_t flush_pending_bytes,
                   const struct shard_pool_io_stats* io)
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
  print_report("  --- Benchmark Results ---");
  char fbuf[32];
  format_bytes(fbuf, sizeof(fbuf), (uint64_t)total_bytes);
  print_report("  Input:        %s (%zu elements)", fbuf, total_elements);
  format_bytes(fbuf, sizeof(fbuf), (uint64_t)ss->total_bytes);
  print_report("  Compressed:   %s (ratio: %.3f)", fbuf, comp_ratio);
  print_report("  Chunks:       %zu (%llu/epoch x %zu epochs)",
               total_chunks,
               (unsigned long long)chunks_per_epoch,
               num_epochs);

  fputc('\n', stderr);
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

  if (metrics->d2h_logical_payload_bytes ||
      metrics->d2h_payload_bytes_transferred ||
      metrics->d2h_metadata_bytes_transferred ||
      metrics->d2h_payload_copy_count) {
    char logical[32], payload[32], metadata[32];
    format_bytes(logical, sizeof(logical), metrics->d2h_logical_payload_bytes);
    format_bytes(
      payload, sizeof(payload), metrics->d2h_payload_bytes_transferred);
    format_bytes(
      metadata, sizeof(metadata), metrics->d2h_metadata_bytes_transferred);
    print_report("  D2H transfer: logical %s, payload %s, metadata %s, "
                 "%llu copies",
                 logical,
                 payload,
                 metadata,
                 (unsigned long long)metrics->d2h_payload_copy_count);
  }

  print_diagnostics_report(metrics, wall_s);

  print_write_report(io);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  fputc('\n', stderr);
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

void
print_bench_json_pass(const struct stream_metrics* m,
                      const struct stream_metric* sink_metric,
                      const struct tile_stream_layout* layout,
                      enum dtype dtype,
                      const struct sink_stats* ss,
                      size_t total_bytes,
                      size_t total_elements,
                      float wall_s,
                      float init_s,
                      float flush_s,
                      const struct bench_memory* mem,
                      int worker_threads,
                      const struct io_write_scheduling* scheduling,
                      const struct shard_pool_io_stats* io)
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
  jw_key(&jw, "memory_measured_bytes");
  jw_uint(&jw, mem->measured_bytes);
  jw_key(&jw, "worker_threads");
  jw_uint(&jw, (uint64_t)worker_threads);

  if (scheduling && scheduling->backend) {
    jw_key(&jw, "io_backend");
    jw_string(&jw, scheduling->backend);
    jw_key(&jw, "io_workers");
    jw_uint(&jw, scheduling->io.workers);
    jw_key(&jw, "io_writes_in_flight");
    jw_uint(&jw, scheduling->io.writes_in_flight);
    jw_key(&jw, "io_writes_in_flight_per_file");
    jw_uint(&jw, scheduling->io.writes_in_flight_per_file);
  }

  if (io && io->queue.writes > 0) {
    jw_key(&jw, "io_files_waiting_mean");
    jw_float(&jw, io->queue.files_waiting_mean);
    jw_key(&jw, "io_files_waiting_peak");
    jw_uint(&jw, io->queue.files_waiting_peak);
    jw_key(&jw, "io_writes_in_flight_mean");
    jw_float(&jw, io->queue.writes_in_flight_mean);
    jw_key(&jw, "io_writes_in_flight_peak");
    jw_uint(&jw, io->queue.writes_in_flight_peak);
    jw_key(&jw, "io_files_opened");
    jw_uint(&jw, io->files_opened);
    jw_key(&jw, "io_files_open_peak");
    jw_uint(&jw, io->files_open_peak);
    jw_key(&jw, "io_writes");
    jw_uint(&jw, io->queue.writes);
    jw_key(&jw, "io_bytes_copied");
    jw_uint(&jw, io->queue.bytes_copied);
    jw_key(&jw, "io_bytes_borrowed");
    jw_uint(&jw, io->queue.bytes_borrowed);
    jw_key(&jw, "io_queued_bytes_peak");
    jw_uint(&jw, io->queue.bytes_waiting_peak);
    jw_key(&jw, "io_queued_jobs_peak");
    jw_uint(&jw, io->queue.jobs_waiting_peak);
    jw_key(&jw, "io_wait_ms_mean");
    jw_float(&jw, io->queue.wait_ms_mean);
    jw_key(&jw, "io_wait_ms_max");
    jw_float(&jw, io->queue.wait_ms_max);
    jw_key(&jw, "io_run_ms_mean");
    jw_float(&jw, io->queue.run_ms_mean);
    jw_key(&jw, "io_run_ms_max");
    jw_float(&jw, io->queue.run_ms_max);
    jw_key(&jw, "io_write_sizes");
    jw_array_begin(&jw);
    for (uint64_t i = 0; i < IO_SIZE_BUCKETS; ++i) {
      if (io->queue.size_buckets[i] == 0)
        continue;
      jw_object_begin(&jw);
      jw_key(&jw, "at_least");
      jw_uint(&jw, (uint64_t)1 << i);
      jw_key(&jw, "n");
      jw_uint(&jw, io->queue.size_buckets[i]);
      jw_object_end(&jw);
    }
    jw_array_end(&jw);
  }

  if (m->d2h_logical_payload_bytes || m->d2h_payload_bytes_transferred ||
      m->d2h_metadata_bytes_transferred || m->d2h_payload_copy_count) {
    jw_key(&jw, "d2h_transfer");
    jw_object_begin(&jw);
    jw_key(&jw, "logical_payload_bytes");
    jw_uint(&jw, m->d2h_logical_payload_bytes);
    jw_key(&jw, "payload_bytes_transferred");
    jw_uint(&jw, m->d2h_payload_bytes_transferred);
    jw_key(&jw, "metadata_bytes_transferred");
    jw_uint(&jw, m->d2h_metadata_bytes_transferred);
    jw_key(&jw, "payload_copy_count");
    jw_uint(&jw, m->d2h_payload_copy_count);
    jw_object_end(&jw);
  }

  jw_key(&jw, "stages");
  jw_object_begin(&jw);
  json_stage_metric(&jw, "memcpy", &m->memcpy);
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
  jw_float(&jw, (double)m->drain_dispatch.ms);
  jw_key(&jw, "drain_dispatch_count");
  jw_uint(&jw, (uint64_t)m->drain_dispatch.count);
  jw_key(&jw, "io_fence_ms");
  jw_float(&jw, (double)m->io_fence_stall.ms);
  jw_key(&jw, "io_fence_count");
  jw_uint(&jw, (uint64_t)m->io_fence_stall.count);
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
  jw_key(&jw, "tail_gate_ms");
  jw_float(&jw, (double)m->tail_gate.ms);
  jw_key(&jw, "tail_gate_count");
  jw_uint(&jw, (uint64_t)m->tail_gate.count);
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
  jw_string(&jw, metric_owner_name(m->drain_dispatch.owner));
  jw_key(&jw, "io_fence");
  jw_string(&jw, metric_owner_name(m->io_fence_stall.owner));
  jw_key(&jw, "footer_buffer");
  jw_string(&jw, metric_owner_name(m->footer_buffer_stall.owner));
  jw_key(&jw, "append_extent");
  jw_string(&jw, metric_owner_name(m->append_extent_stall.owner));
  jw_key(&jw, "flush_writes");
  jw_string(&jw, metric_owner_name(m->flush_writes_stall.owner));
  jw_key(&jw, "backpressure");
  jw_string(&jw, metric_owner_name(m->backpressure.owner));
  jw_key(&jw, "tail_gate");
  jw_string(&jw, metric_owner_name(m->tail_gate.owner));
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
