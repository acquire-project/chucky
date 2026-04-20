#include "bench_report.h"

#include "util/format_bytes.h"
#include "zarr/json_writer.h"

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
  char fbuf[32];
  format_bytes(fbuf, sizeof(fbuf), (uint64_t)total_bytes);
  print_report("  Input:        %s (%zu elements)", fbuf, total_elements);
  format_bytes(fbuf, sizeof(fbuf), (uint64_t)ss->total_bytes);
  print_report("  Compressed:   %s (ratio: %.3f)", fbuf, comp_ratio);
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

  // Stall stats — wall-clock time the host is blocked waiting. Emitted only
  // if any stall was observed.
  int have_stalls =
    metrics->flush_stall.count > 0 || metrics->kick_sync_stall.count > 0 ||
    metrics->io_fence_stall.count > 0 || metrics->backpressure.count > 0 ||
    metrics->max_append_ms > 0 || metrics->peak_pending_bytes > 0;
  if (have_stalls) {
    print_report("");
    print_report("  --- Stall stats ---");
    print_metric_row(&metrics->flush_stall);
    print_metric_row(&metrics->kick_sync_stall);
    print_metric_row(&metrics->io_fence_stall);
    print_metric_row(&metrics->backpressure);
    print_report("  max append ms:   %.2f", (double)metrics->max_append_ms);
    char pbuf[32];
    format_bytes(pbuf, sizeof(pbuf), (uint64_t)metrics->peak_pending_bytes);
    print_report("  peak pending:    %s", pbuf);
  }

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
  jw_key(jw, "avg_ms");
  jw_float(jw, avg_ms);
  if (sm->best_ms < 1e29f) {
    jw_key(jw, "best_ms");
    jw_float(jw, (double)sm->best_ms);
  }
  jw_key(jw, "in_gibs");
  jw_float(jw, in_gibs);
  jw_key(jw, "out_gibs");
  jw_float(jw, out_gibs);
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
                      float flush_s)
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
  jw_key(&jw, "kick_sync_ms");
  jw_float(&jw, (double)m->kick_sync_stall.ms);
  jw_key(&jw, "kick_sync_count");
  jw_uint(&jw, (uint64_t)m->kick_sync_stall.count);
  jw_key(&jw, "io_fence_ms");
  jw_float(&jw, (double)m->io_fence_stall.ms);
  jw_key(&jw, "io_fence_count");
  jw_uint(&jw, (uint64_t)m->io_fence_stall.count);
  jw_key(&jw, "backpressure_ms");
  jw_float(&jw, (double)m->backpressure.ms);
  jw_key(&jw, "backpressure_count");
  jw_uint(&jw, (uint64_t)m->backpressure.count);
  jw_key(&jw, "max_append_ms");
  jw_float(&jw, (double)m->max_append_ms);
  jw_key(&jw, "peak_pending_mib");
  jw_float(&jw, (double)m->peak_pending_bytes / (1024.0 * 1024.0));
  jw_object_end(&jw);

  jw_object_end(&jw);
  printf("%.*s\n", (int)jw_length(&jw), json_buf);
}

void
print_bench_json_error(void)
{
  char buf[64];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));
  jw_object_begin(&jw);
  jw_key(&jw, "status");
  jw_string(&jw, "error");
  jw_object_end(&jw);
  printf("%.*s\n", (int)jw_length(&jw), buf);
}
