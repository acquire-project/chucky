#include "bench_report.h"

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
    print_report("  peak pending:    %.2f MiB",
                 (double)metrics->peak_pending_bytes / (1024.0 * 1024.0));
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
