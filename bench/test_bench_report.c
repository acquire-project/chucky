#include "bench_report.h"

#include <limits.h>
#include <string.h>

int
main(int argc, char** argv)
{
  if (argc != 2)
    return 1;
  const int full = strcmp(argv[1], "full") == 0;
  const int large = strcmp(argv[1], "large") == 0;
  const int empty = strcmp(argv[1], "empty") == 0;
  struct stream_metrics m = { 0 };
  if (!empty) {
    struct stream_metric stage = {
      .name = "backend_internal_name",
      .count = 2,
      .ms = 0.00008f,
      .best_ms = 0.00003f,
      .max_ms = 0.00005f,
      .best_input_bytes = 512,
      .best_output_bytes = 512,
      .input_bytes = 1024,
      .output_bytes = 1024,
    };
    m.memcpy = m.h2d = m.scatter = m.lod_gather = m.lod_reduce =
      m.lod_append_fold = m.lod_morton_chunk = m.compress = m.aggregate =
        m.d2h = m.sink = stage;
    m.scatter.name = strcmp(argv[1], "copy") == 0 ? "Copy" : "scatter";
    m.memcpy_calls = large ? UINT64_MAX : 128;
    m.memcpy_bytes = large ? UINT64_MAX : 65536;
    if (full) {
      m.memcpy.count = 128;
      m.memcpy.ms *= 64;
      m.memcpy.input_bytes = m.memcpy.output_bytes = 65536;
    }
    m.compress.best_ms = 1e30f;
    m.sink.ms = 2000;
    m.sink.best_ms = 1000;
    m.sink.input_bytes = 2147483648.0;
    m.sink.best_input_bytes = 1073741824.0;

    struct stream_metric wait = {
      .owner = METRIC_OWNER_PRODUCER,
      .count = large ? INT_MAX : 2,
      .wait_calls = large ? UINT64_MAX : 3,
      .ms = large ? 1e20f : 0.00008f,
      .max_ms = large ? 1e20f : 0.00005f,
    };
    m.flush_stall = m.footer_buffer_stall = m.append_extent_stall =
      m.flush_writes_stall = m.backpressure = wait;
    m.flush_stall.wait_calls = 0;
    m.edge_stall[0].owner = METRIC_OWNER_PRODUCER;
    m.edge_stall[0].wait_calls = 1;
    wait.owner = METRIC_OWNER_DELIVERY;
    m.edge_stall[1] = m.edge_stall[2] = m.indexed_aggregate_wait =
      m.chunk_metadata_wait = m.delivery_dispatch = wait;
    wait.owner = METRIC_OWNER_D2H;
    m.chunk_metadata_copy = wait;

    struct duration_stats duration = {
      .count = large ? UINT64_MAX : 2,
      .total_ms = large ? 1e20f : 0.00008f,
      .min_ms = 0.00003f,
      .max_ms = large ? 1e20f : 0.00005f,
    };
    m.delivery.submitted_to_start = m.delivery.start_to_payload_ready =
      m.delivery.payload_ready_to_writes_posted =
        m.delivery.submitted_to_slot_reuse = duration;
    m.append_count = large ? UINT64_MAX : 128;
    m.append_ms_buckets[0] = m.append_count;
    m.max_append_ms = 0.00004f;
    m.d2h_payload_bytes_transferred = 65536;
    m.d2h_metadata_bytes_transferred = 1024;
    m.d2h_payload_copy_count = large ? UINT64_MAX : 128;
    m.shard_padding_logical_payload_bytes = 65536;
    m.shard_padding_internal_bytes = 1024;
    m.shard_padding_physical_update_count = large ? UINT64_MAX : 128;
    m.shard_padding_padded_update_count = large ? UINT64_MAX : 8;
    m.scatter_samples_lost = large ? UINT64_MAX : 1;
    m.lod_samples_lost = large ? UINT64_MAX : 2;
    m.peak_pending_bytes = 65536;
  }
  const struct tile_stream_layout layout = {
    .epoch_elements = 4096,
    .chunk_elements = 4096,
    .chunk_stride = 4096,
    .chunk_pool_bytes = 4096,
    .chunks_per_epoch = 1,
  };
  const struct bench_memory mem = {
    .host_reading_failed = empty,
    .host_baseline_bytes = 1024,
    .host_peak_bytes = 65536,
    .device_used_bytes = empty ? 0 : 4096,
    .device_overhead_valid = !empty,
    .device_overhead_bytes = -1024,
    .measured_bytes = empty ? 0 : 4096,
    .estimate_total_bytes = empty ? 0 : 5120,
  };
  const struct sink_stats sink = { .total_bytes = 32768 };
  log_bench_header(&layout,
                   dtype_u8,
                   (struct codec_config){ .id = CODEC_NONE },
                   0,
                   0,
                   65536,
                   65536);
  print_bench_report(&m,
                     &layout,
                     dtype_u8,
                     &sink,
                     65536,
                     65536,
                     1,
                     0.1f,
                     empty ? 0 : 0.2f,
                     empty ? 0 : 65536);
  print_memory_report(&mem);
  struct bench_sustained window = {
    .boundary_timing = 1,
    .elapsed_s = 1,
    .source_bytes = 64u << 20,
    .append_bytes = 512,
  };
  if (large) {
    window.boundary[0] = (struct bench_append_sample){
      .calls = UINT64_MAX,
      .over_100ms = UINT64_MAX,
      .total_ms = 1e20,
      .max_ms = 1e20,
    };
  }
  if (large || empty)
    print_sustained_report(&window);
  print_bench_json_pass(&m,
                        &m.sink,
                        &layout,
                        dtype_u8,
                        (struct codec_config){ .id = CODEC_NONE },
                        &sink,
                        65536,
                        65536,
                        1,
                        0.1f,
                        empty ? 0 : 0.2f,
                        &mem,
                        1,
                        large || empty ? &window : NULL);
  return 0;
}
