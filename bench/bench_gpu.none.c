#include "bench_gpu.h"

#include "log/log.h"

static void
no_gpu(void)
{
  log_error("  built without GPU support: reconfigure with "
            "-DCHUCKY_ENABLE_GPU=ON or run with --backend cpu");
}

int
bench_gpu_enabled(void)
{
  return 0;
}

int
bench_gpu_context_create(void)
{
  no_gpu();
  return 1;
}

void
bench_gpu_context_destroy(void)
{
}

size_t
bench_gpu_free_memory(void)
{
  return 0;
}

int
bench_gpu_advise_layout(struct tile_stream_configuration* config,
                        size_t target_chunk_bytes,
                        size_t min_chunk_bytes,
                        const int* ratios,
                        size_t budget_bytes,
                        size_t min_shard_bytes,
                        uint32_t target_concurrent_shards,
                        uint32_t min_append_shards,
                        size_t shard_alignment,
                        struct advise_layout_diagnostic* diag)
{
  (void)config;
  (void)target_chunk_bytes;
  (void)min_chunk_bytes;
  (void)ratios;
  (void)budget_bytes;
  (void)min_shard_bytes;
  (void)target_concurrent_shards;
  (void)min_append_shards;
  (void)shard_alignment;
  (void)diag;
  no_gpu();
  return 1;
}

void
bench_gpu_report_memory(const struct tile_stream_configuration* config,
                        uint64_t* total_chunks,
                        size_t* device_bytes,
                        size_t* pinned_bytes)
{
  (void)config;
  (void)total_chunks;
  (void)device_bytes;
  (void)pinned_bytes;
}

void
bench_gpu_report_memory_pair(const struct tile_stream_configuration* config)
{
  (void)config;
}

struct tile_stream_gpu*
bench_gpu_create(const struct tile_stream_configuration* config,
                 struct shard_sink* sink)
{
  (void)config;
  (void)sink;
  no_gpu();
  return NULL;
}

void
bench_gpu_destroy(struct tile_stream_gpu* s)
{
  (void)s;
}

const struct tile_stream_layout*
bench_gpu_layout(const struct tile_stream_gpu* s)
{
  (void)s;
  return NULL;
}

struct stream_metrics
bench_gpu_get_metrics(const struct tile_stream_gpu* s)
{
  (void)s;
  struct stream_metrics m = { 0 };
  return m;
}

struct tile_stream_status
bench_gpu_status(const struct tile_stream_gpu* s)
{
  (void)s;
  struct tile_stream_status st = { 0 };
  return st;
}

int
bench_gpu_worker_threads(const struct tile_stream_gpu* s)
{
  (void)s;
  return 0;
}

struct writer*
bench_gpu_writer(struct tile_stream_gpu* s)
{
  (void)s;
  return NULL;
}

uint64_t
bench_gpu_cursor(const struct tile_stream_gpu* s)
{
  (void)s;
  return 0;
}
