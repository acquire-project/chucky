#include "bench_gpu.h"

#include "bench_report.h"
#include "gpu/prelude.cuda.h"
#include "stream.gpu.h"
#include "util/format_bytes.h"

static CUcontext context;

int
bench_gpu_enabled(void)
{
  return 1;
}

int
bench_gpu_context_create(void)
{
  CUdevice dev;
  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cu_ctx_create(&context, 0, dev));
  return 0;
Fail:
  return 1;
}

void
bench_gpu_context_destroy(void)
{
  if (context)
    cuCtxDestroy(context);
  context = 0;
}

size_t
bench_gpu_free_memory(void)
{
  size_t free_mem = 0, total_mem = 0;
  if (cuMemGetInfo(&free_mem, &total_mem) != CUDA_SUCCESS)
    return 0;
  return free_mem;
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
  return tile_stream_gpu_advise_layout(config,
                                       target_chunk_bytes,
                                       min_chunk_bytes,
                                       ratios,
                                       budget_bytes,
                                       min_shard_bytes,
                                       target_concurrent_shards,
                                       min_append_shards,
                                       shard_alignment,
                                       diag);
}

void
bench_gpu_report_memory(const struct tile_stream_configuration* config,
                        uint64_t* total_chunks,
                        size_t* device_bytes,
                        size_t* pinned_bytes)
{
  struct tile_stream_memory_info mem;
  if (tile_stream_gpu_memory_estimate(config, 0, &mem) != 0)
    return;

  *total_chunks = mem.total_chunks;
  *device_bytes = mem.device_bytes;
  *pinned_bytes = mem.host_pinned_bytes;

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

void
bench_gpu_report_memory_pair(const struct tile_stream_configuration* config)
{
  struct tile_stream_memory_info mem;
  if (tile_stream_gpu_memory_estimate(config, 0, &mem) != 0)
    return;

  char a[32], b[32];
  format_bytes(a, sizeof(a), mem.device_bytes);
  format_bytes(b, sizeof(b), mem.host_pinned_bytes);
  print_report("  GPU memory (per stream): %s device, %s pinned", a, b);
  format_bytes(a, sizeof(a), 2 * mem.device_bytes);
  format_bytes(b, sizeof(b), 2 * mem.host_pinned_bytes);
  print_report("  GPU memory (total x2):   %s device, %s pinned", a, b);
}

struct tile_stream_gpu*
bench_gpu_create(const struct tile_stream_configuration* config,
                 struct shard_sink* sink)
{
  return tile_stream_gpu_create(config, sink);
}

void
bench_gpu_destroy(struct tile_stream_gpu* s)
{
  tile_stream_gpu_destroy(s);
}

const struct tile_stream_layout*
bench_gpu_layout(const struct tile_stream_gpu* s)
{
  return tile_stream_gpu_layout(s);
}

struct stream_metrics
bench_gpu_get_metrics(const struct tile_stream_gpu* s)
{
  return tile_stream_gpu_get_metrics(s);
}

struct tile_stream_status
bench_gpu_status(const struct tile_stream_gpu* s)
{
  return tile_stream_gpu_status(s);
}

struct writer*
bench_gpu_writer(struct tile_stream_gpu* s)
{
  return tile_stream_gpu_writer(s);
}

uint64_t
bench_gpu_cursor(const struct tile_stream_gpu* s)
{
  return tile_stream_gpu_cursor(s);
}
