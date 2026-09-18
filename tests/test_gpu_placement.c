#define _GNU_SOURCE

#include "gpu/placement.h"
#include "gpu/stream.internal.h"
#include "multiarray.gpu.h"
#include "platform/platform.h"
#include "stream/host_output_pool.h"
#include "test_placement.h"
#include "test_runner.h"
#include "test_shard_sink.h"
#include "threadpool/threadpool.h"

#include <stdlib.h>

struct observed_sink
{
  struct test_shard_sink sink;
  struct shard_writer* (*open)(struct shard_sink*, uint8_t, uint64_t);
  struct test_placement_state delivery;
  int observed;
};

static struct shard_writer*
observe_open(struct shard_sink* base, uint8_t level, uint64_t index)
{
  struct observed_sink* sink = (struct observed_sink*)base;
  sink->delivery = test_placement_state();
  sink->observed = 1;
  return sink->open(base, level, index);
}

static int
reject_extent(struct shard_sink* sink,
              uint8_t level,
              uint8_t n,
              const uint64_t* sizes)
{
  (void)sink;
  (void)level;
  (void)n;
  (void)sizes;
  return 1;
}

static struct tile_stream_configuration
make_config(struct dimension* dim)
{
  *dim = (struct dimension){ .size = 65536,
                             .chunk_size = 16384,
                             .chunks_per_shard = 4 };
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 1 << 20,
    .dtype = dtype_u8,
    .rank = 1,
    .dimensions = dim,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 4,
    .max_threads = 4,
  };
}

static void
capture_copy_worker(int tid, int threads, void* arg)
{
  (void)threads;
  ((struct test_placement_state*)arg)[tid] = test_placement_state();
}

static int
check_pinned_memory(void* data, size_t bytes, int expected_node)
{
  unsigned memory_type = 0;
  CU(Fail,
     cuPointerGetAttribute(&memory_type,
                           CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                           (CUdeviceptr)(uintptr_t)data));
  CHECK(Fail, memory_type == CU_MEMORYTYPE_HOST);
#if defined(__linux__) && defined(SYS_get_mempolicy)
  unsigned sampled = 0, local = 0;
  const size_t step = platform_page_alignment();
  for (size_t offset = 0; offset < bytes && sampled < 64; offset += step) {
    int node = -1;
    if (syscall(SYS_get_mempolicy,
                &node,
                NULL,
                0,
                (char*)data + offset,
                MPOL_F_NODE | MPOL_F_ADDR) != 0)
      break;
    ++sampled;
    local += node == expected_node;
  }
  log_info("pinned host buffer: %u/%u sampled pages on GPU node %d",
           local,
           sampled,
           expected_node);
#else
  (void)bytes;
  (void)expected_node;
#endif
  return 0;
Fail:
  return 1;
}

struct append_args
{
  struct writer* writer;
  int failed;
};

static void
append_from_worker(void* arg)
{
  struct append_args* args = arg;
#if defined(__linux__)
  cpu_set_t original;
  const int has_affinity =
    sched_getaffinity(0, sizeof(original), &original) == 0;
  if (has_affinity) {
    cpu_set_t one;
    CPU_ZERO(&one);
    for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu)
      if (CPU_ISSET(cpu, &original)) {
        CPU_SET(cpu, &one);
        break;
      }
    if (sched_setaffinity(0, sizeof(one), &one) != 0)
      args->failed = 1;
  }
#endif
  const struct test_placement_state before = test_placement_state();
  uint8_t data[65536] = { 0 };
  const struct writer_result result =
    writer_append(args->writer, (struct slice){ data, data + sizeof(data) });
  args->failed |= result.error || result.rest.beg != result.rest.end;
  args->failed |= !test_placement_unchanged(&before);
  args->failed |= writer_flush(args->writer).error != 0;
  args->failed |= !test_placement_unchanged(&before);
#if defined(__linux__)
  if (has_affinity)
    args->failed |= sched_setaffinity(0, sizeof(original), &original) != 0;
#endif
}

static int
test_stream_placement(void)
{
  int failed = 1;
  struct dimension dim;
  const struct tile_stream_configuration config = make_config(&dim);
  struct observed_sink sink = { 0 };
  test_sink_init(&sink.sink, 1, 128 << 10);
  sink.open = sink.sink.base.open;
  sink.sink.base.open = observe_open;
  struct tile_stream_gpu* stream = NULL;
  struct platform_thread* worker = NULL;
  struct host_output outputs[HOST_OUTPUT_COUNT] = { 0 };
  const struct test_placement_state before = test_placement_state();
  CUdevice device;
  char pci[32];
  CU(Done, cuCtxGetDevice(&device));
  CU(Done, cuDeviceGetPCIBusId(pci, sizeof(pci), device));
  const int node = platform_pci_numa_node(pci);
  log_info("active CUDA device %d, PCI %s, NUMA node %d", device, pci, node);

  stream = tile_stream_gpu_create(&config, &sink.sink.base);
  CHECK(Done, stream);
  CHECK(Done, test_placement_unchanged(&before));
  for (int i = 0; i < 2; ++i)
    CHECK(Done,
          check_pinned_memory(stream->engine.stage.slot[i].h_in,
                              config.buffer_capacity_bytes,
                              node) == 0);
  for (unsigned i = 0; i < HOST_OUTPUT_COUNT; ++i) {
    CHECK(Done,
          host_output_pool_acquire(stream->ar.agg.output_pool, &outputs[i]) ==
            0);
    CHECK(Done,
          check_pinned_memory(outputs[i].data, outputs[i].capacity, node) == 0);
  }
  for (unsigned i = 0; i < HOST_OUTPUT_COUNT; ++i) {
    host_output_group_seal(outputs[i].group);
    outputs[i].group = NULL;
  }

  struct test_placement_state copies[4];
  threadpool_for_threads(stream->engine.copy_pool, capture_copy_worker, copies);
#if defined(__linux__)
  struct platform_placement_scope scope;
  platform_placement_enter(stream->engine.placement, 0, &scope);
  const struct test_placement_state expected = test_placement_state();
  CHECK(Done, platform_placement_leave(&scope) == 0);
  for (int i = 1; i < threadpool_size(stream->engine.copy_pool); ++i) {
    CHECK(Done, copies[i].has_affinity == expected.has_affinity);
    CHECK(Done,
          !expected.has_affinity || CPU_EQUAL(&copies[i].cpus, &expected.cpus));
  }
#endif
  struct append_args args = { .writer = tile_stream_gpu_writer(stream) };
  worker = platform_thread_start(append_from_worker, &args);
  CHECK(Done, worker);
  const int joined = platform_thread_join(worker);
  worker = NULL;
  CHECK(Done, joined == 0 && args.failed == 0);
  CHECK(Done, sink.observed);
#if defined(__linux__)
  if (stream->engine.delivery.thread && expected.has_affinity)
    CHECK(Done,
          sink.delivery.has_affinity &&
            CPU_EQUAL(&sink.delivery.cpus, &expected.cpus));
#endif
  CHECK(Done, test_placement_unchanged(&before));
  tile_stream_gpu_destroy(stream);
  stream = NULL;
  CHECK(Done, test_placement_unchanged(&before));
  failed = 0;
Done:
  if (worker)
    platform_thread_join(worker);
  for (unsigned i = 0; i < HOST_OUTPUT_COUNT; ++i)
    if (outputs[i].group)
      host_output_group_seal(outputs[i].group);
  tile_stream_gpu_destroy(stream);
  test_sink_free(&sink.sink);
  return failed;
}

static int
test_constructor_restoration(void)
{
  int failed = 1;
  struct dimension dim;
  const struct tile_stream_configuration config = make_config(&dim);
  struct test_shard_sink sink;
  test_sink_init(&sink, 1, 128 << 10);
  struct shard_sink* bases[] = { &sink.base };
  struct multiarray_tile_stream_gpu* multi = NULL;
  struct tile_stream_gpu* single = NULL;
  const struct test_placement_state before = test_placement_state();

  multi = multiarray_tile_stream_gpu_create(1, &config, bases, 0);
  CHECK(Done, multi);
  CHECK(Done, test_placement_unchanged(&before));
  multiarray_tile_stream_gpu_destroy(multi);
  multi = NULL;
  CHECK(Done, test_placement_unchanged(&before));

  sink.base.update_append = reject_extent;
  single = tile_stream_gpu_create(&config, &sink.base);
  CHECK(Done, single == NULL);
  CHECK(Done, test_placement_unchanged(&before));
  multi = multiarray_tile_stream_gpu_create(1, &config, bases, 0);
  CHECK(Done, multi == NULL);
  CHECK(Done, test_placement_unchanged(&before));
  failed = 0;
Done:
  tile_stream_gpu_destroy(single);
  multiarray_tile_stream_gpu_destroy(multi);
  test_sink_free(&sink);
  return failed;
}

RUN_GPU_TESTS({ "stream placement and cross-thread append",
                test_stream_placement },
              { "constructor restoration", test_constructor_restoration })
