// Regression test: multiarray GPU flush must never deliver a fence issued
// by one sink to a different sink's wait. In production that deadlocks
// because fences only retire on the sink that issued them; here mock sinks
// detect the cross-sink mis-routing deterministically and flag it without
// blocking, so the assertion is a flag check rather than a timeout.
//
// The fix lives in unbind_context via drain_d2h_for_array, which waits on
// any cached fence with the departing sink and clears it before another
// array binds in. Without that drain, the next array picks up a fence the
// prior sink stamped on the shared delivery pipeline.

#include "dimension.h"
#include "multiarray.gpu.h"
#include "stream.gpu.h"
#include "util/prelude.h"
#include "writer.h"

#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MOCK_MAX_LEVELS 4
#define MOCK_MAX_ISSUED 256
#define DISCARD_BUF_BYTES (1u << 16)

struct mock_shard_sink
{
  struct shard_sink base;
  int id;
  uint64_t next_seq;
  // Per-level record of fences this sink issued; consulted on wait to
  // detect a fence that came from elsewhere.
  uint64_t issued[MOCK_MAX_LEVELS][MOCK_MAX_ISSUED];
  int n_issued[MOCK_MAX_LEVELS];

  int cross_sink_violation;
  uint64_t bad_seq;

  struct shard_writer dwriter;
  uint8_t dbuf[DISCARD_BUF_BYTES];
};

static int
discard_write(struct shard_writer* self,
              uint64_t offset,
              const void* beg,
              const void* end)
{
  (void)self;
  (void)offset;
  (void)beg;
  (void)end;
  return 0;
}

static int
discard_finalize(struct shard_writer* self)
{
  (void)self;
  return 0;
}

static struct shard_writer*
mock_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct mock_shard_sink* m = (struct mock_shard_sink*)self;
  return &m->dwriter;
}

static struct io_event
mock_record_fence(struct shard_sink* self, uint8_t level)
{
  struct mock_shard_sink* m = (struct mock_shard_sink*)self;
  if (level >= MOCK_MAX_LEVELS)
    return (struct io_event){ .seq = 0 };
  uint64_t seq = ++m->next_seq;
  if (m->n_issued[level] < MOCK_MAX_ISSUED)
    m->issued[level][m->n_issued[level]++] = seq;
  return (struct io_event){ .seq = seq };
}

static void
mock_wait_fence(struct shard_sink* self, uint8_t level, struct io_event ev)
{
  struct mock_shard_sink* m = (struct mock_shard_sink*)self;
  if (ev.seq == 0)
    return;
  if (level >= MOCK_MAX_LEVELS) {
    if (!m->cross_sink_violation) {
      m->cross_sink_violation = 1;
      m->bad_seq = ev.seq;
    }
    return;
  }
  for (int i = 0; i < m->n_issued[level]; ++i) {
    if (m->issued[level][i] == ev.seq)
      return;
  }
  if (!m->cross_sink_violation) {
    m->cross_sink_violation = 1;
    m->bad_seq = ev.seq;
  }
}

static int
mock_has_error(const struct shard_sink* self)
{
  (void)self;
  return 0;
}

static void
mock_sink_init(struct mock_shard_sink* m, int id)
{
  memset(m, 0, sizeof(*m));
  m->id = id;
  m->base.open = mock_open;
  m->base.record_fence = mock_record_fence;
  m->base.wait_fence = mock_wait_fence;
  m->base.has_error = mock_has_error;
  m->dwriter.write = discard_write;
  m->dwriter.finalize = discard_finalize;
}

// ---- Test body ----

// Single-LOD config sized so each update triggers an inline flush — that's
// the path that exercises the cached fence the fix targets.
static struct tile_stream_configuration
make_simple_config(struct dimension dims[2])
{
  dims_create(dims, "xy", (uint64_t[]){ 4, 4 });
  dims_set_chunk_sizes(dims, 2, (uint64_t[]){ 2, 2 });
  dims_set_shard_counts(dims, 2, (uint64_t[]){ 2, 1 });
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 1,
  };
}

static int
write_epoch(struct multiarray_writer* w, int idx, size_t n_elements)
{
  size_t bytes = n_elements * sizeof(uint16_t);
  uint8_t* data = (uint8_t*)malloc(bytes);
  if (!data)
    return 1;
  memset(data, (uint8_t)(0xA0 + idx), bytes);
  struct slice sl = { .beg = data, .end = data + bytes };
  struct multiarray_writer_result r = w->update(w, idx, sl);
  free(data);
  return r.error == multiarray_writer_ok ? 0 : 1;
}

static int
test_no_cross_sink_fence_routing(void)
{
  log_info("=== test_no_cross_sink_fence_routing ===");

  struct mock_shard_sink sinkA, sinkB;
  mock_sink_init(&sinkA, 0);
  mock_sink_init(&sinkB, 1);

  struct dimension dimsA[2], dimsB[2];
  struct tile_stream_configuration cfgA = make_simple_config(dimsA);
  struct tile_stream_configuration cfgB = make_simple_config(dimsB);
  struct tile_stream_configuration configs[] = { cfgA, cfgB };
  struct shard_sink* sinks[] = { &sinkA.base, &sinkB.base };

  struct multiarray_tile_stream_gpu* ms =
    multiarray_tile_stream_gpu_create(2, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_gpu_writer(ms);

  // Alternate A and B several times so the shared delivery pipeline carries
  // a fence from the prior sink across each switch. Two rounds cover both
  // double-buffered slots.
  CHECK(Fail, write_epoch(w, 0, 8) == 0);
  CHECK(Fail, write_epoch(w, 1, 8) == 0);
  CHECK(Fail, write_epoch(w, 0, 8) == 0);
  CHECK(Fail, write_epoch(w, 1, 8) == 0);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  // Neither sink should ever receive a fence the other issued. With the fix
  // this is invariant; without it the violation flag fires deterministically.
  if (sinkA.cross_sink_violation)
    log_error("sinkA received foreign seq=%llu",
              (unsigned long long)sinkA.bad_seq);
  if (sinkB.cross_sink_violation)
    log_error("sinkB received foreign seq=%llu",
              (unsigned long long)sinkB.bad_seq);
  CHECK(Fail, sinkA.cross_sink_violation == 0);
  CHECK(Fail, sinkB.cross_sink_violation == 0);

  multiarray_tile_stream_gpu_destroy(ms);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_gpu_destroy(ms);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  CUresult rc = cuInit(0);
  if (rc != CUDA_SUCCESS) {
    log_error("cuInit failed (%d)", rc);
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    log_error("Failed to create CUDA context");
    return 1;
  }

  int ret = test_no_cross_sink_fence_routing();

  cuCtxDestroy(ctx);
  return ret;
}
