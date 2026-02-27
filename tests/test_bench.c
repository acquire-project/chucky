#define _USE_MATH_DEFINES
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// Deterministic source data — mix of compressibility levels.
//   First 1/3:  Gaussian noise in 12-bit range (partially compressible)
//   Middle 1/3: constant 42 (maximally compressible)
//   Last 1/3:   uniform random uint16 (incompressible)

static uint64_t
splitmix64(uint64_t* state)
{
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static double
splitmix64_uniform(uint64_t* state)
{
  return (double)(splitmix64(state) >> 11) * 0x1.0p-53;
}

static void
fill_thirds(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  const size_t third1 = total / 3;
  const size_t third2 = 2 * total / 3;
  uint64_t rng = offset * 0x9e3779b97f4a7c15ULL + 1;

  for (size_t i = 0; i < count; ++i) {
    size_t gi = offset + i;
    if (gi < third1) {
      // Gaussian via Box-Muller, mu=2048 sigma=512, clamped to [0,4095]
      double u1 = splitmix64_uniform(&rng);
      double u2 = splitmix64_uniform(&rng);
      if (u1 < 1e-15)
        u1 = 1e-15;
      double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
      int val = (int)(2048.0 + 512.0 * z);
      if (val < 0)
        val = 0;
      if (val > 4095)
        val = 4095;
      buf[i] = (uint16_t)val;
    } else if (gi < third2) {
      buf[i] = 42;
    } else {
      buf[i] = (uint16_t)splitmix64(&rng);
    }
  }
}

static size_t
dim_total_elements(const struct dimension* dims, uint8_t rank)
{
  size_t n = 1;
  for (uint8_t i = 0; i < rank; ++i)
    n *= dims[i].size;
  return n;
}

// --- Throughput helpers ---

static double
gb_per_s(double bytes, double ms)
{
  if (ms <= 0)
    return 0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// --- Discard shard_sink for benchmarks ---

struct discard_shard_writer
{
  struct shard_writer base;
  struct discard_shard_sink* parent;
};

struct discard_shard_sink
{
  struct shard_sink base;
  struct discard_shard_writer writer;
  size_t total_bytes;
  size_t shards_finalized;
  struct stream_metric sink;
  struct platform_clock clock;
};

static int
discard_shard_write(struct shard_writer* self,
                    uint64_t offset,
                    const void* beg,
                    const void* end)
{
  (void)offset;
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  platform_toc(&w->parent->clock);
  w->parent->total_bytes += (size_t)((const char*)end - (const char*)beg);
  accumulate_metric_ms(&w->parent->sink,
                       (float)(platform_toc(&w->parent->clock) * 1000.0));
  return 0;
}

static int
discard_shard_finalize(struct shard_writer* self)
{
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  w->parent->shards_finalized++;
  return 0;
}

static struct shard_writer*
discard_shard_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct discard_shard_sink* s = (struct discard_shard_sink*)self;
  return &s->writer.base;
}

static void
discard_shard_sink_init(struct discard_shard_sink* s)
{
  *s = (struct discard_shard_sink){
    .base = { .open = discard_shard_open },
    .sink = { .name = "Sink", .best_ms = 1e30f },
  };
  s->writer = (struct discard_shard_writer){
    .base = { .write = discard_shard_write,
              .finalize = discard_shard_finalize },
    .parent = s,
  };
}

static void
fill_zeros(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  (void)offset;
  (void)total;
  memset(buf, 0, count * sizeof(uint16_t));
}

// XOR pattern: out[ravel(r)] = r_{d-1} ^ ... ^ r_0
// Precomputed over `nframes` frames so fill_xor is just memcpy.
static uint16_t* xor_pattern_buf = NULL;
static size_t xor_pattern_len = 0; // elements in the precomputed buffer

static void
xor_pattern_init(const struct dimension* dims, uint8_t rank, size_t nframes)
{
  // frame = product of all dims except the outermost (slowest)
  size_t frame = 1;
  for (uint8_t i = 1; i < rank; ++i)
    frame *= dims[i].size;
  xor_pattern_len = nframes * frame;
  free(xor_pattern_buf);
  xor_pattern_buf = (uint16_t*)malloc(xor_pattern_len * sizeof(uint16_t));

  // strides for unraveling (row-major, dims[0] outermost)
  size_t strides[16];
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * dims[i + 1].size;

  for (size_t gi = 0; gi < xor_pattern_len; ++gi) {
    uint16_t v = 0;
    size_t rem = gi;
    for (uint8_t d = 0; d < rank; ++d) {
      size_t coord = rem / strides[d];
      rem %= strides[d];
      v ^= (uint16_t)coord;
    }
    xor_pattern_buf[gi] = v;
  }
}

static void
xor_pattern_free(void)
{
  free(xor_pattern_buf);
  xor_pattern_buf = NULL;
  xor_pattern_len = 0;
}

static void
fill_xor(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  (void)total;
  for (size_t done = 0; done < count;) {
    size_t src_off = (offset + done) % xor_pattern_len;
    size_t chunk = xor_pattern_len - src_off;
    if (chunk > count - done)
      chunk = count - done;
    memcpy(buf + done, xor_pattern_buf + src_off, chunk * sizeof(uint16_t));
    done += chunk;
  }
}

typedef void (*fill_fn)(uint16_t* buf,
                        size_t count,
                        size_t offset,
                        size_t total);

// Fill data, pump through writer, flush. Returns 0 on success.
static int
pump_data(struct writer* w, size_t total_elements, fill_fn fill)
{
  const size_t nelements = 32 * 1024 * 1024; // 32M elements = 64 MiB
  uint16_t* data = (uint16_t*)malloc(nelements * sizeof(uint16_t));
  if (!data)
    return 1;

  for (size_t offset = 0; offset < total_elements; offset += nelements) {
    size_t n = nelements;
    if (offset + n > total_elements)
      n = total_elements - offset;
    fill(data, n, offset, total_elements);
    struct slice input = { .beg = data, .end = data + n };
    struct writer_result r = writer_append_wait(w, input);
    if (r.error) {
      log_error("  append failed at offset %zu", offset);
      free(data);
      return 1;
    }
  }

  struct writer_result r = writer_flush(w);
  free(data);
  return r.error;
}

// --- Small compressed+shard smoke test ---

static int
test_compressed_small(void)
{
  log_info("=== test_compressed_small ===");

  const struct dimension dims[] = {
    { .size = 40, .tile_size = 4, .tiles_per_shard = 5 },
    { .size = 2048, .tile_size = 256, .tiles_per_shard = 4 },
    { .size = 2048, .tile_size = 512, .tiles_per_shard = 2 },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3 },
  };
  const size_t total_elements = dim_total_elements(dims, 4);

  struct transpose_stream s = { 0 };
  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 8 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &dss.base,
  };

  CHECK(Fail, transpose_stream_create(&config, &s) == 0);

  const size_t num_epochs =
    (total_elements + s.layout.epoch_elements - 1) / s.layout.epoch_elements;
  log_info("  total: %zu elements, %zu epochs", total_elements, num_epochs);

  CHECK(Fail, pump_data(&s.writer, total_elements, fill_zeros) == 0);

  CHECK(Fail, s.cursor == total_elements);
  log_info("  shards finalized: %zu, total bytes: %zu",
           dss.shards_finalized,
           dss.total_bytes);

  transpose_stream_destroy(&s);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  log_error("  FAIL");
  return 1;
}

// --- Report + pipeline helpers ---

static void
print_metric_row(const struct stream_metric* m, double bytes_per_unit)
{
  if (m->count <= 0)
    return;
  const int N = m->count;
  double avg_ms = (double)m->ms / N;
  double avg_gbs = gb_per_s(bytes_per_unit * N, (double)m->ms);
  int has_best = m->best_ms < 1e29f;

  if (has_best) {
    double best_gbs = gb_per_s(bytes_per_unit, (double)m->best_ms);
    log_info("  %-12s %8.2f %8.2f %10.2f %10.2f",
             m->name,
             avg_gbs,
             best_gbs,
             avg_ms,
             (double)m->best_ms);
  } else {
    log_info(
      "  %-12s %8.2f %8s %10.2f %10s", m->name, avg_gbs, "-", avg_ms, "-");
  }
}

struct sink_stats
{
  size_t total_bytes;
  const struct stream_metric* sink;
};

static void
log_bench_header(const struct transpose_stream* s,
                 size_t total_bytes,
                 size_t total_elements)
{
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;

  log_info("  total:       %.2f GiB (%zu elements, %zu epochs)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements,
           num_epochs);
  log_info(
    "  tile:        %lu elements = %lu KiB  (stride=%lu)",
    (unsigned long)s->layout.tile_elements,
    (unsigned long)(s->layout.tile_stride * s->config.bytes_per_element / 1024),
    (unsigned long)s->layout.tile_stride);
  log_info("  epoch:       %lu slots, %lu MiB pool",
           (unsigned long)s->layout.slot_count,
           (unsigned long)(s->layout.tile_pool_bytes / (1024 * 1024)));
  if (s->config.compress)
    log_info("  compress:    max_chunk=%zu comp_pool=%zu MiB",
             s->max_comp_chunk_bytes,
             s->comp_pool_bytes / (1024 * 1024));
}

static void
print_bench_report(const struct transpose_stream* s,
                   const struct sink_stats* ss,
                   size_t total_bytes,
                   size_t total_elements,
                   float wall_s)
{
  struct stream_metrics m = transpose_stream_get_metrics(s);
  const size_t tile_bytes = s->layout.tile_stride * s->config.bytes_per_element;
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;
  const size_t total_tiles = num_epochs * s->layout.slot_count;
  const size_t total_decompressed = total_tiles * tile_bytes;
  const double comp_ratio =
    total_decompressed > 0
      ? (double)ss->total_bytes / (double)total_decompressed
      : 0.0;

  const double pool_bytes = (double)s->layout.tile_pool_bytes;
  const double comp_pool = (double)s->comp_pool_bytes;

  log_info("");
  log_info("  --- Benchmark Results ---");
  log_info("  Input:        %.2f GiB (%zu elements)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements);
  log_info("  Compressed:   %.2f GiB (ratio: %.3f)",
           (double)ss->total_bytes / (1024.0 * 1024.0 * 1024.0),
           comp_ratio);
  log_info("  Tiles:        %zu (%zu/epoch x %zu epochs)",
           total_tiles,
           (size_t)s->layout.slot_count,
           num_epochs);

  log_info("");
  log_info("  %-12s %8s %8s %10s %10s",
           "Stage",
           "avg GB/s",
           "best GB/s",
           "avg ms",
           "best ms");

  double memcpy_per =
    m.memcpy.count > 0 ? (double)total_bytes / m.memcpy.count : 0;
  print_metric_row(&m.memcpy, memcpy_per);
  double h2d_per_dispatch =
    m.h2d.count > 0 ? (double)total_bytes / m.h2d.count : 0;
  double scatter_per_dispatch =
    m.scatter.count > 0 ? (double)total_bytes / m.scatter.count : 0;
  print_metric_row(&m.h2d, h2d_per_dispatch);
  print_metric_row(&m.scatter, scatter_per_dispatch);
  print_metric_row(&m.compress, pool_bytes);
  double agg_per = m.aggregate.count > 0
                     ? m.aggregate.total_bytes / m.aggregate.count
                     : comp_pool;
  print_metric_row(&m.aggregate, agg_per);
  double d2h_per =
    m.d2h.count > 0 ? m.d2h.total_bytes / m.d2h.count : comp_pool;
  print_metric_row(&m.d2h, d2h_per);
  double sink_per_call =
    ss->sink->count > 0 ? (double)ss->total_bytes / ss->sink->count : 0;
  print_metric_row(ss->sink, sink_per_call);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  log_info("");
  log_info("  Wall time:     %.3f s", wall_s);
  log_info("  Throughput:    %.2f GiB/s", throughput_gib);
}

// --- Benchmark ---
//
// 5D: (T, Z, Y, X, C) with isotropic tiles (2, 64, 64, 64, 1)

static int
test_bench(void)
{
  log_info("=== test_bench ===");

  const struct dimension dims[] = {
    {
      .size = 100, // 1024,
      .tile_size = 2,
      .tiles_per_shard = 16,
      .name = "t",
    },
    {
      .size = 256,
      .tile_size = 64,
      .tiles_per_shard = 2,
      .name = "z",
    },
    {
      .size = 256,
      .tile_size = 64,
      .tiles_per_shard = 2,
      .name = "y",
    },
    {
      .size = 256,
      .tile_size = 64,
      .tiles_per_shard = 2,
      .name = "x",
    },
    {
      .size = 3,
      .tile_size = 1,
      .tiles_per_shard = 3,
      .name = "c",
    },
  };
  const size_t total_elements = dim_total_elements(dims, 5);
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  struct transpose_stream s = { 0 };
  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 5,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &dss.base,
  };

  CHECK(Fail, transpose_stream_create(&config, &s) == 0);
  log_bench_header(&s, total_bytes, total_elements);

  xor_pattern_init(dims, 5, 2);

  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  CHECK(Fail, pump_data(&s.writer, total_elements, fill_xor) == 0);
  float wall_s = platform_toc(&clock);

  if (s.cursor != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)s.cursor,
              (ptrdiff_t)((int64_t)s.cursor - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct sink_stats ss = { .total_bytes = dss.total_bytes,
                             .sink = &dss.sink };
    print_bench_report(&s, &ss, total_bytes, total_elements, wall_s);
  }

  transpose_stream_destroy(&s);
  xor_pattern_free();
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  xor_pattern_free();
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_compressed_small();
  if (!ecode)
    ecode |= test_bench();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
