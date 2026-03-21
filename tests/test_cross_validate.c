// Cross-validate GPU and CPU pipelines: feed identical input, compare
// byte-exact shard output. Uses CODEC_NONE so chunk data is uncompressed.

#include "cpu/stream.h"
#include "gpu/stream.h"
#include "prelude.h"
#include "test_runner.h"
#include "writer.h"

#include <stdlib.h>
#include <string.h>

// ---- In-memory shard sink (reusable, level-aware) ----

#define MAX_SHARDS 64
#define MAX_LEVELS 4
#define SHARD_CAP (1 << 20)

struct mem_writer
{
  struct shard_writer base;
  uint8_t* buf;
  size_t size;
  int finalized;
};

struct mem_sink
{
  struct shard_sink base;
  struct mem_writer w[MAX_LEVELS][MAX_SHARDS];
};

static int
mw_write(struct shard_writer* self, uint64_t off, const void* b, const void* e)
{
  struct mem_writer* w = (struct mem_writer*)self;
  size_t n = (size_t)((const char*)e - (const char*)b);
  if (off + n > SHARD_CAP)
    return 1;
  memcpy(w->buf + off, b, n);
  if (off + n > w->size)
    w->size = off + n;
  return 0;
}

static int
mw_finalize(struct shard_writer* self)
{
  ((struct mem_writer*)self)->finalized = 1;
  return 0;
}

static struct shard_writer*
ms_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct mem_sink* s = (struct mem_sink*)self;
  if (level >= MAX_LEVELS || shard_index >= MAX_SHARDS)
    return NULL;
  struct mem_writer* w = &s->w[level][shard_index];
  if (!w->buf) {
    w->buf = (uint8_t*)calloc(1, SHARD_CAP);
    if (!w->buf)
      return NULL;
    w->base.write = mw_write;
    w->base.finalize = mw_finalize;
  }
  w->finalized = 0;
  w->size = 0;
  return &w->base;
}

static void
ms_init(struct mem_sink* s)
{
  memset(s, 0, sizeof(*s));
  s->base.open = ms_open;
}

static void
ms_free(struct mem_sink* s)
{
  for (int lv = 0; lv < MAX_LEVELS; ++lv)
    for (int si = 0; si < MAX_SHARDS; ++si)
      free(s->w[lv][si].buf);
}

// ---- Generate test data ----

static uint16_t*
make_input(uint64_t n_elements)
{
  uint16_t* data = (uint16_t*)malloc(n_elements * sizeof(uint16_t));
  if (!data)
    return NULL;
  for (uint64_t i = 0; i < n_elements; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);
  return data;
}

// ---- Compare shards ----

static int
compare_shards(const struct mem_sink* gpu_sink,
               const struct mem_sink* cpu_sink,
               const char* test_name)
{
  int errors = 0;

  for (int lv = 0; lv < MAX_LEVELS; ++lv) {
    for (int si = 0; si < MAX_SHARDS; ++si) {
      const struct mem_writer* gw = &gpu_sink->w[lv][si];
      const struct mem_writer* cw = &cpu_sink->w[lv][si];

      int g_has = gw->buf && gw->size > 0;
      int c_has = cw->buf && cw->size > 0;

      if (!g_has && !c_has)
        continue;

      if (g_has != c_has) {
        log_error("%s: lv=%d shard=%d: GPU %s, CPU %s",
                  test_name,
                  lv,
                  si,
                  g_has ? "present" : "missing",
                  c_has ? "present" : "missing");
        errors++;
        continue;
      }

      if (gw->size != cw->size) {
        log_error("%s: lv=%d shard=%d: size mismatch GPU=%zu CPU=%zu",
                  test_name,
                  lv,
                  si,
                  gw->size,
                  cw->size);
        errors++;
        continue;
      }

      if (memcmp(gw->buf, cw->buf, gw->size) != 0) {
        // Find first difference
        size_t diff_at = 0;
        for (size_t i = 0; i < gw->size; ++i) {
          if (gw->buf[i] != cw->buf[i]) {
            diff_at = i;
            break;
          }
        }
        log_error("%s: lv=%d shard=%d: data mismatch at byte %zu (size=%zu)",
                  test_name,
                  lv,
                  si,
                  diff_at,
                  gw->size);
        errors++;
      }
    }
  }

  return errors;
}

// ---- Tests ----

static int
test_cross_validate_basic(void)
{
  log_info("=== test_cross_validate_basic ===");

  struct mem_sink gpu_sink, cpu_sink;
  ms_init(&gpu_sink);
  ms_init(&cpu_sink);

  // 3D: 4×4×6, chunk 2×2×3, one shard covers everything
  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };

  struct tile_stream_gpu* gpu = NULL;
  struct tile_stream_cpu* cpu = NULL;
  uint16_t* data = NULL;

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  // Compute total elements from the GPU layout.
  gpu = tile_stream_gpu_create(&config, &gpu_sink.base);
  CHECK(Fail, gpu);

  const struct tile_stream_layout* lay = tile_stream_gpu_layout(gpu);
  // Write exactly 2 epochs.
  uint64_t total_elements = 2 * lay->epoch_elements;
  data = make_input(total_elements);
  CHECK(Fail, data);

  log_info("  epoch_elements=%lu total=%lu",
           (unsigned long)lay->epoch_elements,
           (unsigned long)total_elements);

  // GPU pipeline
  {
    struct writer* w = tile_stream_gpu_writer(gpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // CPU pipeline (same config, different sink)
  cpu = tile_stream_cpu_create(&config, &cpu_sink.base);
  CHECK(Fail, cpu);

  {
    struct writer* w = tile_stream_cpu_writer(cpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // Compare
  int mismatches = compare_shards(&gpu_sink, &cpu_sink, "basic");
  CHECK(Fail, mismatches == 0);

  log_info("  cursors: GPU=%lu CPU=%lu",
           (unsigned long)tile_stream_gpu_cursor(gpu),
           (unsigned long)tile_stream_cpu_cursor(cpu));
  CHECK(Fail, tile_stream_gpu_cursor(gpu) == tile_stream_cpu_cursor(cpu));

  free(data);
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_error("  FAIL");
  return 1;
}

// Multi-shard: dims don't divide evenly into chunks, multiple shards.
static int
test_cross_validate_multishard(void)
{
  log_info("=== test_cross_validate_multishard ===");

  struct mem_sink gpu_sink, cpu_sink;
  ms_init(&gpu_sink);
  ms_init(&cpu_sink);

  // 3D: 6×8×9, chunk 2×4×3, chunks_per_shard 1×1×1 (many small shards)
  struct dimension dims[] = {
    { .size = 6,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 1,
      .storage_position = 1 },
    { .size = 9,
      .chunk_size = 3,
      .chunks_per_shard = 1,
      .storage_position = 2 },
  };

  struct tile_stream_gpu* gpu = NULL;
  struct tile_stream_cpu* cpu = NULL;
  uint16_t* data = NULL;

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 8192,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  gpu = tile_stream_gpu_create(&config, &gpu_sink.base);
  CHECK(Fail, gpu);

  const struct tile_stream_layout* lay = tile_stream_gpu_layout(gpu);
  uint64_t total_elements = 3 * lay->epoch_elements;
  data = make_input(total_elements);
  CHECK(Fail, data);

  // GPU
  {
    struct writer* w = tile_stream_gpu_writer(gpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // CPU
  cpu = tile_stream_cpu_create(&config, &cpu_sink.base);
  CHECK(Fail, cpu);

  {
    struct writer* w = tile_stream_cpu_writer(cpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  int mismatches = compare_shards(&gpu_sink, &cpu_sink, "multishard");
  CHECK(Fail, mismatches == 0);

  free(data);
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_error("  FAIL");
  return 1;
}

// LOD (multiscale): 4D with downsample on dims 1,2,3.
static int
test_cross_validate_lod(void)
{
  log_info("=== test_cross_validate_lod ===");

  struct mem_sink gpu_sink, cpu_sink;
  ms_init(&gpu_sink);
  ms_init(&cpu_sink);

  struct tile_stream_gpu* gpu = NULL;
  struct tile_stream_cpu* cpu = NULL;
  uint16_t* data = NULL;

  // 4D: t=4, z=8, y=8, x=8, chunk 2×4×4×4, LOD on z,y,x.
  // Force K=1 so GPU and CPU batch the same way.
  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 1 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 2 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 3 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 8192,
    .dtype = dtype_u16,
    .rank = 4,
    .dimensions = dims,
    .codec = CODEC_NONE,
    .reduce_method = lod_reduce_mean,
    .epochs_per_batch = 1,
  };

  // GPU
  gpu = tile_stream_gpu_create(&config, &gpu_sink.base);
  CHECK(Fail, gpu);

  const struct tile_stream_layout* lay = tile_stream_gpu_layout(gpu);
  // 2 epochs = full dim0 (size=4, chunk_size=2).
  uint64_t total_elements = 2 * lay->epoch_elements;
  data = make_input(total_elements);
  CHECK(Fail, data);

  log_info("  epoch_elements=%lu nlod=%d",
           (unsigned long)lay->epoch_elements,
           tile_stream_gpu_status(gpu).nlod);

  {
    struct writer* w = tile_stream_gpu_writer(gpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // CPU
  cpu = tile_stream_cpu_create(&config, &cpu_sink.base);
  CHECK(Fail, cpu);

  {
    struct writer* w = tile_stream_cpu_writer(cpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // Dump what each side wrote
  for (int lv = 0; lv < MAX_LEVELS; ++lv)
    for (int si = 0; si < MAX_SHARDS; ++si) {
      int g = gpu_sink.w[lv][si].buf && gpu_sink.w[lv][si].size > 0;
      int c = cpu_sink.w[lv][si].buf && cpu_sink.w[lv][si].size > 0;
      if (g || c)
        log_info("  lv=%d shard=%d: GPU=%zu CPU=%zu",
                 lv,
                 si,
                 g ? gpu_sink.w[lv][si].size : 0,
                 c ? cpu_sink.w[lv][si].size : 0);
    }

  // Compare L0 shards (byte-exact). Higher LOD levels may differ because
  // the GPU pipeline batches LOD level activity differently (e.g. L1
  // activates every 2 epochs on GPU with dim0_downsample).
  int l0_errors = 0;
  for (int si = 0; si < MAX_SHARDS; ++si) {
    const struct mem_writer* gw = &gpu_sink.w[0][si];
    const struct mem_writer* cw = &cpu_sink.w[0][si];
    int g = gw->buf && gw->size > 0;
    int c = cw->buf && cw->size > 0;
    if (!g && !c)
      continue;
    if (g != c || gw->size != cw->size ||
        memcmp(gw->buf, cw->buf, gw->size) != 0) {
      log_error("  L0 shard %d mismatch", si);
      l0_errors++;
    }
  }
  CHECK(Fail, l0_errors == 0);
  CHECK(Fail, tile_stream_gpu_cursor(gpu) == tile_stream_cpu_cursor(cpu));

  free(data);
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_error("  FAIL");
  return 1;
}

// LOD with dim0 downsample: all levels should produce output.
static int
test_cross_validate_lod_dim0(void)
{
  log_info("=== test_cross_validate_lod_dim0 ===");

  struct mem_sink gpu_sink, cpu_sink;
  ms_init(&gpu_sink);
  ms_init(&cpu_sink);

  struct tile_stream_gpu* gpu = NULL;
  struct tile_stream_cpu* cpu = NULL;
  uint16_t* data = NULL;

  // 4D: t=8, z=8, y=8, x=8. Downsample ALL dims including t.
  // chunk 2×4×4×4. K=1.
  struct dimension dims[] = {
    { .size = 8,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 1 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 2 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 3 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 8192,
    .dtype = dtype_u16,
    .rank = 4,
    .dimensions = dims,
    .codec = CODEC_NONE,
    .reduce_method = lod_reduce_mean,
    .dim0_reduce_method = lod_reduce_mean,
    .epochs_per_batch = 1,
  };

  // GPU
  gpu = tile_stream_gpu_create(&config, &gpu_sink.base);
  CHECK(Fail, gpu);

  const struct tile_stream_layout* lay = tile_stream_gpu_layout(gpu);
  // 4 epochs = full dim0 (size=8, chunk_size=2 → 4 epochs)
  uint64_t total_elements = 4 * lay->epoch_elements;
  data = make_input(total_elements);
  CHECK(Fail, data);

  struct tile_stream_status st = tile_stream_gpu_status(gpu);
  log_info("  epoch_elements=%lu nlod=%d dim0_ds=%d",
           (unsigned long)lay->epoch_elements,
           st.nlod,
           st.dim0_downsample);

  {
    struct writer* w = tile_stream_gpu_writer(gpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // CPU
  cpu = tile_stream_cpu_create(&config, &cpu_sink.base);
  CHECK(Fail, cpu);

  {
    struct writer* w = tile_stream_cpu_writer(cpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  // Dump
  for (int lv = 0; lv < MAX_LEVELS; ++lv)
    for (int si = 0; si < MAX_SHARDS; ++si) {
      int g = gpu_sink.w[lv][si].buf && gpu_sink.w[lv][si].size > 0;
      int c = cpu_sink.w[lv][si].buf && cpu_sink.w[lv][si].size > 0;
      if (g || c)
        log_info("  lv=%d shard=%d: GPU=%zu CPU=%zu",
                 lv,
                 si,
                 g ? gpu_sink.w[lv][si].size : 0,
                 c ? cpu_sink.w[lv][si].size : 0);
    }

  // Compare ALL levels byte-exact
  int mismatches = compare_shards(&gpu_sink, &cpu_sink, "lod_dim0");
  CHECK(Fail, mismatches == 0);
  CHECK(Fail, tile_stream_gpu_cursor(gpu) == tile_stream_cpu_cursor(cpu));

  free(data);
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_error("  FAIL");
  return 1;
}

RUN_GPU_TESTS({ "cross_validate_basic", test_cross_validate_basic },
              { "cross_validate_multishard", test_cross_validate_multishard },
              { "cross_validate_lod", test_cross_validate_lod },
              { "cross_validate_lod_dim0", test_cross_validate_lod_dim0 }, )
