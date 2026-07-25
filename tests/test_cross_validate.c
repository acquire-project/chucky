#include "stream/layouts.h"
// Cross-validate GPU and CPU pipelines: feed identical input, compare
// byte-exact shard output. Uses CODEC_NONE so chunk data is uncompressed.

#include "stream.cpu.h"
#include "stream.gpu.h"
#include "test_data.h"
#include "test_runner.h"
#include "test_shard_verify.h"
#include "util/prelude.h"
#include "writer.h"

#include <stdlib.h>
#include <string.h>

// ---- In-memory shard sink (reusable, level-aware) ----

#define MAX_SHARDS 64
#define MAX_LEVELS 4
#define SHARD_CAP (4 << 20)

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
  size_t alignment; // 0 = none; >0 reported as required_shard_alignment
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

static size_t
ms_alignment(const struct shard_sink* self)
{
  return ((const struct mem_sink*)self)->alignment;
}

static void
ms_init(struct mem_sink* s)
{
  memset(s, 0, sizeof(*s));
  s->base.open = ms_open;
  s->base.required_shard_alignment = ms_alignment;
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
    .codec = { .id = CODEC_NONE },
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
    .codec = { .id = CODEC_NONE },
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
    .codec = { .id = CODEC_NONE },
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

  // Compare all levels byte-exact. Higher LODs in contiguous mode (page_size
  // == 0) are sensitive to the GPU lod_view base-pointer (#135): a stale
  // base or incorrect rebase produces wrong bytes for LOD >= 1 shards.
  int mismatches = compare_shards(&gpu_sink, &cpu_sink, "lod");
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
    .codec = { .id = CODEC_NONE },
    .reduce_method = lod_reduce_mean,
    .append_reduce_method = lod_reduce_mean,
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
           st.append_downsample);

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

// ---- Page-aligned tail carry (lazy pipeline) ----
// A sink alignment requirement activates the carry-over delivery path,
// whose tail kernels consume an upload the host makes only after the
// previous batch delivers — later than the kernels are enqueued.
// CODEC_NONE keeps the race window open (aggregate runs as soon as it is
// enqueued); the CPU pipeline is the byte-exact oracle.
#define TC_Z 32  // 32 epochs (chunk_size 1), one shard generation
#define TC_Y 144 // 2 chunks of 72
#define TC_X 80  // 2 chunks of 40

static int
test_gpu_page_aligned_tail_carry(void)
{
  log_info("=== test_gpu_page_aligned_tail_carry ===");

  struct mem_sink gpu_sink, cpu_sink;
  ms_init(&gpu_sink);
  ms_init(&cpu_sink);
  gpu_sink.alignment = 4096;
  cpu_sink.alignment = 4096;

  struct tile_stream_gpu* gpu = NULL;
  struct tile_stream_cpu* cpu = NULL;
  uint16_t* data = NULL;

  struct dimension dims[] = {
    { .size = TC_Z,
      .chunk_size = 1,
      .chunks_per_shard = TC_Z,
      .storage_position = 0 },
    { .size = TC_Y,
      .chunk_size = 72,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = TC_X,
      .chunk_size = 40,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };

  // chunk = 72*40 u16 = 5760 B; 4 chunks/epoch, 2 epochs/batch => 46080 B
  // per shard per batch = 11*4096 + 1024, so every batch moves the ragged
  // tail and a one-generation-stale read is wrong from batch 2 onward.
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 1 << 16,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 2,
  };

  const uint64_t total_elements = (uint64_t)TC_Z * TC_Y * TC_X;

  gpu = tile_stream_gpu_create(&config, &gpu_sink.base);
  CHECK(Fail, gpu);
  data = make_input(total_elements);
  CHECK(Fail, data);

  {
    struct writer* w = tile_stream_gpu_writer(gpu);
    size_t bytes = total_elements * sizeof(uint16_t);
    struct slice sl = { .beg = data, .end = (const char*)data + bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
    r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

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

  int mismatches = compare_shards(&gpu_sink, &cpu_sink, "page_tail_carry");
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

// ---- GPU zstd round trip ----
// nvcomp and libzstd emit different frame bytes for the same input, so
// shards cannot be byte-compared; compare decompressed values instead.

#define RT_Z 16
#define RT_YX 512
#define RT_CHUNK_YX 128
#define RT_CPS 4 // chunks per shard, every dim

static int
test_gpu_zstd_round_trip(void)
{
  log_info("=== test_gpu_zstd_round_trip ===");

  struct mem_sink gpu_sink, cpu_sink;
  ms_init(&gpu_sink);
  ms_init(&cpu_sink);

  struct tile_stream_gpu* gpu = NULL;
  struct tile_stream_cpu* cpu = NULL;
  uint64_t* g_off = NULL;
  uint64_t* g_sz = NULL;
  uint64_t* c_off = NULL;
  uint64_t* c_sz = NULL;
  uint8_t* g_dec = NULL;
  uint8_t* c_dec = NULL;
  uint16_t* expect = NULL;
  int pattern_inited = 0;

  struct dimension dims[] = {
    { .size = RT_Z,
      .chunk_size = 1,
      .chunks_per_shard = RT_CPS,
      .storage_position = 0 },
    { .size = RT_YX,
      .chunk_size = RT_CHUNK_YX,
      .chunks_per_shard = RT_CPS,
      .storage_position = 1 },
    { .size = RT_YX,
      .chunk_size = RT_CHUNK_YX,
      .chunks_per_shard = RT_CPS,
      .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 1 << 20,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .epochs_per_batch = 4,
  };

  const size_t chunk_elems = (size_t)RT_CHUNK_YX * RT_CHUNK_YX;
  const size_t chunk_bytes = chunk_elems * sizeof(uint16_t);
  const uint64_t tps_total = (uint64_t)RT_CPS * RT_CPS * RT_CPS;
  const int n_shards = RT_Z / RT_CPS;
  const uint64_t total_elements = (uint64_t)RT_Z * RT_YX * RT_YX;

  xor_pattern_init(dims, 3, RT_Z);
  pattern_inited = 1;

  gpu = tile_stream_gpu_create(&config, &gpu_sink.base);
  CHECK(Fail, gpu);
  CHECK(Fail,
        pump_data(tile_stream_gpu_writer(gpu), total_elements, fill_xor) == 0);

  cpu = tile_stream_cpu_create(&config, &cpu_sink.base);
  CHECK(Fail, cpu);
  CHECK(Fail,
        pump_data(tile_stream_cpu_writer(cpu), total_elements, fill_xor) == 0);

  CHECK(Fail, tile_stream_gpu_cursor(gpu) == tile_stream_cpu_cursor(cpu));

  g_off = (uint64_t*)malloc(tps_total * sizeof(uint64_t));
  g_sz = (uint64_t*)malloc(tps_total * sizeof(uint64_t));
  c_off = (uint64_t*)malloc(tps_total * sizeof(uint64_t));
  c_sz = (uint64_t*)malloc(tps_total * sizeof(uint64_t));
  g_dec = (uint8_t*)malloc(chunk_bytes);
  c_dec = (uint8_t*)malloc(chunk_bytes);
  expect = (uint16_t*)malloc(chunk_bytes);
  CHECK(Fail, g_off && g_sz && c_off && c_sz && g_dec && c_dec && expect);

  {
    int errors = 0;
    for (int si = 0; si < n_shards; ++si) {
      const struct mem_writer* gw = &gpu_sink.w[0][si];
      const struct mem_writer* cw = &cpu_sink.w[0][si];
      CHECK(Fail, gw->buf && gw->size > 0 && gw->finalized);
      CHECK(Fail, cw->buf && cw->size > 0 && cw->finalized);
      CHECK(Fail,
            shard_index_parse(gw->buf, gw->size, tps_total, g_off, g_sz) == 0);
      CHECK(Fail,
            shard_index_parse(cw->buf, cw->size, tps_total, c_off, c_sz) == 0);

      for (uint64_t slot = 0; slot < tps_total; ++slot) {
        // Zarr v3 shard index: row-major chunk coords within the shard.
        const uint64_t cz = slot / (RT_CPS * RT_CPS);
        const uint64_t cy = (slot / RT_CPS) % RT_CPS;
        const uint64_t cx = slot % RT_CPS;
        const uint64_t z = (uint64_t)si * RT_CPS + cz; // chunk_size_z == 1

        CHECK(Fail, g_sz[slot] > 0 && c_sz[slot] > 0);
        CHECK(Fail, g_off[slot] + g_sz[slot] <= gw->size);
        CHECK(Fail, c_off[slot] + c_sz[slot] <= cw->size);

        CHECK(Fail,
              chunk_decompress(
                gw->buf + g_off[slot], g_sz[slot], g_dec, chunk_bytes) == 0);
        CHECK(Fail,
              chunk_decompress(
                cw->buf + c_off[slot], c_sz[slot], c_dec, chunk_bytes) == 0);

        if (memcmp(g_dec, c_dec, chunk_bytes) != 0) {
          if (errors < 5)
            log_error("  shard %d slot %lu: GPU != CPU decompressed values",
                      si,
                      (unsigned long)slot);
          errors++;
        }

        for (uint64_t r = 0; r < RT_CHUNK_YX; ++r)
          for (uint64_t c = 0; c < RT_CHUNK_YX; ++c)
            expect[r * RT_CHUNK_YX + c] =
              (uint16_t)(z ^ (cy * RT_CHUNK_YX + r) ^ (cx * RT_CHUNK_YX + c));
        if (memcmp(g_dec, expect, chunk_bytes) != 0) {
          if (errors < 5)
            log_error("  shard %d slot %lu: GPU values != fill pattern",
                      si,
                      (unsigned long)slot);
          errors++;
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  free(g_off);
  free(g_sz);
  free(c_off);
  free(c_sz);
  free(g_dec);
  free(c_dec);
  free(expect);
  xor_pattern_free();
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_info("  PASS");
  return 0;

Fail:
  free(g_off);
  free(g_sz);
  free(c_off);
  free(c_sz);
  free(g_dec);
  free(c_dec);
  free(expect);
  if (pattern_inited)
    xor_pattern_free();
  tile_stream_gpu_destroy(gpu);
  tile_stream_cpu_destroy(cpu);
  ms_free(&gpu_sink);
  ms_free(&cpu_sink);
  log_error("  FAIL");
  return 1;
}

// ---- GPU zstd determinism ----
// Write order across shards is not deterministic, so the content hash must
// be order-insensitive.

struct hash_writer
{
  struct shard_writer base;
  struct hash_sink* owner;
  uint8_t level;
  uint64_t shard_index;
};

struct hash_sink
{
  struct shard_sink base;
  struct hash_writer w[MAX_LEVELS][MAX_SHARDS];
  uint64_t fnv_xor;
  uint64_t total_bytes;
  uint64_t write_count;
  uint64_t finalize_count;
};

static uint64_t
fnv1a_u64(uint64_t h, uint64_t v)
{
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xFF;
    h *= 1099511628211ULL;
  }
  return h;
}

static int
hw_write(struct shard_writer* self, uint64_t off, const void* b, const void* e)
{
  struct hash_writer* hw = (struct hash_writer*)self;
  const uint8_t* p = (const uint8_t*)b;
  const size_t n = (size_t)((const uint8_t*)e - p);
  uint64_t h = 14695981039346656037ULL;
  h = fnv1a_u64(h, hw->level);
  h = fnv1a_u64(h, hw->shard_index);
  h = fnv1a_u64(h, off);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  hw->owner->fnv_xor ^= h;
  hw->owner->total_bytes += n;
  hw->owner->write_count++;
  return 0;
}

static int
hw_finalize(struct shard_writer* self)
{
  ((struct hash_writer*)self)->owner->finalize_count++;
  return 0;
}

static struct shard_writer*
hs_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct hash_sink* s = (struct hash_sink*)self;
  if (level >= MAX_LEVELS || shard_index >= MAX_SHARDS)
    return NULL;
  struct hash_writer* w = &s->w[level][shard_index];
  w->base.write = hw_write;
  w->base.finalize = hw_finalize;
  w->owner = s;
  w->level = level;
  w->shard_index = shard_index;
  return &w->base;
}

static void
hs_init(struct hash_sink* s)
{
  memset(s, 0, sizeof(*s));
  s->base.open = hs_open;
}

// Sized so compress is still running while later batches fill; smaller and
// the test stops exercising pool reuse under load.
#define DET_Z 24
#define DET_YX 4096
#define DET_CHUNK_YX 512
#define DET_PATTERN_FRAMES 16

static int
run_gpu_zstd_once(const struct tile_stream_configuration* config,
                  uint64_t total_elements,
                  struct hash_sink* sink)
{
  hs_init(sink);
  struct tile_stream_gpu* gpu = tile_stream_gpu_create(config, &sink->base);
  CHECK(Fail, gpu);
  CHECK(Fail,
        pump_data(tile_stream_gpu_writer(gpu), total_elements, fill_xor) == 0);
  tile_stream_gpu_destroy(gpu);
  return 0;

Fail:
  tile_stream_gpu_destroy(gpu);
  return 1;
}

static int
test_gpu_zstd_determinism(void)
{
  log_info("=== test_gpu_zstd_determinism ===");

  struct hash_sink a, b;
  int pattern_inited = 0;

  struct dimension dims[] = {
    { .size = DET_Z,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .storage_position = 0 },
    { .size = DET_YX,
      .chunk_size = DET_CHUNK_YX,
      .chunks_per_shard = 8,
      .storage_position = 1 },
    { .size = DET_YX,
      .chunk_size = DET_CHUNK_YX,
      .chunks_per_shard = 8,
      .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 16 << 20,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .epochs_per_batch = 2,
  };

  const uint64_t total_elements = (uint64_t)DET_Z * DET_YX * DET_YX;

  xor_pattern_init(dims, 3, DET_PATTERN_FRAMES);
  pattern_inited = 1;

  CHECK(Fail, run_gpu_zstd_once(&config, total_elements, &a) == 0);
  CHECK(Fail, run_gpu_zstd_once(&config, total_elements, &b) == 0);

  log_info("  run1: bytes=%llu writes=%llu hash=%016llx",
           (unsigned long long)a.total_bytes,
           (unsigned long long)a.write_count,
           (unsigned long long)a.fnv_xor);
  log_info("  run2: bytes=%llu writes=%llu hash=%016llx",
           (unsigned long long)b.total_bytes,
           (unsigned long long)b.write_count,
           (unsigned long long)b.fnv_xor);

  CHECK(Fail, a.total_bytes > 0);
  CHECK(Fail, a.finalize_count == DET_Z / 4);
  CHECK(Fail, a.total_bytes == b.total_bytes);
  CHECK(Fail, a.write_count == b.write_count);
  CHECK(Fail, a.fnv_xor == b.fnv_xor);

  xor_pattern_free();
  log_info("  PASS");
  return 0;

Fail:
  if (pattern_inited)
    xor_pattern_free();
  log_error("  FAIL");
  return 1;
}

RUN_GPU_TESTS({ "cross_validate_basic", test_cross_validate_basic },
              { "cross_validate_multishard", test_cross_validate_multishard },
              { "cross_validate_lod", test_cross_validate_lod },
              { "cross_validate_lod_dim0", test_cross_validate_lod_dim0 },
              { "gpu_page_aligned_tail_carry",
                test_gpu_page_aligned_tail_carry },
              { "gpu_zstd_round_trip", test_gpu_zstd_round_trip },
              { "gpu_zstd_determinism", test_gpu_zstd_determinism }, )
