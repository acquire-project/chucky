#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_data.h"
#include "zarr_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- L0 collecting sink (captures only level 0, discards others; for testing) ---

#define L0_MAX_SHARDS 16

struct l0_shard_writer
{
  struct shard_writer base;
  uint8_t* buf;
  size_t capacity;
  size_t size;
  int finalized;
};

struct l0_discard_writer
{
  struct shard_writer base;
};

struct l0_collecting_sink
{
  struct shard_sink base;
  struct l0_shard_writer writers[L0_MAX_SHARDS];
  struct l0_discard_writer discard;
  int num_shards;
};

static int
l0_write(struct shard_writer* self,
         uint64_t offset,
         const void* beg,
         const void* end)
{
  struct l0_shard_writer* w = (struct l0_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > w->capacity) {
    log_error("l0_write: overflow shard (offset=%lu nbytes=%lu cap=%lu)",
              (unsigned long)offset,
              (unsigned long)nbytes,
              (unsigned long)w->capacity);
    return 1;
  }
  memcpy(w->buf + offset, beg, nbytes);
  if (offset + nbytes > w->size)
    w->size = offset + nbytes;
  return 0;
}

static int
l0_finalize(struct shard_writer* self)
{
  struct l0_shard_writer* w = (struct l0_shard_writer*)self;
  w->finalized = 1;
  return 0;
}

static int
l0_discard_write(struct shard_writer* self,
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
l0_discard_finalize(struct shard_writer* self)
{
  (void)self;
  return 0;
}

static struct shard_writer*
l0_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct l0_collecting_sink* s = (struct l0_collecting_sink*)self;
  if (level != 0)
    return &s->discard.base;
  if ((int)shard_index >= s->num_shards) {
    log_error("l0_open: shard_index %lu >= num_shards %d",
              (unsigned long)shard_index,
              s->num_shards);
    return NULL;
  }
  struct l0_shard_writer* w = &s->writers[shard_index];
  w->finalized = 0;
  w->size = 0;
  return &w->base;
}

static int
l0_sink_init(struct l0_collecting_sink* s,
             int num_shards,
             size_t per_shard_capacity)
{
  *s = (struct l0_collecting_sink){
    .base = { .open = l0_open },
    .discard = { .base = { .write = l0_discard_write,
                           .finalize = l0_discard_finalize } },
    .num_shards = num_shards,
  };
  for (int i = 0; i < num_shards; ++i) {
    s->writers[i] = (struct l0_shard_writer){
      .base = { .write = l0_write, .finalize = l0_finalize },
      .buf = (uint8_t*)calloc(1, per_shard_capacity),
      .capacity = per_shard_capacity,
    };
    if (!s->writers[i].buf)
      return 1;
  }
  return 0;
}

static void
l0_sink_free(struct l0_collecting_sink* s)
{
  for (int i = 0; i < s->num_shards; ++i)
    free(s->writers[i].buf);
  *s = (struct l0_collecting_sink){ 0 };
}

// --- L0 correctness: multiscale vs non-multiscale ---

static int
test_multiscale_l0_correctness(void)
{
  log_info("=== test_multiscale_l0_correctness ===");

  // 5D: t, z, y, x, c. LOD on z, y, x.
  // Small enough for fast testing, large enough to exercise multiple epochs.
  const struct dimension dims[] = {
    { .size = 8, .tile_size = 2, .tiles_per_shard = 2, .name = "t", .storage_position = 0 },
    { .size = 16, .tile_size = 8, .tiles_per_shard = 2, .name = "z", .storage_position = 1 },
    { .size = 16, .tile_size = 8, .tiles_per_shard = 2, .name = "y", .storage_position = 2 },
    { .size = 16, .tile_size = 8, .tiles_per_shard = 2, .name = "x", .storage_position = 3 },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c", .storage_position = 4 },
  };
  const struct dimension dims_ms[] = {
    { .size = 8, .tile_size = 2, .tiles_per_shard = 2, .name = "t", .storage_position = 0 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c", .storage_position = 4 },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  // Compute number of L0 shards
  int num_shards = 1;
  for (int d = 0; d < rank; ++d) {
    int tile_count = (int)(dims[d].size / dims[d].tile_size);
    int shard_count = (int)(tile_count / dims[d].tiles_per_shard);
    num_shards *= shard_count;
  }
  const size_t shard_cap = total_bytes + 4096; // generous per-shard capacity

  log_info("  total: %zu elements, %d L0 shards", total_elements, num_shards);

  // --- Run 1: non-multiscale (baseline) ---
  struct l0_collecting_sink baseline_sink;
  CHECK(Fail0, l0_sink_init(&baseline_sink, num_shards, shard_cap) == 0);

  {
    struct tile_stream_gpu s = { 0 };
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .bytes_per_element = sizeof(uint16_t),
      .rank = rank,
      .dimensions = dims,
      .codec = CODEC_ZSTD,
      .shard_sink = &baseline_sink.base,
    };
    CHECK(Fail1, tile_stream_gpu_create(&config, &s) == 0);
    xor_pattern_init(dims, rank, 2);
    CHECK(Fail1b, pump_data(&s.writer, total_elements, fill_xor) == 0);
    CHECK(Fail1b, s.cursor == total_elements);
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Run2;
  Fail1b:
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Fail1;
  }

Run2:
  // --- Run 2: multiscale ---
  ;
  struct l0_collecting_sink ms_sink;
  CHECK(Fail1, l0_sink_init(&ms_sink, num_shards, shard_cap) == 0);

  {
    struct tile_stream_gpu s = { 0 };
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .bytes_per_element = sizeof(uint16_t),
      .rank = rank,
      .dimensions = dims_ms,
      .codec = CODEC_ZSTD,
      .shard_sink = &ms_sink.base,
    };
    CHECK(Fail2b, tile_stream_gpu_create(&config, &s) == 0);
    xor_pattern_init(dims_ms, rank, 2);
    CHECK(Fail2c, pump_data(&s.writer, total_elements, fill_xor) == 0);
    CHECK(Fail2c, s.cursor == total_elements);
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Compare;
  Fail2c:
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Fail2b;
  }

Compare:
  // --- Compare L0 shard output ---
  {
    int errors = 0;
    for (int i = 0; i < num_shards; ++i) {
      struct l0_shard_writer* b = &baseline_sink.writers[i];
      struct l0_shard_writer* m = &ms_sink.writers[i];

      if (!b->finalized) {
        log_error("  shard %d: baseline not finalized", i);
        errors++;
        continue;
      }
      if (!m->finalized) {
        log_error("  shard %d: multiscale not finalized", i);
        errors++;
        continue;
      }
      if (b->size != m->size) {
        log_error("  shard %d: size mismatch (baseline=%zu, multiscale=%zu)",
                  i,
                  b->size,
                  m->size);
        errors++;
        continue;
      }
      if (memcmp(b->buf, m->buf, b->size) != 0) {
        // Find first difference
        size_t diff_off = 0;
        for (size_t j = 0; j < b->size; ++j) {
          if (b->buf[j] != m->buf[j]) {
            diff_off = j;
            break;
          }
        }
        log_error("  shard %d: data mismatch at byte %zu (of %zu)",
                  i,
                  diff_off,
                  b->size);
        errors++;
      }
    }

    if (errors > 0) {
      log_error("  %d shard comparison errors", errors);
      goto Fail2b;
    }
  }

  log_info("  PASS");
  l0_sink_free(&ms_sink);
  l0_sink_free(&baseline_sink);
  return 0;

Fail2b:
  l0_sink_free(&ms_sink);
Fail1:
  l0_sink_free(&baseline_sink);
Fail0:
  xor_pattern_free();
  log_error("  FAIL");
  return 1;
}

// --- Multiscale zarr visual test ---

static int
test_multiscale_zarr_visual(const char* output_path)
{
  log_info("=== test_multiscale_zarr_visual ===");
  log_info("  output: %s", output_path);

  const struct dimension dims[] = {
    {
      .size = 100,
      .tile_size = 2,
      .tiles_per_shard = 32,
      .name = "t",
      .storage_position = 0,
    },
    {
      .size = 128,
      .tile_size = 16,
      .tiles_per_shard = 4,
      .name = "z",
      .downsample = 1,
      .storage_position = 1,
    },
    {
      .size = 256,
      .tile_size = 16,
      .tiles_per_shard = 4,
      .name = "y",
      .downsample = 1,
      .storage_position = 2,
    },
    {
      .size = 256,
      .tile_size = 16,
      .tiles_per_shard = 4,
      .name = "x",
      .downsample = 1,
      .storage_position = 3,
    },
    {
      .size = 3,
      .tile_size = 1,
      .tiles_per_shard = 3,
      .name = "c",
      .storage_position = 4,
    },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims, rank);

  struct zarr_multiscale_config zcfg = {
    .store_path = output_path,
    .data_type = zarr_dtype_uint16,
    .fill_value = 0,
    .rank = rank,
    .dimensions = dims,
    .nlod = 0, // auto
  };

  struct zarr_multiscale_sink* ms = zarr_multiscale_sink_create(&zcfg);
  CHECK(Fail, ms);

  struct tile_stream_gpu s = { 0 };
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = rank,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = zarr_multiscale_sink_as_shard_sink(ms),
  };

  CHECK(Fail2, tile_stream_gpu_create(&config, &s) == 0);
  log_info("  total: %zu elements, LOD levels: %d", total_elements, s.lod.plan.nlod);

  xor_pattern_init(dims, rank, 2);
  CHECK(Fail3, pump_data(&s.writer, total_elements, fill_xor) == 0);
  CHECK(Fail3, s.cursor == total_elements);

  tile_stream_gpu_destroy(&s);
  xor_pattern_free();
  zarr_multiscale_sink_flush(ms);
  zarr_multiscale_sink_destroy(ms);
  log_info("  PASS — wrote %s", output_path);
  return 0;

Fail3:
  tile_stream_gpu_destroy(&s);
  xor_pattern_free();
Fail2:
  zarr_multiscale_sink_destroy(ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Dim0 LOD correctness: L0 must match spatial-only multiscale ---

static int
test_dim0_l0_correctness(void)
{
  log_info("=== test_dim0_l0_correctness ===");

  // 5D: t, z, y, x, c. Spatial LOD on z, y, x.
  // 8 epochs along t so dim0 levels 1+ accumulate and emit.
  const struct dimension dims_spatial[] = {
    { .size = 8, .tile_size = 2, .tiles_per_shard = 2, .name = "t", .storage_position = 0 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c", .storage_position = 4 },
  };
  const struct dimension dims_dim0[] = {
    { .size = 8,
      .tile_size = 2,
      .tiles_per_shard = 2,
      .name = "t",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c", .storage_position = 4 },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims_spatial, rank);

  // Compute number of L0 shards
  int num_shards = 1;
  for (int d = 0; d < rank; ++d) {
    int tile_count = (int)(dims_spatial[d].size / dims_spatial[d].tile_size);
    int shard_count = (int)(tile_count / dims_spatial[d].tiles_per_shard);
    num_shards *= shard_count;
  }
  const size_t shard_cap = total_elements * sizeof(uint16_t) + 4096;
  log_info("  total: %zu elements, %d L0 shards", total_elements, num_shards);

  // --- Run 1: spatial-only multiscale (baseline) ---
  struct l0_collecting_sink baseline_sink;
  CHECK(Fail0, l0_sink_init(&baseline_sink, num_shards, shard_cap) == 0);

  {
    struct tile_stream_gpu s = { 0 };
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .bytes_per_element = sizeof(uint16_t),
      .rank = rank,
      .dimensions = dims_spatial,
      .codec = CODEC_ZSTD,
      .shard_sink = &baseline_sink.base,
      .reduce_method = lod_reduce_mean,
    };
    CHECK(Fail1, tile_stream_gpu_create(&config, &s) == 0);
    xor_pattern_init(dims_spatial, rank, 2);
    CHECK(Fail1b, pump_data(&s.writer, total_elements, fill_xor) == 0);
    CHECK(Fail1b, s.cursor == total_elements);
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Run2d;
  Fail1b:
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Fail1;
  }

Run2d:
  // --- Run 2: spatial + dim0 multiscale ---
  ;
  struct l0_collecting_sink dim0_sink;
  CHECK(Fail1, l0_sink_init(&dim0_sink, num_shards, shard_cap) == 0);

  {
    struct tile_stream_gpu s = { 0 };
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .bytes_per_element = sizeof(uint16_t),
      .rank = rank,
      .dimensions = dims_dim0,
      .codec = CODEC_ZSTD,
      .shard_sink = &dim0_sink.base,
      .reduce_method = lod_reduce_mean,
      .dim0_reduce_method = lod_reduce_mean,
    };
    CHECK(Fail2b, tile_stream_gpu_create(&config, &s) == 0);
    log_info("  dim0 enabled: nlod=%d, dim0_downsample=%d",
             s.levels.nlod, s.levels.dim0_downsample);
    xor_pattern_init(dims_dim0, rank, 2);
    CHECK(Fail2c, pump_data(&s.writer, total_elements, fill_xor) == 0);
    CHECK(Fail2c, s.cursor == total_elements);
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Compared;
  Fail2c:
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    goto Fail2b;
  }

Compared:
  // --- Compare L0 shard output (must be identical) ---
  {
    int errors = 0;
    for (int i = 0; i < num_shards; ++i) {
      struct l0_shard_writer* b = &baseline_sink.writers[i];
      struct l0_shard_writer* m = &dim0_sink.writers[i];

      if (!b->finalized || !m->finalized) {
        log_error("  shard %d: not finalized (baseline=%d, dim0=%d)",
                  i, b->finalized, m->finalized);
        errors++;
        continue;
      }
      if (b->size != m->size) {
        log_error("  shard %d: size mismatch (baseline=%zu, dim0=%zu)",
                  i, b->size, m->size);
        errors++;
        continue;
      }
      if (memcmp(b->buf, m->buf, b->size) != 0) {
        size_t diff_off = 0;
        for (size_t j = 0; j < b->size; ++j) {
          if (b->buf[j] != m->buf[j]) {
            diff_off = j;
            break;
          }
        }
        log_error("  shard %d: data mismatch at byte %zu (of %zu)",
                  i, diff_off, b->size);
        errors++;
      }
    }

    if (errors > 0) {
      log_error("  %d shard comparison errors", errors);
      goto Fail2b;
    }
  }

  log_info("  PASS");
  l0_sink_free(&dim0_sink);
  l0_sink_free(&baseline_sink);
  return 0;

Fail2b:
  l0_sink_free(&dim0_sink);
Fail1:
  l0_sink_free(&baseline_sink);
Fail0:
  xor_pattern_free();
  log_error("  FAIL");
  return 1;
}

// --- All-level collecting sink (captures every level) ---

#define ALL_MAX_SHARDS 64
#define ALL_MAX_LEVELS 8

struct all_shard_writer
{
  struct shard_writer base;
  uint8_t* buf;
  size_t capacity;
  size_t size;
  int finalized;
  uint8_t level;
  uint64_t shard_index;
};

struct all_level_collecting_sink
{
  struct shard_sink base;
  struct all_shard_writer writers[ALL_MAX_LEVELS][ALL_MAX_SHARDS];
  int num_shards_per_level[ALL_MAX_LEVELS];
  int num_levels;
  size_t per_shard_capacity;
};

static int
all_write(struct shard_writer* self,
          uint64_t offset,
          const void* beg,
          const void* end)
{
  struct all_shard_writer* w = (struct all_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > w->capacity) {
    log_error("all_write: overflow level=%d shard=%lu",
              w->level,
              (unsigned long)w->shard_index);
    return 1;
  }
  memcpy(w->buf + offset, beg, nbytes);
  if (offset + nbytes > w->size)
    w->size = offset + nbytes;
  return 0;
}

static int
all_finalize(struct shard_writer* self)
{
  struct all_shard_writer* w = (struct all_shard_writer*)self;
  w->finalized = 1;
  return 0;
}

static struct shard_writer*
all_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct all_level_collecting_sink* s = (struct all_level_collecting_sink*)self;
  if (level >= s->num_levels) {
    log_error("all_open: level %d >= num_levels %d", level, s->num_levels);
    return NULL;
  }
  if ((int)shard_index >= s->num_shards_per_level[level]) {
    log_error("all_open: level %d shard %lu >= %d",
              level,
              (unsigned long)shard_index,
              s->num_shards_per_level[level]);
    return NULL;
  }
  struct all_shard_writer* w = &s->writers[level][shard_index];
  w->level = level;
  w->shard_index = shard_index;
  w->finalized = 0;
  w->size = 0;
  return &w->base;
}

static void
all_sink_free(struct all_level_collecting_sink* s)
{
  for (int lv = 0; lv < s->num_levels; ++lv)
    for (int i = 0; i < s->num_shards_per_level[lv]; ++i)
      free(s->writers[lv][i].buf);
  *s = (struct all_level_collecting_sink){ 0 };
}

static int
all_sink_init(struct all_level_collecting_sink* s,
              int num_levels,
              const int* num_shards_per_level,
              size_t per_shard_capacity)
{
  *s = (struct all_level_collecting_sink){
    .base = { .open = all_open },
    .num_levels = num_levels,
    .per_shard_capacity = per_shard_capacity,
  };
  for (int lv = 0; lv < num_levels; ++lv) {
    s->num_shards_per_level[lv] = num_shards_per_level[lv];
    for (int i = 0; i < num_shards_per_level[lv]; ++i) {
      s->writers[lv][i] = (struct all_shard_writer){
        .base = { .write = all_write, .finalize = all_finalize },
        .buf = (uint8_t*)calloc(1, per_shard_capacity),
        .capacity = per_shard_capacity,
      };
      if (!s->writers[lv][i].buf) {
        all_sink_free(s);
        return 1;
      }
    }
  }
  return 0;
}

// --- Dim0 multi-epoch: verify higher LOD levels are populated ---

static int
test_dim0_multi_epoch_levels(void)
{
  log_info("=== test_dim0_multi_epoch_levels ===");

  // 5D: t, z, y, x, c. Dim0 + spatial downsample.
  // 16 epochs along t to trigger multiple dim0 emissions.
  // Spatial dims 32 with tile 8 → 4 tiles → level 1 has 16 (≥ tile 8) → nlod≥2
  const struct dimension dims[] = {
    { .size = 32,
      .tile_size = 2,
      .tiles_per_shard = 4,
      .name = "t",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 32,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 32,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c", .storage_position = 4 },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims, rank);

  // We need nlod — compute from lod_plan_init_shapes
  struct lod_plan plan = { 0 };
  {
    uint64_t shape[5], tile_shape[5];
    uint32_t lod_mask = 0;
    for (int i = 0; i < rank; ++i) {
      shape[i] = dims[i].size;
      tile_shape[i] = dims[i].tile_size;
      if (dims[i].downsample)
        lod_mask |= (1u << i);
    }
    CHECK(Fail, lod_plan_init_shapes(&plan, rank, shape, tile_shape,
                                      lod_mask, LOD_MAX_LEVELS, 0) == 0);
  }

  int nlod = plan.nlod;
  log_info("  nlod=%d total_elements=%zu", nlod, total_elements);
  CHECK(Fail, nlod >= 2); // need at least 2 levels for this test

  // Use generous shard allocation — higher levels may have different shard
  // layouts than what we can compute from just shapes. L0 gets a careful count,
  // higher levels get ALL_MAX_SHARDS.
  int num_shards_per_level[ALL_MAX_LEVELS];
  for (int lv = 0; lv < nlod; ++lv) {
    if (lv == 0) {
      int ns = 1;
      for (int d = 0; d < rank; ++d) {
        uint64_t tc = ceildiv(dims[d].size, dims[d].tile_size);
        uint64_t tps = dims[d].tiles_per_shard;
        if (tps == 0) tps = tc;
        uint64_t sc = ceildiv(tc, tps);
        ns *= (int)sc;
      }
      num_shards_per_level[lv] = ns < ALL_MAX_SHARDS ? ns : ALL_MAX_SHARDS;
    } else {
      num_shards_per_level[lv] = ALL_MAX_SHARDS;
    }
    log_info("  level %d: shape=(%lu,%lu,%lu,%lu,%lu) shards=%d",
             lv,
             (unsigned long)plan.shapes[lv][0],
             (unsigned long)plan.shapes[lv][1],
             (unsigned long)plan.shapes[lv][2],
             (unsigned long)plan.shapes[lv][3],
             (unsigned long)plan.shapes[lv][4],
             num_shards_per_level[lv]);
  }

  lod_plan_free(&plan);

  struct all_level_collecting_sink sink;
  CHECK(Fail,
        all_sink_init(&sink, nlod, num_shards_per_level, 256 * 1024) == 0);

  {
    struct tile_stream_gpu s = { 0 };
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .bytes_per_element = sizeof(uint16_t),
      .rank = rank,
      .dimensions = dims,
      .codec = CODEC_ZSTD,
      .shard_sink = &sink.base,
      .reduce_method = lod_reduce_mean,
      .dim0_reduce_method = lod_reduce_mean,
    };

    CHECK(Fail2, tile_stream_gpu_create(&config, &s) == 0);
    log_info("  stream nlod=%d dim0_downsample=%d epochs_per_batch=%u",
             s.levels.nlod,
             s.levels.dim0_downsample,
             s.batch.epochs_per_batch);

    xor_pattern_init(dims, rank, 2);
    int pump_ok = (pump_data(&s.writer, total_elements, fill_xor) == 0);
    int cursor_ok = pump_ok && (s.cursor == total_elements);
    tile_stream_gpu_destroy(&s);
    xor_pattern_free();
    CHECK(Fail2, pump_ok);
    CHECK(Fail2, cursor_ok);
  }

  // Verify: L0 shards should be finalized and non-empty
  {
    int l0_finalized = 0;
    for (int i = 0; i < num_shards_per_level[0]; ++i) {
      if (sink.writers[0][i].finalized && sink.writers[0][i].size > 0)
        l0_finalized++;
    }
    log_info("  L0: %d/%d shards finalized",
             l0_finalized,
             num_shards_per_level[0]);
    CHECK(Fail2, l0_finalized > 0);
  }

  // Verify: level 1+ shards should have at least some finalized non-empty
  for (int lv = 1; lv < nlod; ++lv) {
    int finalized = 0;
    size_t total_bytes = 0;
    for (int i = 0; i < num_shards_per_level[lv]; ++i) {
      if (sink.writers[lv][i].finalized) {
        finalized++;
        total_bytes += sink.writers[lv][i].size;
      }
    }
    log_info("  L%d: %d/%d shards finalized, %zu total bytes",
             lv,
             finalized,
             num_shards_per_level[lv],
             total_bytes);
    CHECK(Fail2, finalized > 0);
    CHECK(Fail2, total_bytes > 0);
  }

  all_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail2:
  all_sink_free(&sink);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_multiscale_l0_correctness();
  ecode |= test_dim0_l0_correctness();
  ecode |= test_dim0_multi_epoch_levels();

  if (!ecode && ac > 1) {
    ecode |= test_multiscale_zarr_visual(av[1]);
  }

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
