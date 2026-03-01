#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_data.h"
#include "zarr_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- L0 collecting sink (captures only level 0, discards others) ---

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
    { .size = 8, .tile_size = 2, .tiles_per_shard = 2, .name = "t" },
    { .size = 16, .tile_size = 8, .tiles_per_shard = 2, .name = "z" },
    { .size = 16, .tile_size = 8, .tiles_per_shard = 2, .name = "y" },
    { .size = 16, .tile_size = 8, .tiles_per_shard = 2, .name = "x" },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c" },
  };
  const struct dimension dims_ms[] = {
    { .size = 8, .tile_size = 2, .tiles_per_shard = 2, .name = "t" },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "z",
      .downsample = 1 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "y",
      .downsample = 1 },
    { .size = 16,
      .tile_size = 8,
      .tiles_per_shard = 2,
      .name = "x",
      .downsample = 1 },
    { .size = 1, .tile_size = 1, .tiles_per_shard = 1, .name = "c" },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  // Compute number of L0 shards
  int num_shards = 1;
  for (int d = 0; d < rank; ++d) {
    int tile_count = (int)(dims[d].size / dims[d].tile_size);
    int shard_count = tile_count / dims[d].tiles_per_shard;
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
    CHECK(Fail1, tile_stream_gpu_create(&config, &s));
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
    CHECK(Fail2b, tile_stream_gpu_create(&config, &s));
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
    },
    {
      .size = 128,
      .tile_size = 16,
      .tiles_per_shard = 4,
      .name = "z",
      .downsample = 1,
    },
    {
      .size = 256,
      .tile_size = 16,
      .tiles_per_shard = 4,
      .name = "y",
      .downsample = 1,
    },
    {
      .size = 256,
      .tile_size = 16,
      .tiles_per_shard = 4,
      .name = "x",
      .downsample = 1,
    },
    {
      .size = 3,
      .tile_size = 1,
      .tiles_per_shard = 3,
      .name = "c",
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
    .nlev = 0, // auto
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

  CHECK(Fail2, tile_stream_gpu_create(&config, &s));
  log_info("  total: %zu elements, LOD levels: %d", total_elements, s.lod.plan.nlev);

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

  if (!ecode && ac > 1) {
    ecode |= test_multiscale_zarr_visual(av[1]);
  }

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
