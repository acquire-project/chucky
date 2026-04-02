#include "stream/config.h"
#include "util/prelude.h"

#include <string.h>

// Stub max_output_size: returns input + 64 for any codec.
static size_t
stub_max_output_size(enum compression_codec codec, size_t chunk_bytes)
{
  (void)codec;
  return chunk_bytes + 64;
}

// Log callback state: records whether the LZ4 warning was seen.
struct warn_state
{
  int saw_lz4_warning;
};

static void
warn_callback(log_Event* ev)
{
  struct warn_state* st = (struct warn_state*)ev->udata;
  if (ev->level == LOG_WARN) {
    char buf[256];
    vsnprintf(buf, sizeof(buf), ev->fmt, ev->ap);
    if (strstr(buf, "LZ4 raw block format is not interoperable"))
      st->saw_lz4_warning = 1;
  }
}

static int
test_lz4_non_standard_warns(void)
{
  log_info("=== test_lz4_non_standard_warns ===");

  struct warn_state state = { 0 };
  log_add_callback(warn_callback, &state, LOG_WARN);

  struct dimension dims[3];
  uint64_t sizes[] = { 0, 64, 64 };
  dims_create(dims, "tyx", sizes);
  uint64_t cs[] = { 1, 32, 32 };
  dims_set_chunk_sizes(dims, 3, cs);
  dims[0].chunks_per_shard = 4;

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 1 << 20,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_LZ4_NON_STANDARD, .level = 1 },
  };

  struct computed_stream_layouts cl;
  CHECK(Fail,
        compute_stream_layouts(&config, 1, stub_max_output_size, &cl) == 0);
  CHECK(Fail, state.saw_lz4_warning);

  computed_stream_layouts_free(&cl);
  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  return test_lz4_non_standard_warns();
}
