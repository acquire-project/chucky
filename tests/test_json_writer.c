#include "defs.limits.h"
#include "ngff.h"
#include "ngff/ngff_metadata.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/json_writer.h"
#include "zarr/zarr_metadata.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int
test_simple_object(void)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "hello");
  jw_key(&jw, "count");
  jw_int(&jw, 42);
  jw_key(&jw, "enabled");
  jw_bool(&jw, 1);
  jw_key(&jw, "nothing");
  jw_null(&jw);
  jw_object_end(&jw);

  const char* expected =
    "{\"name\":\"hello\",\"count\":42,\"enabled\":true,\"nothing\":null}";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strbuf_len(&sb) == strlen(expected));
  CHECK(Fail, memcmp(strbuf_cstr(&sb), expected, strbuf_len(&sb)) == 0);
  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_nested(void)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_object_begin(&jw);
  jw_key(&jw, "a");
  jw_array_begin(&jw);
  jw_int(&jw, 1);
  jw_int(&jw, 2);
  jw_int(&jw, 3);
  jw_array_end(&jw);
  jw_key(&jw, "b");
  jw_object_begin(&jw);
  jw_key(&jw, "x");
  jw_float(&jw, 3.14);
  jw_object_end(&jw);
  jw_object_end(&jw);

  const char* expected = "{\"a\":[1,2,3],\"b\":{\"x\":3.14}}";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strbuf_len(&sb) == strlen(expected));
  CHECK(Fail, memcmp(strbuf_cstr(&sb), expected, strbuf_len(&sb)) == 0);
  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_string_escaping(void)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_string(&jw, "hello \"world\"\nnew\tline\\back\x01");

  const char* expected = "\"hello \\\"world\\\"\\nnew\\tline\\\\back\\u0001\"";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strbuf_len(&sb) == strlen(expected));
  CHECK(Fail, memcmp(strbuf_cstr(&sb), expected, strbuf_len(&sb)) == 0);
  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_grows_past_inline(void)
{
  // Many writes into a fresh strbuf should succeed (no fixed cap).
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_array_begin(&jw);
  for (int i = 0; i < 1000; ++i)
    jw_int(&jw, i);
  jw_array_end(&jw);

  CHECK(Fail, !jw_error(&jw));
  // 1000 integers comma-separated, plus brackets. Easily past
  // STRBUF_INLINE_CAP.
  CHECK(Fail, strbuf_len(&sb) > 2000);
  CHECK(Fail, strbuf_cstr(&sb)[0] == '[');
  CHECK(Fail, strbuf_cstr(&sb)[strbuf_len(&sb) - 1] == ']');
  strbuf_free(&sb);
  return 0;

Fail:
  strbuf_free(&sb);
  return 1;
}

// Verify that jw_error is 0 on a fresh writer and that downstream writes
// succeed cleanly. Note: actually triggering jw_error in a unit test
// requires either injecting a strbuf OOM or forcing vsnprintf to fail —
// neither is portable without a mock allocator. The underlying overflow
// guard is covered by test_strbuf's test_reserve_overflow_guard; this test
// just confirms the happy-path error flag stays clean.
static int
test_jw_error_clean(void)
{
  log_info("=== test_jw_error_clean ===");
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);
  CHECK(Fail, !jw_error(&jw));
  jw_object_begin(&jw);
  jw_key(&jw, "k");
  jw_string(&jw, "v");
  jw_object_end(&jw);
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strcmp(strbuf_cstr(&sb), "{\"k\":\"v\"}") == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

static int
test_array_commas(void)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_array_begin(&jw);
  jw_string(&jw, "a");
  jw_string(&jw, "b");
  jw_string(&jw, "c");
  jw_array_end(&jw);

  const char* expected = "[\"a\",\"b\",\"c\"]";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strbuf_len(&sb) == strlen(expected));
  CHECK(Fail, memcmp(strbuf_cstr(&sb), expected, strbuf_len(&sb)) == 0);
  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_uint(void)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_uint(&jw, 18446744073709551615ULL);

  const char* expected = "18446744073709551615";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strbuf_len(&sb) == strlen(expected));
  CHECK(Fail, memcmp(strbuf_cstr(&sb), expected, strbuf_len(&sb)) == 0);
  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_zarr_metadata(void)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "array");
  jw_key(&jw, "shape");
  jw_array_begin(&jw);
  jw_uint(&jw, 100);
  jw_uint(&jw, 200);
  jw_uint(&jw, 300);
  jw_array_end(&jw);
  jw_key(&jw, "data_type");
  jw_string(&jw, "uint16");
  jw_key(&jw, "chunk_grid");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "regular");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  jw_uint(&jw, 10);
  jw_uint(&jw, 20);
  jw_uint(&jw, 30);
  jw_array_end(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_key(&jw, "fill_value");
  jw_int(&jw, 0);
  jw_object_end(&jw);

  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, strbuf_len(&sb) > 0);

  const char* s = strbuf_cstr(&sb);
  CHECK(Fail, s[0] == '{');
  CHECK(Fail, s[strbuf_len(&sb) - 1] == '}');

  CHECK(Fail, strstr(s, "\"zarr_format\":3"));
  CHECK(Fail, strstr(s, "\"node_type\":\"array\""));
  CHECK(Fail, strstr(s, "\"shape\":[100,200,300]"));
  CHECK(Fail, strstr(s, "\"chunk_shape\":[10,20,30]"));

  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_zarr_root_json(void)
{
  struct strbuf sb = { 0 };
  CHECK(Fail, zarr_root_json(&sb) == 0);

  const char* expected = "{\"zarr_format\":3,\"node_type\":\"group\","
                         "\"consolidated_metadata\":null,\"attributes\":{}}";
  CHECK(Fail, strbuf_len(&sb) == strlen(expected));
  CHECK(Fail, memcmp(strbuf_cstr(&sb), expected, strbuf_len(&sb)) == 0);

  // Check no duplicate "attributes" key
  const char* s = strbuf_cstr(&sb);
  const char* first = strstr(s, "\"attributes\"");
  CHECK(Fail, first);
  CHECK(Fail, !strstr(first + 1, "\"attributes\""));

  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_zarr_multiscale_group_json(void)
{
  // 3-dim config (t/y/x) with 2 LOD levels
  struct dimension l0_dims[3] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 4, .name = "t" },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "y",
      .downsample = 1 },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "x",
      .downsample = 1 },
  };
  struct dimension l1_dims[3] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 4, .name = "t" },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1 },
  };

  const struct dimension* levels[2] = { l0_dims, l1_dims };

  struct strbuf sb = { 0 };
  CHECK(Fail, ngff_multiscale_group_json(&sb, 3, 2, levels, NULL, NULL) == 0);

  const char* s = strbuf_cstr(&sb);
  CHECK(Fail, strstr(s, "\"version\":\"0.5\""));
  CHECK(Fail, strstr(s, "\"axes\""));
  CHECK(Fail, strstr(s, "\"datasets\""));
  CHECK(Fail, strstr(s, "\"coordinateTransformations\""));
  CHECK(Fail, strstr(s, "\"attributes\":{\"ome\""));

  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_scale_clamped_dim(void)
{
  // y has size 10, chunk_size 8 -> ceildiv(10,8)=2 chunks, but halving
  // clamps to 8 (chunk_size). The scale factor must be 2x (one downsample),
  // not 10/8=1.25 (the old L0/Ln ratio).
  struct dimension l0[3] = {
    { .size = 0, .chunk_size = 1, .name = "t" },
    { .size = 10, .chunk_size = 8, .name = "y", .downsample = 1 },
    { .size = 64, .chunk_size = 8, .name = "x", .downsample = 1 },
  };
  struct dimension l1[3] = {
    { .size = 0, .chunk_size = 1, .name = "t" },
    { .size = 8, .chunk_size = 8, .name = "y", .downsample = 1 },
    { .size = 32, .chunk_size = 8, .name = "x", .downsample = 1 },
  };
  struct dimension l2[3] = {
    { .size = 0, .chunk_size = 1, .name = "t" },
    { .size = 8, .chunk_size = 8, .name = "y", .downsample = 1 },
    { .size = 16, .chunk_size = 8, .name = "x", .downsample = 1 },
  };

  const struct dimension* levels[3] = { l0, l1, l2 };

  struct strbuf sb = { 0 };
  CHECK(Fail, ngff_multiscale_group_json(&sb, 3, 3, levels, NULL, NULL) == 0);

  const char* s = strbuf_cstr(&sb);
  // L0: scale=[1.0, 1.0, 1.0]
  CHECK(Fail, strstr(s, "\"scale\":[1.0,1.0,1.0]"));
  // L1: t=1, y=2 (one downsample), x=2 (one downsample)
  CHECK(Fail, strstr(s, "\"scale\":[1.0,2.0,2.0]"));
  // L2: t=1, y=2 (still one -- y dropped after L0->L1), x=4 (two downsamples)
  CHECK(Fail, strstr(s, "\"scale\":[1.0,2.0,4.0]"));

  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_zarr_array_json_lz4(void)
{
  struct dimension dims[3] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 2, .name = "t" },
    { .size = 64, .chunk_size = 32, .chunks_per_shard = 1, .name = "y" },
    { .size = 64, .chunk_size = 32, .chunks_per_shard = 1, .name = "x" },
  };
  uint64_t cps[3] = { 2, 1, 1 };
  struct codec_config codec = { .id = CODEC_LZ4_NON_STANDARD, .level = 1 };

  struct strbuf sb = { 0 };
  CHECK(Fail,
        zarr_array_json(&sb, 3, dims, dtype_u16, 0.0, cps, codec, NULL) == 0);

  // The codec name in zarr metadata must be "lz4" (not "lz4_raw")
  CHECK(Fail, strstr(strbuf_cstr(&sb), "\"name\":\"lz4\""));

  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_zarr_array_json_zstd(void)
{
  struct dimension dims[3] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 2, .name = "t" },
    { .size = 64, .chunk_size = 32, .chunks_per_shard = 1, .name = "y" },
    { .size = 64, .chunk_size = 32, .chunks_per_shard = 1, .name = "x" },
  };
  uint64_t cps[3] = { 2, 1, 1 };
  struct codec_config codec = { .id = CODEC_ZSTD, .level = 3 };

  struct strbuf sb = { 0 };
  CHECK(Fail,
        zarr_array_json(&sb, 3, dims, dtype_u16, 0.0, cps, codec, NULL) == 0);

  CHECK(Fail,
        strstr(strbuf_cstr(&sb),
               "\"name\":\"zstd\",\"configuration\":"
               "{\"level\":3,\"checksum\":false}"));

  strbuf_free(&sb);
  return 0;

Fail:
  log_error("  got: %s", strbuf_cstr(&sb));
  strbuf_free(&sb);
  return 1;
}

static int
test_blosc_block_metadata(void)
{
  struct dimension dims[1] = {
    { .size = 65536, .chunk_size = 32768, .chunks_per_shard = 2, .name = "x" },
  };
  const uint64_t cps[] = { 2 };
  struct strbuf sb = { 0 };
  struct codec_config codec = {
    .id = CODEC_BLOSC_ZSTD,
    .level = 5,
    .shuffle = CODEC_SHUFFLE_BIT,
    .blosc_block_bytes = 4097,
  };
  CHECK(Fail,
        zarr_array_json(&sb, 1, dims, dtype_u16, 0, cps, codec, NULL) == 0);
  CHECK(Fail, strstr(strbuf_cstr(&sb), "\"blocksize\":4097"));
  CHECK(Fail, strstr(strbuf_cstr(&sb), "\"typesize\":2"));
  strbuf_free(&sb);
  codec.blosc_block_bytes = 0;
  CHECK(Fail,
        zarr_array_json(&sb, 1, dims, dtype_u16, 0, cps, codec, NULL) != 0);
  CHECK(Fail, strbuf_len(&sb) == 0);
  codec.id = CODEC_BLOSC_LZ4;
  codec.level = 0;
  CHECK(Fail,
        zarr_array_json(&sb, 1, dims, dtype_u16, 0, cps, codec, NULL) != 0);
  // Non-Blosc codecs do not require this field.
  codec.id = CODEC_ZSTD;
  CHECK(Fail,
        zarr_array_json(&sb, 1, dims, dtype_u16, 0, cps, codec, NULL) == 0);
  strbuf_free(&sb);
  return 0;
Fail:
  strbuf_free(&sb);
  return 1;
}

int
main(void)
{
  int rc = 0;
  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "simple_object", test_simple_object },
    { "nested", test_nested },
    { "string_escaping", test_string_escaping },
    { "grows_past_inline", test_grows_past_inline },
    { "jw_error_clean", test_jw_error_clean },
    { "array_commas", test_array_commas },
    { "uint", test_uint },
    { "zarr_metadata", test_zarr_metadata },
    { "zarr_root_json", test_zarr_root_json },
    { "zarr_multiscale_group_json", test_zarr_multiscale_group_json },
    { "scale_clamped_dim", test_scale_clamped_dim },
    { "zarr_array_json_lz4", test_zarr_array_json_lz4 },
    { "zarr_array_json_zstd", test_zarr_array_json_zstd },
    { "blosc_block_metadata", test_blosc_block_metadata },
  };
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    int r = tests[i].fn();
    if (r) {
      log_error("  FAIL: %s", tests[i].name);
      rc = 1;
    } else {
      log_info("  PASS: %s", tests[i].name);
    }
  }
  return rc;
}
