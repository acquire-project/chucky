#include "dimension.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[4096];

static int
make_tmpdir(void)
{
  return test_tmpdir_create(tmpdir, sizeof(tmpdir));
}

static char*
read_file(const char* key, size_t* out_len)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, key);
  FILE* f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  char* buf = (char*)malloc((size_t)sz + 1);
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[n] = '\0';
  if (out_len)
    *out_len = n;
  return buf;
}

static int
contains(const char* haystack, const char* needle)
{
  return strstr(haystack, needle) != NULL;
}

static int
mk_subdir(const char* name)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, name);
  return test_mkdir(path);
}

// --- array tests ---

static int
test_array_set_and_flush(void)
{
  log_info("=== test_array_set_and_flush ===");
  CHECK(Fail, mk_subdir("arr") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 64, .chunk_size = 16, .name = "y" },
    { .size = 128, .chunk_size = 32, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "arr", &cfg);
  CHECK(Fail2, a);

  CHECK(Fail3, zarr_array_set_attribute(a, "label", "\"hello\"") == 0);
  CHECK(Fail3, zarr_array_flush_metadata(a) == 0);

  char* out = read_file("arr/zarr.json", NULL);
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"shape\":[64,128]"));
  CHECK(Fail_out, contains(out, "\"data_type\":\"uint16\""));
  CHECK(Fail_out, contains(out, "\"label\":\"hello\""));

  free(out);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_attrs_survive_shape_change(void)
{
  log_info("=== test_array_attrs_survive_shape_change ===");
  CHECK(Fail, mk_subdir("stream") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 4, .name = "t" },
    { .size = 64, .chunk_size = 16, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_f32,
    .rank = 2,
    .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "stream", &cfg);
  CHECK(Fail2, a);

  CHECK(Fail3, zarr_array_set_attribute(a, "tag", "\"meta\"") == 0);

  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  uint64_t new_sizes[1] = { 10 };
  CHECK(Fail3, sink->update_append(sink, 0, 1, new_sizes) == 0);

  char* out = read_file("stream/zarr.json", NULL);
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"shape\":[10,64]"));
  CHECK(Fail_out, contains(out, "\"tag\":\"meta\""));

  free(out);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_replace_key(void)
{
  log_info("=== test_array_replace_key ===");
  CHECK(Fail, mk_subdir("rep") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[1] = { { .size = 8, .chunk_size = 4, .name = "x" } };
  struct zarr_array_config cfg = {
    .data_type = dtype_u8, .rank = 1, .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "rep", &cfg);
  CHECK(Fail2, a);

  CHECK(Fail3, zarr_array_set_attribute(a, "k", "1") == 0);
  CHECK(Fail3, zarr_array_set_attribute(a, "k", "99") == 0);
  CHECK(Fail3, zarr_array_flush_metadata(a) == 0);

  char* out = read_file("rep/zarr.json", NULL);
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"k\":99"));
  CHECK(Fail_out, !contains(out, "\"k\":1,"));
  CHECK(Fail_out, !contains(out, "\"k\":1}"));

  free(out);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_reject_bad_inputs(void)
{
  log_info("=== test_array_reject_bad_inputs ===");
  CHECK(Fail, mk_subdir("rej") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[1] = { { .size = 8, .chunk_size = 4, .name = "x" } };
  struct zarr_array_config cfg = {
    .data_type = dtype_u8, .rank = 1, .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "rej", &cfg);
  CHECK(Fail2, a);

  // Malformed JSON.
  CHECK(Fail3, zarr_array_set_attribute(a, "k", "{oops") != 0);
  CHECK(Fail3, zarr_array_set_attribute(a, "k", "[1,") != 0);
  CHECK(Fail3, zarr_array_set_attribute(a, "k", "") != 0);
  // Bad keys.
  CHECK(Fail3, zarr_array_set_attribute(a, "", "1") != 0);
  CHECK(Fail3, zarr_array_set_attribute(a, "a\"b", "1") != 0);
  CHECK(Fail3, zarr_array_set_attribute(a, "a\nb", "1") != 0);

  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_value_shapes(void)
{
  log_info("=== test_array_value_shapes ===");
  CHECK(Fail, mk_subdir("shapes") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[1] = { { .size = 8, .chunk_size = 4, .name = "x" } };
  struct zarr_array_config cfg = {
    .data_type = dtype_u8, .rank = 1, .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "shapes", &cfg);
  CHECK(Fail2, a);

  const char* vals[] = {
    "42",
    "-1.5e2",
    "\"hello\"",
    "null",
    "true",
    "false",
    "[1,2,3]",
    "{\"nested\":{\"k\":[true,null,\"s\"]}}",
    "\"with \\\"escapes\\\" and \\\\backslash\"",
  };
  for (size_t i = 0; i < countof(vals); ++i) {
    char key[16];
    snprintf(key, sizeof(key), "v%zu", i);
    CHECK(Fail3, zarr_array_set_attribute(a, key, vals[i]) == 0);
  }
  CHECK(Fail3, zarr_array_flush_metadata(a) == 0);

  char* out = read_file("shapes/zarr.json", NULL);
  CHECK(Fail3, out);
  for (size_t i = 0; i < countof(vals); ++i) {
    char key[16];
    snprintf(key, sizeof(key), "v%zu", i);
    CHECK(Fail_out, contains(out, key));
  }

  free(out);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_large_value(void)
{
  log_info("=== test_array_large_value ===");
  CHECK(Fail, mk_subdir("big") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[1] = { { .size = 8, .chunk_size = 4, .name = "x" } };
  struct zarr_array_config cfg = {
    .data_type = dtype_u8, .rank = 1, .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "big", &cfg);
  CHECK(Fail2, a);

  // build "[0,1,...,999]"
  const int count = 1000;
  size_t cap = (size_t)count * 8 + 4;
  char* v = (char*)malloc(cap);
  CHECK(Fail3, v);
  size_t p = 0;
  v[p++] = '[';
  for (int i = 0; i < count; ++i) {
    int n = snprintf(v + p, cap - p, "%s%d", i ? "," : "", i);
    CHECK(Fail_v, n > 0 && (size_t)n < cap - p);
    p += (size_t)n;
  }
  v[p++] = ']';
  v[p] = '\0';
  CHECK(Fail_v, zarr_array_set_attribute(a, "big", v) == 0);
  CHECK(Fail_v, zarr_array_flush_metadata(a) == 0);

  char* out = read_file("big/zarr.json", NULL);
  CHECK(Fail_v, out);
  CHECK(Fail_out, contains(out, "\"big\":["));
  CHECK(Fail_out, contains(out, "[0,1,2,"));
  CHECK(Fail_out, contains(out, ",998,999]"));

  free(out);
  free(v);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail_v:
  free(v);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_collide_top_level(void)
{
  log_info("=== test_array_collide_top_level ===");
  CHECK(Fail, mk_subdir("col") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 8, .chunk_size = 4, .name = "y" },
    { .size = 16, .chunk_size = 8, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_u8, .rank = 2, .dimensions = dims,
  };
  struct zarr_array* a = zarr_array_create(s, "col", &cfg);
  CHECK(Fail2, a);

  CHECK(Fail3, zarr_array_set_attribute(a, "shape", "\"annotated\"") == 0);
  CHECK(Fail3, zarr_array_flush_metadata(a) == 0);

  char* out = read_file("col/zarr.json", NULL);
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"shape\":[8,16]"));
  CHECK(Fail_out, contains(out, "\"shape\":\"annotated\""));
  const char* attrs = strstr(out, "\"attributes\"");
  CHECK(Fail_out, attrs);
  CHECK(Fail_out, strstr(attrs, "\"shape\":\"annotated\""));
  const char* ts = strstr(out, "\"shape\":[8,16]");
  CHECK(Fail_out, ts && ts < attrs);

  free(out);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- group handle tests ---

static int
test_group_create_set_destroy(void)
{
  log_info("=== test_group_create_set_destroy ===");
  CHECK(Fail, mk_subdir("g1") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct zarr_group* g = zarr_group_create(s, "g1");
  CHECK(Fail2, g);
  if (zarr_group_set_attribute(g, "meta", "{\"v\":1}") != 0) {
    zarr_group_destroy(g);
    goto Fail2;
  }
  zarr_group_destroy(g); // flushes dirty attrs

  char* out = read_file("g1/zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail_out, contains(out, "\"node_type\":\"group\""));
  CHECK(Fail_out, contains(out, "\"meta\":{\"v\":1}"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_group_multiple_keys(void)
{
  log_info("=== test_group_multiple_keys ===");
  CHECK(Fail, mk_subdir("g2") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct zarr_group* g = zarr_group_create(s, "g2");
  CHECK(Fail2, g);
  CHECK(Fail3, zarr_group_set_attribute(g, "a", "1") == 0);
  CHECK(Fail3, zarr_group_set_attribute(g, "b", "[true,false]") == 0);
  CHECK(Fail3, zarr_group_set_attribute(g, "a", "99") == 0);
  CHECK(Fail3, zarr_group_flush_metadata(g) == 0);

  char* out = read_file("g2/zarr.json", NULL);
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"a\":99"));
  CHECK(Fail_out, contains(out, "\"b\":[true,false]"));
  CHECK(Fail_out, !contains(out, "\"a\":1,"));

  free(out);
  zarr_group_destroy(g);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  zarr_group_destroy(g);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_group_root(void)
{
  log_info("=== test_group_root ===");
  char root[4096];
  snprintf(root, sizeof(root), "%s/gr", tmpdir);
  CHECK(Fail, mk_subdir("gr") == 0);
  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);

  struct zarr_group* g = zarr_group_create(s, "");
  CHECK(Fail2, g);
  if (zarr_group_set_attribute(g, "k", "\"root\"") != 0) {
    zarr_group_destroy(g);
    goto Fail2;
  }
  zarr_group_destroy(g);

  char path[4096];
  snprintf(path, sizeof(path), "%s/gr/zarr.json", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail2, f);
  char buf[4096];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';
  CHECK(Fail2, contains(buf, "\"k\":\"root\""));

  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_group_large_value(void)
{
  log_info("=== test_group_large_value ===");
  CHECK(Fail, mk_subdir("gbig") == 0);
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct zarr_group* g = zarr_group_create(s, "gbig");
  CHECK(Fail2, g);

  // Build a large JSON array (~10 KB) of integers, well over the prior
  // ZARR_GROUP_JSON_MAX_LENGTH (8192) static buffer.
  const int count = 2000;
  size_t cap = (size_t)count * 8 + 4;
  char* v = (char*)malloc(cap);
  CHECK(Fail3, v);
  size_t p = 0;
  v[p++] = '[';
  for (int i = 0; i < count; ++i) {
    int n = snprintf(v + p, cap - p, "%s%d", i ? "," : "", i);
    CHECK(Fail_v, n > 0 && (size_t)n < cap - p);
    p += (size_t)n;
  }
  v[p++] = ']';
  v[p] = '\0';

  CHECK(Fail_v, zarr_group_set_attribute(g, "huge", v) == 0);
  CHECK(Fail_v, zarr_group_flush_metadata(g) == 0);

  size_t out_len = 0;
  char* out = read_file("gbig/zarr.json", &out_len);
  CHECK(Fail_v, out);
  CHECK(Fail_out, contains(out, "\"huge\":["));
  CHECK(Fail_out, contains(out, "[0,1,2,"));
  CHECK(Fail_out, contains(out, ",1998,1999]"));
  CHECK(Fail_out, out_len > p);

  free(out);
  free(v);
  zarr_group_destroy(g);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail_v:
  free(v);
Fail3:
  zarr_group_destroy(g);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  if (make_tmpdir())
    return 1;
  log_info("tmpdir: %s", tmpdir);

  int err = 0;
  err |= test_array_set_and_flush();
  err |= test_array_attrs_survive_shape_change();
  err |= test_array_replace_key();
  err |= test_array_reject_bad_inputs();
  err |= test_array_value_shapes();
  err |= test_array_large_value();
  err |= test_array_collide_top_level();
  err |= test_group_create_set_destroy();
  err |= test_group_multiple_keys();
  err |= test_group_root();
  err |= test_group_large_value();

  test_tmpdir_remove(tmpdir);
  return err;
}
