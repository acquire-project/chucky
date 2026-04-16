#include "dimension.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[4096];

static int
make_tmpdir(void)
{
  return test_tmpdir_create(tmpdir, sizeof(tmpdir));
}

static int
write_file(const char* key, const char* text)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, key);
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  size_t n = strlen(text);
  size_t w = fwrite(text, 1, n, f);
  fclose(f);
  return w == n ? 0 : 1;
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
mk_subdir(const char* name)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, name);
  return test_mkdir(path);
}

static int
contains(const char* haystack, const char* needle)
{
  return strstr(haystack, needle) != NULL;
}

// --- tests ---

static int
test_merge_into_empty_attrs(void)
{
  log_info("=== test_merge_into_empty_attrs ===");
  const char* src =
    "{\"zarr_format\":3,\"node_type\":\"group\","
    "\"consolidated_metadata\":null,\"attributes\":{}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2,
        zarr_write_attribute(s, NULL, "experiment", "{\"id\":42}") == 0);

  size_t n = 0;
  char* out = read_file("zarr.json", &n);
  CHECK(Fail2, out);
  CHECK(Fail3, contains(out, "\"attributes\":{\"experiment\":{\"id\":42}}"));
  CHECK(Fail3, contains(out, "\"zarr_format\":3"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_merge_preserves_ome(void)
{
  log_info("=== test_merge_preserves_ome ===");
  const char* src =
    "{\"zarr_format\":3,\"node_type\":\"group\","
    "\"attributes\":{\"ome\":{\"version\":\"0.5\"}}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, zarr_write_attribute(s, "", "experiment", "{\"id\":7}") == 0);

  char* out = read_file("zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail3, contains(out, "\"ome\":{\"version\":\"0.5\"}"));
  CHECK(Fail3, contains(out, "\"experiment\":{\"id\":7}"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_replace_existing(void)
{
  log_info("=== test_replace_existing ===");
  const char* src =
    "{\"zarr_format\":3,\"attributes\":{\"ome\":{\"v\":1},"
    "\"experiment\":{\"id\":1}}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2,
        zarr_write_attribute(s, NULL, "experiment", "{\"id\":99}") == 0);

  char* out = read_file("zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail3, contains(out, "\"experiment\":{\"id\":99}"));
  CHECK(Fail3, !contains(out, "\"id\":1"));
  CHECK(Fail3, contains(out, "\"ome\":{\"v\":1}"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_reject_ome(void)
{
  log_info("=== test_reject_ome ===");
  const char* src = "{\"attributes\":{}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, zarr_write_attribute(s, NULL, "ome", "{\"v\":\"0.5\"}") != 0);

  // File must be unchanged.
  char* out = read_file("zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail3, strcmp(out, src) == 0);

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_reject_malformed(void)
{
  log_info("=== test_reject_malformed ===");
  const char* src = "{\"attributes\":{}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, zarr_write_attribute(s, NULL, "bad", "{oops") != 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "bad", "{\"k\":}") != 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "bad", "[1,2,") != 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "bad", "") != 0);

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
test_reject_missing_zarr_json(void)
{
  log_info("=== test_reject_missing_zarr_json ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2,
        zarr_write_attribute(s, "does_not_exist", "k", "\"v\"") != 0);

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
test_reject_bad_attr_key(void)
{
  log_info("=== test_reject_bad_attr_key ===");
  const char* src = "{\"attributes\":{}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, zarr_write_attribute(s, NULL, "", "1") != 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "a\"b", "1") != 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "a\nb", "1") != 0);

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
test_various_value_shapes(void)
{
  log_info("=== test_various_value_shapes ===");
  const char* src = "{\"attributes\":{}}";
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  const char* values[] = {
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

  for (size_t i = 0; i < countof(values); ++i) {
    CHECK(Fail2, write_file("zarr.json", src) == 0);
    char key[32];
    snprintf(key, sizeof(key), "k%zu", i);
    CHECK(Fail2, zarr_write_attribute(s, NULL, key, values[i]) == 0);
    // Re-read and verify.
    char* out = read_file("zarr.json", NULL);
    CHECK(Fail2, out);
    CHECK(Fail2, contains(out, key));
    free(out);
  }

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
test_merge_into_subgroup(void)
{
  log_info("=== test_merge_into_subgroup ===");
  CHECK(Fail, mk_subdir("sub") == 0);
  CHECK(Fail,
        write_file("sub/zarr.json",
                   "{\"attributes\":{\"existing\":1}}") == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, zarr_write_attribute(s, "sub", "added", "[1,2,3]") == 0);

  char* out = read_file("sub/zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail3, contains(out, "\"existing\":1"));
  CHECK(Fail3, contains(out, "\"added\":[1,2,3]"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_no_attributes_in_zarr_json(void)
{
  log_info("=== test_no_attributes_in_zarr_json ===");
  const char* src = "{\"zarr_format\":3,\"node_type\":\"group\"}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  // Missing "attributes" key should fail.
  CHECK(Fail2, zarr_write_attribute(s, NULL, "x", "1") != 0);

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
test_root_null_vs_empty(void)
{
  log_info("=== test_root_null_vs_empty ===");
  const char* src =
    "{\"zarr_format\":3,\"node_type\":\"group\","
    "\"consolidated_metadata\":null,\"attributes\":{}}";

  CHECK(Fail, write_file("zarr.json", src) == 0);
  struct store* s1 = store_fs_create(tmpdir, 0);
  CHECK(Fail, s1);
  CHECK(Fail1, zarr_write_attribute(s1, NULL, "k", "\"v\"") == 0);
  size_t n1 = 0;
  char* out1 = read_file("zarr.json", &n1);
  CHECK(Fail1, out1);
  store_destroy(s1);

  CHECK(Fail_out1, write_file("zarr.json", src) == 0);
  struct store* s2 = store_fs_create(tmpdir, 0);
  CHECK(Fail_out1, s2);
  CHECK(Fail2, zarr_write_attribute(s2, "", "k", "\"v\"") == 0);
  size_t n2 = 0;
  char* out2 = read_file("zarr.json", &n2);
  CHECK(Fail2, out2);

  CHECK(Fail_both, n1 == n2);
  CHECK(Fail_both, memcmp(out1, out2, n1) == 0);
  CHECK(Fail_both, contains(out1, "\"k\":\"v\""));

  free(out1);
  free(out2);
  store_destroy(s2);
  log_info("  PASS");
  return 0;
Fail_both:
  free(out2);
Fail2:
  store_destroy(s2);
Fail_out1:
  free(out1);
  goto Fail;
Fail1:
  store_destroy(s1);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_array_zarr_json(void)
{
  log_info("=== test_array_zarr_json ===");

  CHECK(Fail, mk_subdir("arr") == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 64, .chunk_size = 16, .name = "y" },
    { .size = 128, .chunk_size = 32, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 2,
    .dimensions = dims,
  };

  struct zarr_array* a = zarr_array_create(s, "arr", &cfg);
  CHECK(Fail2, a);

  // Pre-check: top-level array fields present.
  char* pre = read_file("arr/zarr.json", NULL);
  CHECK(Fail3, pre);
  CHECK(Fail_pre, contains(pre, "\"node_type\":\"array\""));
  CHECK(Fail_pre, contains(pre, "\"shape\":[64,128]"));
  CHECK(Fail_pre, contains(pre, "\"data_type\":\"uint16\""));
  CHECK(Fail_pre, contains(pre, "\"chunk_grid\""));
  CHECK(Fail_pre, contains(pre, "\"codecs\""));
  CHECK(Fail_pre, contains(pre, "\"fill_value\""));
  free(pre);
  pre = NULL;

  CHECK(Fail3,
        zarr_write_attribute(s, "arr", "label", "\"hello\"") == 0);

  char* post = read_file("arr/zarr.json", NULL);
  CHECK(Fail3, post);
  // Top-level fields must remain intact.
  CHECK(Fail_post, contains(post, "\"node_type\":\"array\""));
  CHECK(Fail_post, contains(post, "\"shape\":[64,128]"));
  CHECK(Fail_post, contains(post, "\"data_type\":\"uint16\""));
  CHECK(Fail_post, contains(post, "\"chunk_grid\""));
  CHECK(Fail_post, contains(post, "\"codecs\""));
  CHECK(Fail_post, contains(post, "\"fill_value\""));
  // The new attribute landed inside attributes.
  CHECK(Fail_post, contains(post, "\"attributes\""));
  CHECK(Fail_post, contains(post, "\"label\":\"hello\""));

  free(post);
  zarr_array_destroy(a);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_post:
  free(post);
  goto Fail3;
Fail_pre:
  free(pre);
Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_sequential_merges(void)
{
  log_info("=== test_sequential_merges ===");
  const char* src =
    "{\"zarr_format\":3,\"node_type\":\"group\","
    "\"attributes\":{\"ome\":{\"version\":\"0.5\"}}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, zarr_write_attribute(s, NULL, "A", "{\"n\":1}") == 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "B", "[true,false]") == 0);
  CHECK(Fail2, zarr_write_attribute(s, NULL, "A", "{\"n\":42}") == 0);

  char* out = read_file("zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail3, contains(out, "\"zarr_format\":3"));
  CHECK(Fail3, contains(out, "\"node_type\":\"group\""));
  CHECK(Fail3, contains(out, "\"ome\":{\"version\":\"0.5\"}"));
  CHECK(Fail3, contains(out, "\"A\":{\"n\":42}"));
  CHECK(Fail3, contains(out, "\"B\":[true,false]"));
  CHECK(Fail3, !contains(out, "\"n\":1"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_top_level_fields_preserved(void)
{
  log_info("=== test_top_level_fields_preserved ===");
  const char* src =
    "{\"zarr_format\":3,\"node_type\":\"group\","
    "\"consolidated_metadata\":null,"
    "\"attributes\":{\"ome\":{\"version\":\"0.5\","
    "\"multiscales\":[{\"name\":\"x\",\"axes\":[]}]}}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2,
        zarr_write_attribute(s, NULL, "extra", "{\"k\":1}") == 0);

  char* out = read_file("zarr.json", NULL);
  CHECK(Fail2, out);
  CHECK(Fail3, contains(out, "\"zarr_format\":3"));
  CHECK(Fail3, contains(out, "\"node_type\":\"group\""));
  CHECK(Fail3, contains(out, "\"consolidated_metadata\":null"));
  // The entire ome block must be preserved verbatim.
  CHECK(Fail3,
        contains(out,
                 "\"ome\":{\"version\":\"0.5\","
                 "\"multiscales\":[{\"name\":\"x\",\"axes\":[]}]}"));
  CHECK(Fail3, contains(out, "\"extra\":{\"k\":1}"));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  free(out);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_attr_key_collides_with_top_level(void)
{
  log_info("=== test_attr_key_collides_with_top_level ===");

  CHECK(Fail, mk_subdir("arr2") == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 8, .chunk_size = 4, .name = "y" },
    { .size = 16, .chunk_size = 8, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_u8,
    .fill_value = 0,
    .rank = 2,
    .dimensions = dims,
  };

  struct zarr_array* a = zarr_array_create(s, "arr2", &cfg);
  CHECK(Fail2, a);

  // Use attr_key="shape" — collides with top-level array field.
  CHECK(Fail3,
        zarr_write_attribute(s, "arr2", "shape", "\"annotated\"") == 0);

  char* out = read_file("arr2/zarr.json", NULL);
  CHECK(Fail3, out);

  // Top-level shape array must be untouched.
  CHECK(Fail_out, contains(out, "\"shape\":[8,16]"));
  // attributes.shape must contain the new value.
  CHECK(Fail_out, contains(out, "\"shape\":\"annotated\""));

  // Sanity: verify the "annotated" string appears *after* attributes
  // (i.e. inside the attributes object, not replacing top-level shape).
  const char* attrs = strstr(out, "\"attributes\"");
  CHECK(Fail_out, attrs);
  CHECK(Fail_out, strstr(attrs, "\"shape\":\"annotated\""));

  // Top-level shape must appear before the attributes key.
  const char* top_shape = strstr(out, "\"shape\":[8,16]");
  CHECK(Fail_out, top_shape);
  CHECK(Fail_out, top_shape < attrs);

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
test_large_json_value(void)
{
  log_info("=== test_large_json_value ===");
  const char* src = "{\"zarr_format\":3,\"attributes\":{}}";
  CHECK(Fail, write_file("zarr.json", src) == 0);

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  // Build a large JSON array of ~1000 integers.
  const int count = 1000;
  size_t cap = (size_t)count * 8 + 4;
  char* value = (char*)malloc(cap);
  CHECK(Fail2, value);
  size_t p = 0;
  value[p++] = '[';
  for (int i = 0; i < count; ++i) {
    int n = snprintf(value + p, cap - p, "%s%d", i ? "," : "", i);
    CHECK(Fail_val, n > 0 && (size_t)n < cap - p);
    p += (size_t)n;
  }
  value[p++] = ']';
  value[p] = '\0';

  CHECK(Fail_val,
        zarr_write_attribute(s, NULL, "big", value) == 0);

  size_t out_len = 0;
  char* out = read_file("zarr.json", &out_len);
  CHECK(Fail_val, out);
  CHECK(Fail_out, contains(out, "\"big\":["));
  CHECK(Fail_out, contains(out, "\"zarr_format\":3"));
  // Must contain the first and last array element.
  CHECK(Fail_out, contains(out, "[0,1,2,"));
  CHECK(Fail_out, contains(out, ",998,999]"));
  // Sanity: output must be at least as long as the value blob.
  CHECK(Fail_out, out_len > p);

  free(out);
  free(value);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail_val:
  free(value);
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
  err |= test_merge_into_empty_attrs();
  err |= test_merge_preserves_ome();
  err |= test_replace_existing();
  err |= test_reject_ome();
  err |= test_reject_malformed();
  err |= test_reject_missing_zarr_json();
  err |= test_reject_bad_attr_key();
  err |= test_various_value_shapes();
  err |= test_merge_into_subgroup();
  err |= test_no_attributes_in_zarr_json();
  err |= test_root_null_vs_empty();
  err |= test_array_zarr_json();
  err |= test_sequential_merges();
  err |= test_top_level_fields_preserved();
  err |= test_attr_key_collides_with_top_level();
  err |= test_large_json_value();

  test_tmpdir_remove(tmpdir);
  return err;
}
