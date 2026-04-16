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

  test_tmpdir_remove(tmpdir);
  return err;
}
