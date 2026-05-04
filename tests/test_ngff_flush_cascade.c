// Regression: writer_flush must cascade through ngff_multiscale's flush hook
// to the multiscale's own group metadata and to each child zarr_array. Two
// cases are covered:
//   A) attribute set on the multiscale itself surfaces in ms/zarr.json.
//   B) attribute set on a child level surfaces in ms/<level>/zarr.json.

#include "dimension.h"
#include "ngff.h"
#include "ngff/ngff_multiscale.h"
#include "store.h"
#include "stream.cpu.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr.h"
#include "zarr/zarr_array.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[4096];

static char*
read_file(const char* rel)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, rel);
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
  return buf;
}

static int
contains(const char* h, const char* n)
{
  return strstr(h, n) != NULL;
}

static int
mk_subdir(const char* name)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, name);
  return test_mkdir(path);
}

// Build a 2D multiscale at "ms": t (unbounded append) + x (size 16, ds=1).
// The downsample on x enables multiscale (nlod >= 2).
static void
build_dims(struct dimension dims[2])
{
  dims[0] = (struct dimension){ .size = 0,
                                .chunk_size = 1,
                                .chunks_per_shard = 2,
                                .name = "t",
                                .storage_position = 0 };
  dims[1] = (struct dimension){ .size = 16,
                                .chunk_size = 8,
                                .chunks_per_shard = 1,
                                .name = "x",
                                .downsample = 1,
                                .storage_position = 1 };
}

// Drive a tile_stream_cpu over the given sink and call writer_flush.
// No data is appended: flush with cursor==0 is enough to exercise the
// cascade and avoids the side-effect metadata rewrite that update_append
// would trigger when dim0 grows.
static int
run_stream_with_flush(struct shard_sink* sink)
{
  struct dimension dims[2];
  build_dims(dims);

  struct tile_stream_configuration cfg = {
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .buffer_capacity_bytes = 4096,
    .reduce_method = lod_reduce_mean,
  };

  struct tile_stream_cpu* ts = tile_stream_cpu_create(&cfg, sink);
  if (!ts)
    return 1;

  struct writer_result fr = writer_flush(tile_stream_cpu_writer(ts));
  int rc = fr.error ? 1 : 0;
  tile_stream_cpu_destroy(ts);
  return rc;
}

// Case A: setting an attribute on the multiscale group should be persisted
// when writer_flush cascades to ngff_multiscale_flush_fn.
static int
test_cascade_to_multiscale_group(void)
{
  log_info("=== test_cascade_to_multiscale_group ===");
  CHECK(Fail, mk_subdir("caseA") == 0);
  char root[4096];
  snprintf(root, sizeof(root), "%s/caseA", tmpdir);
  struct store* s = store_fs_create(root, 1);
  CHECK(Fail, s);

  // Parent must seed the root group before the multiscale subgroup.
  {
    struct zarr_group* g = zarr_group_create(s, "");
    CHECK(Fail2, g);
    zarr_group_destroy(g);
  }

  struct dimension dims[2];
  build_dims(dims);
  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .nlod = 2,
  };
  struct ngff_multiscale* ms = ngff_multiscale_create(s, "ms", &cfg);
  CHECK(Fail2, ms);

  CHECK(Fail3, ngff_multiscale_set_attribute(ms, "tag", "\"streamed\"") == 0);

  CHECK(Fail3, run_stream_with_flush(ngff_multiscale_as_shard_sink(ms)) == 0);

  char* out = read_file("caseA/ms/zarr.json");
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"tag\":\"streamed\""));
  // OME block must still be present alongside the custom attr.
  CHECK(Fail_out, contains(out, "\"multiscales\""));

  free(out);
  ngff_multiscale_destroy(ms);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  ngff_multiscale_destroy(ms);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// Case B: setting an attribute on a child zarr_array should be persisted
// when writer_flush cascades through ngff_multiscale_flush_fn into each
// child's flush hook.
static int
test_cascade_to_child_array(void)
{
  log_info("=== test_cascade_to_child_array ===");
  CHECK(Fail, mk_subdir("caseB") == 0);
  char root[4096];
  snprintf(root, sizeof(root), "%s/caseB", tmpdir);
  struct store* s = store_fs_create(root, 1);
  CHECK(Fail, s);

  {
    struct zarr_group* g = zarr_group_create(s, "");
    CHECK(Fail2, g);
    zarr_group_destroy(g);
  }

  struct dimension dims[2];
  build_dims(dims);
  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .nlod = 2,
  };
  struct ngff_multiscale* ms = ngff_multiscale_create(s, "ms", &cfg);
  CHECK(Fail2, ms);

  // Set an attribute on level 1 (the downsampled child).
  struct zarr_array* lvl1 = ngff_multiscale_level(ms, 1);
  CHECK(Fail3, lvl1);
  CHECK(Fail3, zarr_array_set_attribute(lvl1, "child", "\"l1\"") == 0);

  CHECK(Fail3, run_stream_with_flush(ngff_multiscale_as_shard_sink(ms)) == 0);

  char* out = read_file("caseB/ms/1/zarr.json");
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"child\":\"l1\""));
  CHECK(Fail_out, contains(out, "\"node_type\":\"array\""));

  free(out);
  ngff_multiscale_destroy(ms);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  ngff_multiscale_destroy(ms);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  if (test_tmpdir_create(tmpdir, sizeof(tmpdir)))
    return 1;
  log_info("tmpdir: %s", tmpdir);

  int err = 0;
  err |= test_cascade_to_multiscale_group();
  err |= test_cascade_to_child_array();

  test_tmpdir_remove(tmpdir);
  return err;
}
