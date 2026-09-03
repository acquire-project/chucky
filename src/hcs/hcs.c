#include "hcs.h"
#include "defs.limits.h"
#include "hcs/hcs_metadata.h"
#include "ngff/ngff_multiscale.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr.h"
#include "zarr/attr_set.h"
#include "zarr/store.h"
#include "zarr/zarr_group.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct hcs_plate
{
  struct store* store;
  struct shard_pool* pool; // owned
  int rows, cols, field_count;
  int* well_mask;     // rows*cols, owned
  struct strbuf name; // owned
  char row_names[64];
  int has_row_names;

  // fovs[row * cols * field_count + col * field_count + fov]
  struct ngff_multiscale** fovs;
  uint64_t nfovs; // total allocated FOV slots

  // Plate-level custom attrs.
  struct attr_set plate_attrs;
  // Well-level custom attrs (rows*cols entries; inactive wells empty).
  struct attr_set* well_attrs;
};

static int
well_active(const struct hcs_plate* p, int r, int c)
{
  return !p->well_mask || p->well_mask[r * p->cols + c];
}

static char
row_char(const struct hcs_plate_config* cfg, int r)
{
  return cfg->row_names ? cfg->row_names[r] : (char)('A' + r);
}

static uint64_t
fov_index(const struct hcs_plate* p, int row, int col, int fov)
{
  return ((uint64_t)row * p->cols + col) * p->field_count + fov;
}

static int
well_idx(const struct hcs_plate* p, int row, int col)
{
  return row * p->cols + col;
}

static char
plate_row_char(const struct hcs_plate* p, int r)
{
  return p->has_row_names ? p->row_names[r] : (char)('A' + r);
}

static const char*
plate_row_names_ptr(const struct hcs_plate* p)
{
  return p->has_row_names ? p->row_names : NULL;
}

// Build a printf-formatted path into a scratch strbuf, mkdir it, and free.
// Returns 0 on success.
static int
mkdirs_fmt(struct store* store, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 2, 3)))
#endif
  ;
static int
mkdirs_fmt(struct store* store, const char* fmt, ...)
{
  struct strbuf path = { 0 };
  va_list ap;
  va_start(ap, fmt);
  int rc = strbuf_vappendf(&path, fmt, ap);
  va_end(ap);
  if (rc == 0)
    rc = store->mkdirs(store, strbuf_cstr(&path));
  strbuf_free(&path);
  return rc;
}

static int
write_plate_group(struct hcs_plate* p)
{
  struct strbuf attrs = { 0 };
  struct strbuf key = { 0 };
  int rc = 1;

  if (hcs_plate_attributes_json(&attrs,
                                strbuf_cstr(&p->name),
                                p->rows,
                                p->cols,
                                plate_row_names_ptr(p),
                                p->field_count,
                                p->well_mask,
                                &p->plate_attrs))
    goto done;
  if (strbuf_appendf(&key, "%s/zarr.json", strbuf_cstr(&p->name)))
    goto done;
  rc = zarr_group_write_with_raw_attrs(
    p->store, strbuf_cstr(&key), strbuf_cstr(&attrs));
  if (rc == 0)
    p->plate_attrs.dirty = 0;

done:
  strbuf_free(&attrs);
  strbuf_free(&key);
  return rc;
}

static int
write_well_group(struct hcs_plate* p, int r, int c)
{
  struct attr_set* w = &p->well_attrs[well_idx(p, r, c)];
  struct strbuf attrs = { 0 };
  struct strbuf key = { 0 };
  int rc = 1;

  if (hcs_well_attributes_json(&attrs, p->field_count, w))
    goto done;
  char rc_ch = plate_row_char(p, r);
  if (strbuf_appendf(
        &key, "%s/%c/%d/zarr.json", strbuf_cstr(&p->name), rc_ch, c + 1))
    goto done;
  rc = zarr_group_write_with_raw_attrs(
    p->store, strbuf_cstr(&key), strbuf_cstr(&attrs));
  if (rc == 0)
    w->dirty = 0;

done:
  strbuf_free(&attrs);
  strbuf_free(&key);
  return rc;
}

struct hcs_plate*
hcs_plate_create(struct store* store, const struct hcs_plate_config* cfg)
{
  CHECK(Fail, store);
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->name);
  CHECK(Fail, cfg->rows > 0);
  CHECK(Fail, cfg->rows <= 26 || cfg->row_names);
  CHECK(Fail, cfg->cols > 0);
  CHECK(Fail, cfg->field_count > 0);
  CHECK(Fail, cfg->fov.dimensions);
  CHECK(Fail, cfg->fov.rank > 0 && cfg->fov.rank <= MAX_ZARR_RANK);
  // Reserve room for "/<row>/<col>/<fov>/zarr.json" when building child keys.
  CHECK(Fail, strlen(cfg->name) < 4000);
  if (cfg->row_names) {
    size_t rn_len = strlen(cfg->row_names);
    CHECK(Fail, rn_len >= (size_t)cfg->rows);
    CHECK(Fail, rn_len < 64); // bounded by hcs_plate.row_names buffer
  }

  // One pool per plate keeps its fields of view on one io queue.
  uint64_t slots_per_fov = ngff_multiscale_slot_count(&cfg->fov);
  CHECK(Fail, slots_per_fov > 0);

  uint64_t fov_count = (uint64_t)cfg->rows * cfg->cols * cfg->field_count;
  struct shard_pool* pool =
    store->create_pool(store, fov_count * slots_per_fov);
  CHECK(Fail, pool);

  struct hcs_plate* p = (struct hcs_plate*)calloc(1, sizeof(*p));
  CHECK(Fail_pool, p);

  p->store = store;
  p->pool = pool;
  p->rows = cfg->rows;
  p->cols = cfg->cols;
  p->field_count = cfg->field_count;
  CHECK(Fail_alloc, strbuf_set(&p->name, cfg->name) == 0);
  if (cfg->row_names) {
    p->has_row_names = 1;
    snprintf(p->row_names, sizeof(p->row_names), "%s", cfg->row_names);
  }
  attr_set_init(&p->plate_attrs);

  if (cfg->well_mask) {
    size_t mask_size = (size_t)(cfg->rows * cfg->cols) * sizeof(int);
    p->well_mask = (int*)malloc(mask_size);
    CHECK(Fail_alloc, p->well_mask);
    memcpy(p->well_mask, cfg->well_mask, mask_size);
  }

  p->well_attrs = (struct attr_set*)calloc((size_t)(cfg->rows * cfg->cols),
                                           sizeof(struct attr_set));
  CHECK(Fail_mask, p->well_attrs);
  for (int i = 0; i < cfg->rows * cfg->cols; ++i)
    attr_set_init(&p->well_attrs[i]);

  p->nfovs = fov_count;
  p->fovs = (struct ngff_multiscale**)calloc((size_t)p->nfovs, sizeof(void*));
  CHECK(Fail_well_attrs, p->fovs);

  // --- Write hierarchy ---

  // Root group
  {
    struct zarr_group* g = zarr_group_create(store, "");
    CHECK(Fail_fovs, g);
    zarr_group_destroy(g);
  }

  // Plate group with OME plate attributes
  CHECK(Fail_fovs, store->mkdirs(store, cfg->name) == 0);
  CHECK(Fail_fovs, write_plate_group(p) == 0);

  // Row groups, well groups, and FOV multiscale sinks
  for (int r = 0; r < cfg->rows; ++r) {
    char rc = row_char(cfg, r);

    // Row group
    struct strbuf row_dir = { 0 };
    int row_rc = strbuf_appendf(&row_dir, "%s/%c", cfg->name, rc);
    if (row_rc == 0)
      row_rc = store->mkdirs(store, strbuf_cstr(&row_dir));
    struct zarr_group* g =
      row_rc == 0 ? zarr_group_create(store, strbuf_cstr(&row_dir)) : NULL;
    strbuf_free(&row_dir);
    CHECK(Fail_fovs, row_rc == 0 && g);
    zarr_group_destroy(g);

    for (int c = 0; c < cfg->cols; ++c) {
      if (!well_active(p, r, c))
        continue;

      // Well group with OME well attributes
      CHECK(Fail_fovs,
            mkdirs_fmt(store, "%s/%c/%d", cfg->name, rc, c + 1) == 0);
      CHECK(Fail_fovs, write_well_group(p, r, c) == 0);

      // FOV multiscale sinks
      for (int f = 0; f < cfg->field_count; ++f) {
        struct strbuf fov_prefix = { 0 };
        uint64_t idx = fov_index(p, r, c, f);
        if (strbuf_appendf(
              &fov_prefix, "%s/%c/%d/%d", cfg->name, rc, c + 1, f) == 0)
          p->fovs[idx] =
            ngff_multiscale_create_with_pool(store,
                                             pool,
                                             idx * slots_per_fov,
                                             strbuf_cstr(&fov_prefix),
                                             &cfg->fov);
        strbuf_free(&fov_prefix);
        CHECK(Fail_fovs, p->fovs[idx]);
      }
    }
  }

  return p;

Fail_fovs:
  for (uint64_t i = 0; i < p->nfovs; ++i) {
    if (p->fovs[i])
      ngff_multiscale_destroy(p->fovs[i]);
  }
  free(p->fovs);
Fail_well_attrs:
  for (int i = 0; i < cfg->rows * cfg->cols; ++i)
    attr_set_destroy(&p->well_attrs[i]);
  free(p->well_attrs);
Fail_mask:
  free(p->well_mask);
Fail_alloc:
  attr_set_destroy(&p->plate_attrs);
  strbuf_free(&p->name);
  free(p);
Fail_pool:
  shard_pool_destroy(pool);
Fail:
  return NULL;
}

void
hcs_plate_destroy(struct hcs_plate* p)
{
  if (!p)
    return;
  // Flush dirty plate/well group metadata before teardown.
  if (p->plate_attrs.dirty)
    write_plate_group(p);
  for (int r = 0; r < p->rows; ++r)
    for (int c = 0; c < p->cols; ++c)
      if (well_active(p, r, c) && p->well_attrs[well_idx(p, r, c)].dirty)
        write_well_group(p, r, c);

  // FOV multiscales flush their own metadata in their destroy.
  for (uint64_t i = 0; i < p->nfovs; ++i) {
    if (p->fovs[i])
      ngff_multiscale_destroy(p->fovs[i]);
  }
  free(p->fovs);
  for (int i = 0; i < p->rows * p->cols; ++i)
    attr_set_destroy(&p->well_attrs[i]);
  free(p->well_attrs);
  free(p->well_mask);
  attr_set_destroy(&p->plate_attrs);
  strbuf_free(&p->name);
  struct shard_pool* pool = p->pool;
  free(p);
  shard_pool_destroy(pool);
}

struct shard_sink*
hcs_plate_fov_sink(struct hcs_plate* p, int row, int col, int fov)
{
  CHECK_SILENT(Bad, p);
  CHECK_SILENT(Bad, row >= 0 && row < p->rows);
  CHECK_SILENT(Bad, col >= 0 && col < p->cols);
  CHECK_SILENT(Bad, fov >= 0 && fov < p->field_count);
  CHECK_SILENT(Bad, well_active(p, row, col));

  uint64_t idx = fov_index(p, row, col, fov);
  return ngff_multiscale_as_shard_sink(p->fovs[idx]);
Bad:
  return NULL;
}

int
hcs_plate_set_attribute(struct hcs_plate* p,
                        const char* attr_key,
                        const char* json_value)
{
  CHECK(Fail, p);
  return attr_set_upsert(&p->plate_attrs, attr_key, json_value);
Fail:
  return 1;
}

int
hcs_plate_set_well_attribute(struct hcs_plate* p,
                             int row,
                             int col,
                             const char* attr_key,
                             const char* json_value)
{
  CHECK(Fail, p);
  CHECK(Fail, row >= 0 && row < p->rows);
  CHECK(Fail, col >= 0 && col < p->cols);
  CHECK(Fail, well_active(p, row, col));
  return attr_set_upsert(
    &p->well_attrs[well_idx(p, row, col)], attr_key, json_value);
Fail:
  return 1;
}

int
hcs_plate_set_fov_attribute(struct hcs_plate* p,
                            int row,
                            int col,
                            int fov,
                            const char* attr_key,
                            const char* json_value)
{
  CHECK(Fail, p);
  CHECK(Fail, row >= 0 && row < p->rows);
  CHECK(Fail, col >= 0 && col < p->cols);
  CHECK(Fail, fov >= 0 && fov < p->field_count);
  CHECK(Fail, well_active(p, row, col));
  return ngff_multiscale_set_attribute(
    p->fovs[fov_index(p, row, col, fov)], attr_key, json_value);
Fail:
  return 1;
}

int
hcs_plate_flush_metadata(struct hcs_plate* p)
{
  CHECK(Fail, p);
  int err = 0;
  if (p->plate_attrs.dirty)
    err |= write_plate_group(p);
  for (int r = 0; r < p->rows; ++r)
    for (int c = 0; c < p->cols; ++c) {
      if (!well_active(p, r, c))
        continue;
      if (p->well_attrs[well_idx(p, r, c)].dirty)
        err |= write_well_group(p, r, c);
      for (int f = 0; f < p->field_count; ++f) {
        struct ngff_multiscale* ms = p->fovs[fov_index(p, r, c, f)];
        if (ms)
          err |= ngff_multiscale_flush_metadata(ms);
      }
    }
  return err;
Fail:
  return 1;
}
