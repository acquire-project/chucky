#include "hcs/hcs.h"
#include "defs.limits.h"
#include "hcs/hcs_metadata.h"
#include "util/prelude.h"
#include "zarr/zarr_group.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct hcs_plate
{
  struct store* store;
  struct shard_pool* pool;
  int rows, cols, field_count;
  int* well_mask; // rows*cols, owned

  // fovs[row * cols * field_count + col * field_count + fov]
  struct ngff_multiscale** fovs;
  int nfovs; // total allocated FOV slots
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

static int
fov_index(const struct hcs_plate* p, int row, int col, int fov)
{
  return (row * p->cols + col) * p->field_count + fov;
}

struct hcs_plate*
hcs_plate_create(struct store* store,
                 struct shard_pool* pool,
                 const struct hcs_plate_config* cfg)
{
  CHECK(Fail, store);
  CHECK(Fail, pool);
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->name);
  CHECK(Fail, cfg->rows > 0);
  CHECK(Fail, cfg->rows <= 26 || cfg->row_names);
  CHECK(Fail, cfg->cols > 0);
  CHECK(Fail, cfg->field_count > 0);

  struct hcs_plate* p = (struct hcs_plate*)calloc(1, sizeof(*p));
  CHECK(Fail, p);

  p->store = store;
  p->pool = pool;
  p->rows = cfg->rows;
  p->cols = cfg->cols;
  p->field_count = cfg->field_count;

  if (cfg->well_mask) {
    size_t mask_size = (size_t)(cfg->rows * cfg->cols) * sizeof(int);
    p->well_mask = (int*)malloc(mask_size);
    CHECK(Fail_alloc, p->well_mask);
    memcpy(p->well_mask, cfg->well_mask, mask_size);
  }

  p->nfovs = cfg->rows * cfg->cols * cfg->field_count;
  p->fovs = (struct ngff_multiscale**)calloc((size_t)p->nfovs, sizeof(void*));
  CHECK(Fail_mask, p->fovs);

  // --- Write hierarchy ---

  // Root group
  CHECK(Fail_fovs, zarr_write_group(store, "zarr.json", NULL) == 0);

  // Plate group with OME plate attributes
  {
    CHECK(Fail_fovs, store->mkdirs(store, cfg->name) == 0);

    // ~50 bytes per well entry; allocate generously
    size_t attr_cap = 512 + (size_t)(cfg->rows * cfg->cols) * 80;
    char* attrs = (char*)malloc(attr_cap);
    CHECK(Fail_fovs, attrs);
    int alen = hcs_plate_attributes_json(attrs,
                                         attr_cap,
                                         cfg->name,
                                         cfg->rows,
                                         cfg->cols,
                                         cfg->row_names,
                                         cfg->field_count,
                                         cfg->well_mask);
    char key[4096];
    snprintf(key, sizeof(key), "%s/zarr.json", cfg->name);
    int rc = alen < 0 ? 1 : zarr_write_group(store, key, attrs);
    free(attrs);
    CHECK(Fail_fovs, rc == 0);
  }

  // Row groups, well groups, and FOV multiscale sinks
  for (int r = 0; r < cfg->rows; ++r) {
    char rc = row_char(cfg, r);

    // Row group
    char row_dir[4096];
    snprintf(row_dir, sizeof(row_dir), "%s/%c", cfg->name, rc);
    CHECK(Fail_fovs, store->mkdirs(store, row_dir) == 0);
    {
      char key[4096];
      snprintf(key, sizeof(key), "%s/%c/zarr.json", cfg->name, rc);
      CHECK(Fail_fovs, zarr_write_group(store, key, NULL) == 0);
    }

    for (int c = 0; c < cfg->cols; ++c) {
      if (!well_active(p, r, c))
        continue;

      // Well group with OME well attributes
      char well_dir[4096];
      snprintf(well_dir, sizeof(well_dir), "%s/%c/%d", cfg->name, rc, c + 1);
      CHECK(Fail_fovs, store->mkdirs(store, well_dir) == 0);
      {
        char attrs[4096];
        int alen =
          hcs_well_attributes_json(attrs, sizeof(attrs), cfg->field_count);
        CHECK(Fail_fovs, alen > 0);
        char key[4096];
        snprintf(key, sizeof(key), "%s/%c/%d/zarr.json", cfg->name, rc, c + 1);
        CHECK(Fail_fovs, zarr_write_group(store, key, attrs) == 0);
      }

      // FOV multiscale sinks
      for (int f = 0; f < cfg->field_count; ++f) {
        char fov_prefix[4096];
        snprintf(fov_prefix,
                 sizeof(fov_prefix),
                 "%s/%c/%d/%d",
                 cfg->name,
                 rc,
                 c + 1,
                 f);

        int idx = fov_index(p, r, c, f);
        p->fovs[idx] =
          ngff_multiscale_create(store, pool, fov_prefix, &cfg->fov);
        CHECK(Fail_fovs, p->fovs[idx]);
      }
    }
  }

  return p;

Fail_fovs:
  for (int i = 0; i < p->nfovs; ++i) {
    if (p->fovs[i])
      ngff_multiscale_destroy(p->fovs[i]);
  }
  free(p->fovs);
Fail_mask:
  free(p->well_mask);
Fail_alloc:
  free(p);
Fail:
  return NULL;
}

void
hcs_plate_destroy(struct hcs_plate* p)
{
  if (!p)
    return;
  for (int i = 0; i < p->nfovs; ++i) {
    if (p->fovs[i])
      ngff_multiscale_destroy(p->fovs[i]);
  }
  free(p->fovs);
  free(p->well_mask);
  free(p);
}

struct shard_sink*
hcs_plate_fov_sink(struct hcs_plate* p, int row, int col, int fov)
{
  CHECK_SILENT(Bad, p);
  CHECK_SILENT(Bad, row >= 0 && row < p->rows);
  CHECK_SILENT(Bad, col >= 0 && col < p->cols);
  CHECK_SILENT(Bad, fov >= 0 && fov < p->field_count);
  CHECK_SILENT(Bad, well_active(p, row, col));

  int idx = fov_index(p, row, col, fov);
  return ngff_multiscale_as_shard_sink(p->fovs[idx]);
Bad:
  return NULL;
}
