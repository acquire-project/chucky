#include "zarr/zarr_group.h"
#include "defs.limits.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/attr_set.h"
#include "zarr/json_writer.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char group_prefix[] =
  "{\"zarr_format\":3,\"node_type\":\"group\","
  "\"consolidated_metadata\":null,\"attributes\":";
static const char group_suffix[] = "}";

int
zarr_group_write_with_raw_attrs(struct store* store,
                                const char* key,
                                const char* attributes_json)
{
  CHECK(Fail, store);
  CHECK(Fail, key);
  CHECK(Fail, attributes_json);

  size_t attr_len = strlen(attributes_json);
  size_t total = sizeof(group_prefix) - 1 + attr_len + sizeof(group_suffix);
  char* buf = (char*)malloc(total);
  CHECK(Fail, buf);

  memcpy(buf, group_prefix, sizeof(group_prefix) - 1);
  memcpy(buf + sizeof(group_prefix) - 1, attributes_json, attr_len);
  memcpy(buf + sizeof(group_prefix) - 1 + attr_len,
         group_suffix,
         sizeof(group_suffix));
  size_t len = total - 1; // exclude null terminator from group_suffix

  int rc = store->put(store, key, buf, len);
  free(buf);
  return rc;

Fail:
  return 1;
}

// --- Handle-based group with buffered attributes ---

struct zarr_group
{
  struct store* store; // borrowed
  char key[4096];
  struct attr_set attrs;
};

static int
zarr_group_write(struct zarr_group* g)
{
  // Group skeleton is ~100 bytes; size for actual attrs to avoid truncation.
  size_t attr_bytes = 0;
  for (size_t i = 0; i < g->attrs.count; ++i) {
    attr_bytes += strlen(g->attrs.items[i].key);
    attr_bytes += strlen(g->attrs.items[i].json_value);
    attr_bytes += 16;
  }
  size_t cap = 256 + attr_bytes;
  char* buf = (char*)malloc(cap);
  if (!buf)
    return 1;

  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "group");
  jw_key(&jw, "consolidated_metadata");
  jw_null(&jw);
  jw_key(&jw, "attributes");
  jw_object_begin(&jw);
  attr_set_emit(&g->attrs, &jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  if (jw_error(&jw)) {
    free(buf);
    return 1;
  }
  int rc = g->store->put(g->store, g->key, buf, jw_length(&jw));
  free(buf);
  if (rc == 0)
    g->attrs.dirty = 0;
  return rc;
}

struct zarr_group*
zarr_group_create(struct store* store, const char* key)
{
  CHECK(Fail, store);
  CHECK(Fail, key);

  struct zarr_group* g = (struct zarr_group*)calloc(1, sizeof(*g));
  CHECK(Fail, g);
  g->store = store;
  attr_set_init(&g->attrs);
  if (key[0])
    snprintf(g->key, sizeof(g->key), "%s/zarr.json", key);
  else
    snprintf(g->key, sizeof(g->key), "zarr.json");

  if (zarr_group_write(g) != 0) {
    attr_set_destroy(&g->attrs);
    free(g);
    return NULL;
  }
  return g;

Fail:
  return NULL;
}

void
zarr_group_destroy(struct zarr_group* g)
{
  if (!g)
    return;
  if (g->attrs.dirty)
    zarr_group_write(g);
  attr_set_destroy(&g->attrs);
  free(g);
}

int
zarr_group_set_attribute(struct zarr_group* g,
                         const char* attr_key,
                         const char* json_value)
{
  CHECK(Fail, g);
  return attr_set_upsert(&g->attrs, attr_key, json_value);
Fail:
  return 1;
}

int
zarr_group_flush_metadata(struct zarr_group* g)
{
  CHECK(Fail, g);
  if (!g->attrs.dirty)
    return 0;
  return zarr_group_write(g);
Fail:
  return 1;
}
