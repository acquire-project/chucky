#include "util/prelude.h"
#include "zarr.h"
#include "zarr/json_merge.h"
#include "zarr/store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
zarr_write_attribute(struct store* store,
                     const char* key_path,
                     const char* attr_key,
                     const char* json_value)
{
  CHECK(Fail, store);
  CHECK(Fail, store->get);
  CHECK(Fail, store->put);
  CHECK(Fail, attr_key);
  CHECK(Fail, attr_key[0]);
  CHECK(Fail, json_value);

  // "ome" is reserved for OME-NGFF metadata.
  if (strcmp(attr_key, "ome") == 0) {
    log_error("zarr_write_attribute: 'ome' is reserved");
    return 1;
  }

  // Build the zarr.json key.
  char key[4096];
  if (!key_path || !key_path[0])
    snprintf(key, sizeof(key), "zarr.json");
  else
    snprintf(key, sizeof(key), "%s/zarr.json", key_path);

  void* raw = NULL;
  size_t raw_len = 0;
  if (store->get(store, key, &raw, &raw_len) != 0) {
    log_error("zarr_write_attribute: failed to read %s", key);
    return 1;
  }

  char* out = NULL;
  size_t out_len = 0;
  int rc = json_merge_attribute(
    (const char*)raw, raw_len, attr_key, json_value, &out, &out_len);
  free(raw);
  if (rc != 0) {
    log_error("zarr_write_attribute: merge failed for %s", key);
    return 1;
  }

  rc = store->put(store, key, out, out_len);
  free(out);
  if (rc != 0) {
    log_error("zarr_write_attribute: failed to write %s", key);
    return 1;
  }
  return 0;

Fail:
  return 1;
}
