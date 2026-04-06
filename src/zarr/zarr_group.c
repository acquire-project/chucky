#include "zarr/zarr_group.h"
#include "defs.limits.h"
#include "util/prelude.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

static const char group_prefix[] =
  "{\"zarr_format\":3,\"node_type\":\"group\","
  "\"consolidated_metadata\":null,\"attributes\":";
static const char group_suffix[] = "}";

int
zarr_write_group(struct store* store,
                 const char* key,
                 const char* attributes_json)
{
  CHECK(Fail, store);
  CHECK(Fail, key);

  if (!attributes_json) {
    char buf[ZARR_GROUP_JSON_MAX_LENGTH];
    int len = zarr_root_json(buf, sizeof(buf));
    CHECK(Fail, len >= 0);
    return store->put(store, key, buf, (size_t)len);
  }

  // Dynamically allocate for large attributes (e.g. HCS plate metadata).
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
