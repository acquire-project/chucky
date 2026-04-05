#include "zarr/zarr_group.h"
#include "defs.limits.h"
#include "util/prelude.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <string.h>

int
zarr_write_group(struct store* store,
                 const char* key,
                 const char* attributes_json)
{
  CHECK(Fail, store);
  CHECK(Fail, key);

  if (!attributes_json) {
    // Plain group — use zarr_root_json which writes empty attributes
    char buf[ZARR_GROUP_JSON_MAX_LENGTH];
    int len = zarr_root_json(buf, sizeof(buf));
    CHECK(Fail, len >= 0);
    return store->put(store, key, buf, (size_t)len);
  }

  // Group with custom attributes — build JSON manually.
  // Format: {"zarr_format":3,"node_type":"group",
  //          "consolidated_metadata":null,"attributes":<json>}
  char buf[ZARR_GROUP_JSON_MAX_LENGTH];
  int len = snprintf(buf,
                     sizeof(buf),
                     "{\"zarr_format\":3,\"node_type\":\"group\","
                     "\"consolidated_metadata\":null,\"attributes\":%s}",
                     attributes_json);
  CHECK(Fail, len > 0 && (size_t)len < sizeof(buf));
  return store->put(store, key, buf, (size_t)len);

Fail:
  return 1;
}
