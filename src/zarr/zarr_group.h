// Private group helpers used by ngff/hcs layers.
#pragma once

#include "zarr/store.h"

// Submit a group at prefix using a prevalidated attributes JSON object.
// A queue-capable pool copies the snapshot; call zarr_metadata_wait for
// synchronous visibility. NULL/non-queue pools write synchronously via store.
// Returns 0 when accepted, non-zero on serialization/submission failure.
int
zarr_group_submit(struct store* store,
                  struct shard_pool* pool,
                  const char* prefix,
                  const char* attributes_json);
