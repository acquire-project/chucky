// Public zarr v3 array and group interface.
// Create arrays that implement the shard_sink interface for streaming writes.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "store.h"
#include "types.codec.h"
#include "writer.h"

#include <stdint.h>

struct zarr_array_config
{
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  struct codec_config codec;
};

struct zarr_array;

// Create a zarr v3 array.
// Writes {prefix}/zarr.json. Does NOT write root or intermediate groups.
// The caller must ensure the prefix directory exists (via zarr_group_create
// or the higher-level ngff/hcs layers).
// prefix may be "" to write at the store root.
// Returns NULL on error.
struct zarr_array*
zarr_array_create(struct store* store,
                  const char* prefix,
                  const struct zarr_array_config* cfg);

// Auto-flushes pending metadata best-effort; ignores flush errors. Call
// zarr_array_flush_metadata before destroy if you need to detect failures.
void
zarr_array_destroy(struct zarr_array* a);

struct shard_sink*
zarr_array_as_shard_sink(struct zarr_array* a);

int
zarr_array_flush(struct zarr_array* a);

int
zarr_array_has_error(const struct zarr_array* a);

// Returns number of bytes queued but not yet written.
size_t
zarr_array_pending_bytes(const struct zarr_array* a);

// Access live dimensions (reflects append-dimension updates).
const struct dimension*
zarr_array_dimensions(const struct zarr_array* a);

// Attach a custom JSON attribute to the array's zarr.json under
// attributes.<attr_key>. Value is validated and copied. attr_key must be
// non-empty and contain no quotes or control chars. Becomes visible on the
// next metadata rewrite (shape advance, explicit flush, or destroy).
// Replaces any prior value for the same key. Returns 0 on success.
int
zarr_array_set_attribute(struct zarr_array* a,
                         const char* attr_key,
                         const char* json_value);

// Force the array's zarr.json to be rewritten now with current shape and
// buffered attributes.
int
zarr_array_flush_metadata(struct zarr_array* a);

// --- Group handle ---

struct zarr_group;

// Create a zarr v3 group handle. Writes an initial zarr.json (empty
// attributes) at the given key. key may be "" to write at store root.
// Returns NULL on error.
struct zarr_group*
zarr_group_create(struct store* store, const char* key);

// Destroy a group handle. Auto-flushes pending metadata best-effort; ignores
// flush errors. Call zarr_group_flush_metadata before destroy if you need to
// detect failures.
void
zarr_group_destroy(struct zarr_group* g);

// Buffer a custom attribute on the group. Rewrite on flush or destroy.
int
zarr_group_set_attribute(struct zarr_group* g,
                         const char* attr_key,
                         const char* json_value);

// Force the group's zarr.json to be rewritten now.
int
zarr_group_flush_metadata(struct zarr_group* g);
