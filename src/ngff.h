// Public OME-NGFF v0.5 multiscale interface.
// Creates a multiscale array hierarchy with per-level zarr arrays and
// OME-NGFF group metadata.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "store.h"
#include "types.codec.h"
#include "writer.h"

#include <stdint.h>

// OME-NGFF v0.5 axis types. Only "space", "time", and "channel" are valid.
enum ngff_axis_type
{
  ngff_axis_space = 0, // default (zero-init = space)
  ngff_axis_time,
  ngff_axis_channel,
};

struct ngff_axis
{
  const char* unit; // axis unit (e.g. "micrometer"),
                    // NULL defaults to "index" in metadata.
                    // Copied at ngff_multiscale_create; caller may free after.
  double scale;     // physical pixel scale for coordinateTransformations
                    // (must be non-negative; 0 treated as 1.0)
  enum ngff_axis_type type; // space, time, or channel
};

// Deep-copy rank axes from src to dst, duplicating unit strings.
// After success, dst owns its unit strings; free with ngff_axes_free_units.
// Returns 0 on success. On failure, frees any partial unit allocations and
// leaves dst zero-initialized for the affected slots.
int
ngff_axes_copy(struct ngff_axis* dst,
               const struct ngff_axis* src,
               uint8_t rank);

// Free unit strings previously allocated by ngff_axes_copy. Safe on
// zero-initialized slots (free(NULL)). Does not free the array itself.
// Only the unit pointer is owned; scale and type are plain scalars and are
// left untouched.
void
ngff_axes_free_units(struct ngff_axis* axes, uint8_t rank);

struct ngff_multiscale_config
{
  enum dtype data_type;
  uint8_t rank;
  const struct dimension* dimensions; // L0 dimensions
  int nlod;                           // 0 = auto
  struct codec_config codec;
  const struct ngff_axis* axes; // rank elements, NULL = defaults
};

struct ngff_multiscale;

// Create a multiscale sink.
// Writes group metadata and per-level array metadata via the store.
// Returns NULL on error.
struct ngff_multiscale*
ngff_multiscale_create(struct store* store,
                       const char* prefix,
                       const struct ngff_multiscale_config* cfg);

// Auto-flushes pending metadata best-effort; ignores flush errors. Call
// ngff_multiscale_flush_metadata before destroy if you need to detect failures.
void
ngff_multiscale_destroy(struct ngff_multiscale* ms);

struct shard_sink*
ngff_multiscale_as_shard_sink(struct ngff_multiscale* ms);

// Flush all pending I/O. Returns non-zero on error.
int
ngff_multiscale_flush(struct ngff_multiscale* ms);

// Returns non-zero if any I/O has failed.
int
ngff_multiscale_has_error(const struct ngff_multiscale* ms);

// Returns number of bytes queued but not yet written.
size_t
ngff_multiscale_pending_bytes(const struct ngff_multiscale* ms);

// Buffer a custom attribute on the multiscale's OME group (sibling of the
// "ome" block). Validated and copied; rewritten on next metadata write or
// destroy. Returns 0 on success.
int
ngff_multiscale_set_attribute(struct ngff_multiscale* ms,
                              const char* attr_key,
                              const char* json_value);

// Force the multiscale's group zarr.json to be rewritten now.
int
ngff_multiscale_flush_metadata(struct ngff_multiscale* ms);
