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
                    // NULL defaults to "index" in metadata
  double scale;     // physical pixel scale for coordinateTransformations
                    // (must be non-negative; 0 treated as 1.0)
  enum ngff_axis_type type; // space, time, or channel
};

struct ngff_multiscale_config
{
  enum dtype data_type;
  double fill_value;
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
