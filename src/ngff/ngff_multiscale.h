// Storage-agnostic OME-NGFF multiscale format layer.
// Composes N zarr_array children (one per LOD level) and writes
// OME-NGFF v0.5 group metadata with multiscales attributes.
#pragma once

#include "dtype.h"
#include "ngff/ngff_axis.h"
#include "types.codec.h"
#include "writer.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"
#include "zarr/zarr_array.h"

#include <stdint.h>

struct dimension;

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
// Writes root group, intermediate groups, NGFF group metadata,
// and per-level array metadata. All via store.
// Returns NULL on error.
struct ngff_multiscale*
ngff_multiscale_create(struct store* store,
                       struct shard_pool* pool,
                       const char* prefix,
                       const struct ngff_multiscale_config* cfg);

void
ngff_multiscale_destroy(struct ngff_multiscale* ms);

struct shard_sink*
ngff_multiscale_as_shard_sink(struct ngff_multiscale* ms);
