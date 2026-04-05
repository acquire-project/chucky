// OME-NGFF v0.5 metadata generation.
#pragma once

#include "dimension.h"
#include "ngff/ngff_axis.h"

#include <stddef.h>
#include <stdint.h>

// Generate OME-NGFF v0.5 multiscale group JSON into buf.
// level_dims[lv] points to the rank-length dimension array for level lv.
// axes may be NULL; if so, all axes default to space/no-unit/scale-1.0.
// Returns JSON length on success, -1 on error.
int
ngff_multiscale_group_json(char* buf,
                           size_t cap,
                           uint8_t rank,
                           int nlod,
                           const struct dimension* const* level_dims,
                           const struct ngff_axis* axes);
