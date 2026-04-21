// OME-NGFF v0.5 metadata generation.
#pragma once

#include "dimension.h"
#include "util/strbuf.h"

#include <stddef.h>
#include <stdint.h>

struct ngff_axis;
struct attr_set;

// Append OME-NGFF v0.5 multiscale group JSON to sb.
// level_dims[lv] points to the rank-length dimension array for level lv.
// axes may be NULL; if so, all axes default to space/no-unit/scale-1.0.
// extras: optional custom attrs written alongside the OME block. May be NULL.
// Returns 0 on success.
int
ngff_multiscale_group_json(struct strbuf* sb,
                           uint8_t rank,
                           int nlod,
                           const struct dimension* const* level_dims,
                           const struct ngff_axis* axes,
                           const struct attr_set* extras);
