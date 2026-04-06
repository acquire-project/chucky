// OME-NGFF v0.5 axis metadata: unit, scale, and type.
// Only consumed by NGFF metadata writers.
#pragma once

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
