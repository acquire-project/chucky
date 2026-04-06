#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  enum lod_reduce_method
  {
    lod_reduce_mean,
    lod_reduce_min,
    lod_reduce_max,
    lod_reduce_median,
    lod_reduce_max_suppressed, // 2nd highest value
    lod_reduce_min_suppressed, // 2nd lowest value
  };

#ifdef __cplusplus
}
#endif
