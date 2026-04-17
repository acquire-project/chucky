#include "ngff/ngff_metadata.h"
#include "defs.limits.h"
#include "ngff.h"
#include "util/ceildiv.h"
#include "zarr/attr_set.h"
#include "zarr/json_writer.h"

#include <stdio.h>

int
ngff_multiscale_group_json(char* buf,
                           size_t cap,
                           uint8_t rank,
                           int nlod,
                           const struct dimension* const* level_dims,
                           const struct ngff_axis* axes,
                           const struct attr_set* extras)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  const struct dimension* l0 = level_dims[0];

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "group");

  jw_key(&jw, "consolidated_metadata");
  jw_null(&jw);

  jw_key(&jw, "attributes");
  jw_object_begin(&jw);

  jw_key(&jw, "ome");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "/");
  jw_key(&jw, "version");
  jw_string(&jw, "0.5");

  jw_key(&jw, "multiscales");
  jw_array_begin(&jw);
  jw_object_begin(&jw);

  jw_key(&jw, "axes");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    if (l0[d].name)
      jw_string(&jw, l0[d].name);
    else {
      char name[8];
      snprintf(name, sizeof(name), "d%d", d);
      jw_string(&jw, name);
    }
    jw_key(&jw, "type");
    {
      const char* type = "space";
      if (axes) {
        switch (axes[d].type) {
          case ngff_axis_time:
            type = "time";
            break;
          case ngff_axis_channel:
            type = "channel";
            break;
          default:
            type = "space";
            break;
        }
      }
      jw_string(&jw, type);
    }
    if (axes && axes[d].unit) {
      jw_key(&jw, "unit");
      jw_string(&jw, axes[d].unit);
    }
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "datasets");
  jw_array_begin(&jw);
  for (int lv = 0; lv < nlod; ++lv) {
    jw_object_begin(&jw);
    jw_key(&jw, "path");
    char lvstr[8];
    snprintf(lvstr, sizeof(lvstr), "%d", lv);
    jw_string(&jw, lvstr);

    double scale[MAX_ZARR_RANK];
    for (int d = 0; d < rank; ++d) {
      double phys = (axes && axes[d].scale > 0) ? axes[d].scale : 1.0;
      int n_down = 0;
      if (l0[d].downsample) {
        for (int k = 0; k < lv; ++k) {
          uint64_t sz = level_dims[k][d].size;
          uint64_t cs = level_dims[k][d].chunk_size;
          if (cs > 0 && sz > 0 && ceildiv(sz, cs) > 1)
            ++n_down;
        }
      }
      scale[d] = phys * (double)(1 << n_down);
    }

    jw_key(&jw, "coordinateTransformations");
    jw_array_begin(&jw);
    // scale
    jw_object_begin(&jw);
    jw_key(&jw, "type");
    jw_string(&jw, "scale");
    jw_key(&jw, "scale");
    jw_array_begin(&jw);
    for (int d = 0; d < rank; ++d)
      jw_float(&jw, scale[d]);
    jw_array_end(&jw);
    jw_object_end(&jw);
    jw_array_end(&jw);

    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "type");
  jw_string(&jw, "decimate");

  jw_key(&jw, "metadata");
  jw_object_begin(&jw);
  jw_key(&jw, "method");
  jw_string(&jw, "np.ndarray.__getitem__");
  jw_key(&jw, "version");
  jw_string(&jw, "2.2.6");
  jw_key(&jw, "description");
  jw_string(
    &jw,
    "Subsampling by taking every 2nd pixel/voxel (top-left corner of each "
    "2x2 block). Equivalent to numpy array slicing with stride 2.");
  jw_key(&jw, "args");
  jw_array_begin(&jw);
  jw_string(&jw, "(slice(0, None, 2), slice(0, None, 2))");
  jw_array_end(&jw);
  jw_object_end(&jw);

  jw_object_end(&jw); // multiscales[0]
  jw_array_end(&jw);  // multiscales

  jw_object_end(&jw); // ome
  attr_set_emit(extras, &jw);
  jw_object_end(&jw); // attributes
  jw_object_end(&jw); // root

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}
