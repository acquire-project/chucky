#include "hcs/hcs_metadata.h"
#include "zarr/attr_set.h"
#include "zarr/json_writer.h"

#include <stdio.h>

int
hcs_plate_attributes_json(char* buf,
                          size_t cap,
                          const char* plate_name,
                          int rows,
                          int cols,
                          const char* row_names,
                          int field_count,
                          const int* well_mask,
                          const struct attr_set* extras)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw); // attributes root

  jw_key(&jw, "ome");
  jw_object_begin(&jw);
  jw_key(&jw, "version");
  jw_string(&jw, "0.5");

  jw_key(&jw, "plate");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, plate_name ? plate_name : "plate");
  jw_key(&jw, "field_count");
  jw_int(&jw, field_count);

  // acquisitions: single default acquisition
  jw_key(&jw, "acquisitions");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "id");
  jw_int(&jw, 0);
  jw_object_end(&jw);
  jw_array_end(&jw);

  // columns
  jw_key(&jw, "columns");
  jw_array_begin(&jw);
  for (int c = 0; c < cols; ++c) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    char name[16];
    snprintf(name, sizeof(name), "%d", c + 1);
    jw_string(&jw, name);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  // rows
  jw_key(&jw, "rows");
  jw_array_begin(&jw);
  for (int r = 0; r < rows; ++r) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    char name[2];
    name[0] = row_names ? row_names[r] : (char)('A' + r);
    name[1] = '\0';
    jw_string(&jw, name);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  // wells
  jw_key(&jw, "wells");
  jw_array_begin(&jw);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (well_mask && !well_mask[r * cols + c])
        continue;
      jw_object_begin(&jw);
      jw_key(&jw, "path");
      char path[32];
      char row_ch = row_names ? row_names[r] : (char)('A' + r);
      snprintf(path, sizeof(path), "%c/%d", row_ch, c + 1);
      jw_string(&jw, path);
      jw_key(&jw, "rowIndex");
      jw_int(&jw, r);
      jw_key(&jw, "columnIndex");
      jw_int(&jw, c);
      jw_object_end(&jw);
    }
  }
  jw_array_end(&jw);

  jw_object_end(&jw); // plate
  jw_object_end(&jw); // ome
  attr_set_emit(extras, &jw);
  jw_object_end(&jw); // attributes root

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}

int
hcs_well_attributes_json(char* buf,
                         size_t cap,
                         int field_count,
                         const struct attr_set* extras)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw); // attributes root

  jw_key(&jw, "ome");
  jw_object_begin(&jw);
  jw_key(&jw, "version");
  jw_string(&jw, "0.5");

  jw_key(&jw, "well");
  jw_object_begin(&jw);
  jw_key(&jw, "images");
  jw_array_begin(&jw);
  for (int f = 0; f < field_count; ++f) {
    jw_object_begin(&jw);
    jw_key(&jw, "path");
    char path[8];
    snprintf(path, sizeof(path), "%d", f);
    jw_string(&jw, path);
    jw_key(&jw, "acquisition");
    jw_int(&jw, 0);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_object_end(&jw); // well
  jw_object_end(&jw); // ome
  attr_set_emit(extras, &jw);
  jw_object_end(&jw); // attributes root

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}
