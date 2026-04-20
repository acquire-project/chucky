#pragma once

#include "writer.h"

enum multiarray_writer_error
{
  multiarray_writer_ok = 0,
  multiarray_writer_fail = 1,
  multiarray_writer_finished = 2,
  multiarray_writer_not_flushable = 3,
};

struct multiarray_writer_result
{
  int error;
  struct slice rest;
};

struct multiarray_writer
{
  struct multiarray_writer_result (*update)(struct multiarray_writer* self,
                                            int array_index,
                                            struct slice data);
  struct multiarray_writer_result (*flush)(struct multiarray_writer* self);
};
