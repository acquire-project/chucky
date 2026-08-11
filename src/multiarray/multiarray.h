#pragma once

#include "writer.h"

enum multiarray_writer_error
{
  multiarray_writer_ok = 0,
  multiarray_writer_fail = 1,
  multiarray_writer_finished = 2,
  multiarray_writer_not_flushable = 3,
};

// update_impl pass-through assigns writer_result.error directly into a
// multiarray_writer_result.error field; the enums must agree on shared codes.
_Static_assert((int)multiarray_writer_ok == (int)writer_error_ok,
               "writer/multiarray ok codes must match");
_Static_assert((int)multiarray_writer_fail == (int)writer_error_fail,
               "writer/multiarray fail codes must match");
_Static_assert((int)multiarray_writer_finished == (int)writer_error_finished,
               "writer/multiarray finished codes must match");

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
  // Finalizes every array: writes out what each one holds and stops taking
  // input. Returns once the writes are queued; `close` waits for them.
  struct multiarray_writer_result (*flush)(struct multiarray_writer* self);
  // Waits for those writes to land and publishes each array's append extent.
  // Every array has queued its work by then, so the waits overlap instead of
  // draining one array at a time. Idempotent; destroy runs it if the caller did
  // not.
  struct multiarray_writer_result (*close)(struct multiarray_writer* self);
};
