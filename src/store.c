#include "store.h"
#include "zarr/store.h"

void
store_destroy(struct store* s)
{
  if (s)
    s->destroy(s);
}
