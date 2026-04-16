#include "store.h"
#include "zarr/store.h"

int
store_has_existing_data(struct store* s)
{
  if (!s)
    return -1;
  return s->has_existing_data(s);
}

void
store_destroy(struct store* s)
{
  if (s)
    s->destroy(s);
}
