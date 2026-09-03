#pragma once

#include "zarr/io_request.h"

struct io_backend
{
  void* ctx;
  void (*execute)(void* ctx, const struct io_request* req);
};
