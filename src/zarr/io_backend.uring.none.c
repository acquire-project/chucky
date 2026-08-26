// Stands in for the ring where there is none: not Linux, or liburing was not
// found when the build was configured.
#include "zarr/io_backend.uring.h"

int
io_backend_uring_supported(void)
{
  return 0;
}

struct io_backend_uring*
io_backend_uring_create(struct io_backend_fs* files,
                        _Atomic int* io_error,
                        uint64_t depth)
{
  (void)files;
  (void)io_error;
  (void)depth;
  return 0;
}

int
io_backend_uring_start(struct io_backend_uring* b, struct io_queue* queue)
{
  (void)b;
  (void)queue;
  return 1;
}

void
io_backend_uring_destroy(struct io_backend_uring* b)
{
  (void)b;
}

struct io_backend
io_backend_uring_as_backend(struct io_backend_uring* b)
{
  (void)b;
  return (struct io_backend){ 0 };
}
