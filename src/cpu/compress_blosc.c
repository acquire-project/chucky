#include "cpu/compress_blosc.h"

#include "threadpool/threadpool.h"

#include <blosc.h>
#include <stdatomic.h>

// Real impl always succeeds for blosc ids. The stub overrides this to
// return 1 with an error, providing build-time unavailability detection.
int
compress_blosc_validate(struct codec_config codec)
{
  if (!codec_is_blosc(codec.id))
    return 1;
  return 0;
}

size_t
compress_blosc_max_output_size(size_t chunk_bytes)
{
  return chunk_bytes + BLOSC_MAX_OVERHEAD;
}

struct blosc_ctx
{
  const char* src;
  size_t input_stride;
  char* dst;
  size_t max_output_size;
  size_t* comp_sizes;
  size_t chunk_bytes;
  size_t typesize;
  const char* compname;
  int clevel;
  int doshuffle;
  _Atomic int err;
};

static void
blosc_one(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct blosc_ctx* c = (struct blosc_ctx*)vctx;
  if (atomic_load_explicit(&c->err, memory_order_relaxed))
    return;
  int rc = blosc_compress_ctx(c->clevel,
                              c->doshuffle,
                              c->typesize,
                              c->chunk_bytes,
                              c->src + i * c->input_stride,
                              c->dst + i * c->max_output_size,
                              c->max_output_size,
                              c->compname,
                              0,  // blocksize (auto)
                              1); // numinternalthreads
  if (rc <= 0)
    atomic_store_explicit(&c->err, 1, memory_order_relaxed);
  else
    c->comp_sizes[i] = (size_t)rc;
}

int
compress_blosc(struct codec_config codec,
               const void* src,
               size_t input_stride,
               void* dst,
               size_t max_output_size,
               size_t* comp_sizes,
               size_t chunk_bytes,
               size_t batch_size,
               size_t bytes_per_element,
               struct threadpool* pool)
{
  struct blosc_ctx c = {
    .src = (const char*)src,
    .input_stride = input_stride,
    .dst = (char*)dst,
    .max_output_size = max_output_size,
    .comp_sizes = comp_sizes,
    .chunk_bytes = chunk_bytes,
    .typesize = bytes_per_element > 0 ? bytes_per_element : 1,
    .compname =
      codec.id == CODEC_BLOSC_LZ4 ? BLOSC_LZ4_COMPNAME : BLOSC_ZSTD_COMPNAME,
    .clevel = codec.level,
    .doshuffle = codec.shuffle,
    .err = 0,
  };
  threadpool_for_n_dynamic(pool, batch_size, blosc_one, &c);
  return atomic_load_explicit(&c.err, memory_order_acquire);
}
