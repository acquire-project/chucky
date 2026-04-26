#include "cpu/compress.h"
#include "cpu/compress_blosc.h"

#include "threadpool/threadpool.h"
#include "util/prelude.h"

#include <lz4hc.h>
#include <stdatomic.h>
#include <string.h>
#include <zstd.h>

size_t
compress_cpu_max_output_size(enum compression_codec type, size_t chunk_bytes)
{
  switch (type) {
    case CODEC_NONE:
      return chunk_bytes;
    case CODEC_LZ4_NON_STANDARD:
      return (size_t)LZ4_compressBound((int)chunk_bytes);
    case CODEC_ZSTD:
      return ZSTD_compressBound(chunk_bytes);
    case CODEC_BLOSC_LZ4:
    case CODEC_BLOSC_ZSTD:
      return compress_blosc_max_output_size(chunk_bytes);
    default:
      return 0;
  }
}

struct copy_ctx
{
  const char* src;
  size_t input_stride;
  char* dst;
  size_t max_output_size;
  size_t* comp_sizes;
  size_t chunk_bytes;
};

static void
copy_range(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct copy_ctx* c = (struct copy_ctx*)vctx;
  for (size_t i = beg; i < end; ++i) {
    memcpy(c->dst + i * c->max_output_size,
           c->src + i * c->input_stride,
           c->chunk_bytes);
    c->comp_sizes[i] = c->chunk_bytes;
  }
}

struct lz4_ctx
{
  const char* src;
  size_t input_stride;
  char* dst;
  size_t max_output_size;
  size_t* comp_sizes;
  size_t chunk_bytes;
  int level;
  _Atomic int err;
};

static void
lz4_one(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct lz4_ctx* c = (struct lz4_ctx*)vctx;
  if (atomic_load_explicit(&c->err, memory_order_relaxed))
    return;
  int rc = LZ4_compress_HC(c->src + i * c->input_stride,
                           c->dst + i * c->max_output_size,
                           (int)c->chunk_bytes,
                           (int)c->max_output_size,
                           c->level);
  if (rc <= 0)
    atomic_store_explicit(&c->err, 1, memory_order_relaxed);
  else
    c->comp_sizes[i] = (size_t)rc;
}

struct zstd_ctx
{
  const char* src;
  size_t input_stride;
  char* dst;
  size_t max_output_size;
  size_t* comp_sizes;
  size_t chunk_bytes;
  int level;
  _Atomic int err;
};

static void
zstd_one(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct zstd_ctx* c = (struct zstd_ctx*)vctx;
  if (atomic_load_explicit(&c->err, memory_order_relaxed))
    return;
  size_t rc = ZSTD_compress(c->dst + i * c->max_output_size,
                            c->max_output_size,
                            c->src + i * c->input_stride,
                            c->chunk_bytes,
                            c->level);
  if (ZSTD_isError(rc))
    atomic_store_explicit(&c->err, 1, memory_order_relaxed);
  else
    c->comp_sizes[i] = rc;
}

int
compress_cpu(struct codec_config codec,
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
  switch (codec.id) {
    case CODEC_NONE: {
      struct copy_ctx c = { (const char*)src, input_stride, (char*)dst,
                            max_output_size,  comp_sizes,   chunk_bytes };
      threadpool_for_n(pool, batch_size, copy_range, &c);
      return 0;
    }

    case CODEC_LZ4_NON_STANDARD: {
      struct lz4_ctx c = { (const char*)src, input_stride,
                           (char*)dst,       max_output_size,
                           comp_sizes,       chunk_bytes,
                           codec.level,      0 };
      threadpool_for_n_dynamic(pool, batch_size, lz4_one, &c);
      return atomic_load_explicit(&c.err, memory_order_acquire);
    }

    case CODEC_ZSTD: {
      struct zstd_ctx c = { (const char*)src, input_stride,
                            (char*)dst,       max_output_size,
                            comp_sizes,       chunk_bytes,
                            codec.level,      0 };
      threadpool_for_n_dynamic(pool, batch_size, zstd_one, &c);
      return atomic_load_explicit(&c.err, memory_order_acquire);
    }

    case CODEC_BLOSC_LZ4:
    case CODEC_BLOSC_ZSTD:
      return compress_blosc(codec,
                            src,
                            input_stride,
                            dst,
                            max_output_size,
                            comp_sizes,
                            chunk_bytes,
                            batch_size,
                            bytes_per_element,
                            pool);

    default:
      return 1;
  }
}
