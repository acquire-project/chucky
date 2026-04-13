#include "bench_parse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Byte-size parser: "256K", "1M", "8G" etc. ---

size_t
parse_bytes(const char* s)
{
  char* end = NULL;
  size_t val = (size_t)strtoull(s, &end, 10);
  if (end && *end) {
    switch (*end) {
      case 'k':
      case 'K':
        val <<= 10;
        break;
      case 'm':
      case 'M':
        val <<= 20;
        break;
      case 'g':
      case 'G':
        val <<= 30;
        break;
    }
  }
  return val;
}

// --- dtype helpers ---

static const char* const dtype_names[] = {
  "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "f32", "f64",
};
static const enum dtype dtype_vals[] = {
  dtype_u8,  dtype_u16, dtype_u32, dtype_u64, dtype_i8,  dtype_i16,
  dtype_i32, dtype_i64, dtype_f16, dtype_f32, dtype_f64,
};
#define NUM_DTYPES (sizeof(dtype_vals) / sizeof(dtype_vals[0]))

// Returns the index of the matching string, or n if no match.
static int
match_option(const char* s, const char* const* options, int n)
{
  for (int i = 0; i < n; ++i)
    if (strcmp(s, options[i]) == 0)
      return i;
  return n;
}

fill_fn
parse_fill(const char* s)
{
  static const char* const names[] = { "xor", "zeros", "rand" };
  static const fill_fn fns[] = { fill_xor, fill_zeros, fill_rand };
  int i = match_option(s, names, 3);
  if (i < 3)
    return fns[i];
  fprintf(stderr, "Unknown fill: %s (expected xor, zeros, rand)\n", s);
  return NULL;
}

int
parse_codec(const char* s, struct codec_config* out)
{
  static const char* const names[] = {
    "none", "lz4", "zstd", "blosc-lz4", "blosc-zstd"
  };
  static const enum compression_codec vals[] = { CODEC_NONE,
                                                 CODEC_LZ4_NON_STANDARD,
                                                 CODEC_ZSTD,
                                                 CODEC_BLOSC_LZ4,
                                                 CODEC_BLOSC_ZSTD };
  int i = match_option(s, names, 5);
  if (i < 5) {
    out->id = vals[i];
    if (out->id == CODEC_LZ4_NON_STANDARD && out->level == 0)
      out->level = 1;
    if (codec_is_blosc(out->id) && out->level == 0)
      out->level = 3;
    return 1;
  }
  fprintf(stderr,
          "Unknown codec: %s (expected none, lz4, zstd, blosc-lz4, "
          "blosc-zstd)\n",
          s);
  return 0;
}

int
parse_reduce(const char* s, enum lod_reduce_method* out)
{
  static const char* const names[] = { "mean",   "min",     "max",
                                       "median", "max_sup", "min_sup" };
  static const enum lod_reduce_method vals[] = {
    lod_reduce_mean,
    lod_reduce_min,
    lod_reduce_max,
    lod_reduce_median,
    lod_reduce_max_suppressed,
    lod_reduce_min_suppressed,
  };
  int i = match_option(s, names, 6);
  if (i < 6) {
    *out = vals[i];
    return 1;
  }
  fprintf(stderr,
          "Unknown reduce: %s (expected mean, min, max, median, max_sup, "
          "min_sup)\n",
          s);
  return 0;
}

int
parse_backend(const char* s, enum bench_backend* out)
{
  static const char* const names[] = { "gpu", "cpu" };
  static const enum bench_backend vals[] = { BENCH_GPU, BENCH_CPU };
  int i = match_option(s, names, 2);
  if (i < 2) {
    *out = vals[i];
    return 1;
  }
  fprintf(stderr, "Unknown backend: %s (expected gpu, cpu)\n", s);
  return 0;
}

int
parse_dtype(const char* s, enum dtype* out)
{
  int i = match_option(s, dtype_names, NUM_DTYPES);
  if (i < (int)NUM_DTYPES) {
    *out = dtype_vals[i];
    return 1;
  }
  fprintf(stderr,
          "Unknown dtype: %s (expected u8, u16, u32, u64, i8, i16, i32, i64, "
          "f16, f32, f64)\n",
          s);
  return 0;
}
