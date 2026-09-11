#include "bench_parse.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Byte-size parser: "256K", "1M", "8G" etc. ---

// Accept the "B" and "iB" tails people write, so "256KB" and "1GiB" mean
// what they look like.
static int
is_byte_tail(const char* s)
{
  if (*s == 'i' || *s == 'I') {
    ++s;
    if (*s != 'b' && *s != 'B')
      return 0;
  }
  if (*s == 'b' || *s == 'B')
    ++s;
  return *s == '\0';
}

int
parse_bytes(const char* s, uint64_t* out)
{
  // strtoull would turn a negative into a huge size, so refuse it first.
  while (isspace((unsigned char)*s))
    ++s;
  if (*s == '-')
    return 0;

  char* end = NULL;
  errno = 0;
  unsigned long long val = strtoull(s, &end, 10);
  if (end == s || errno == ERANGE)
    return 0;

  unsigned shift = 0;
  const char* tail = end;
  switch (*tail) {
    case 'k':
    case 'K':
      shift = 10;
      ++tail;
      break;
    case 'm':
    case 'M':
      shift = 20;
      ++tail;
      break;
    case 'g':
    case 'G':
      shift = 30;
      ++tail;
      break;
    default:
      break;
  }
  if (!is_byte_tail(tail))
    return 0;

  if (shift) {
    if (val > (ULLONG_MAX >> shift))
      return 0;
    val <<= shift;
  }
  *out = val;
  return 1;
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
    return 1;
  }
  fprintf(stderr,
          "Unknown codec: %s (expected none, lz4, zstd, blosc-lz4, "
          "blosc-zstd)\n",
          s);
  return 0;
}

uint8_t
bench_default_level(enum compression_codec codec)
{
  if (codec == CODEC_LZ4_NON_STANDARD)
    return 1;
  return codec_is_blosc(codec) ? 3 : 0;
}

static const char* const shuffle_names[] = { "none", "byte", "bit" };

const char*
bench_shuffle_name(enum codec_shuffle shuffle)
{
  return shuffle >= CODEC_SHUFFLE_NONE && shuffle <= CODEC_SHUFFLE_BIT
           ? shuffle_names[shuffle]
           : "unknown";
}

int
parse_shuffle(const char* s, enum codec_shuffle* out)
{
  int i = match_option(s, shuffle_names, 3);
  if (i < 3) {
    *out = (enum codec_shuffle)i;
    return 1;
  }
  fprintf(stderr, "Unknown shuffle: %s (expected none, byte, bit)\n", s);
  return 0;
}

int
parse_level(const char* s, uint8_t* out)
{
  char* end = NULL;
  errno = 0;
  long value = strtol(s, &end, 10);
  if (end != s && *end == '\0' && errno != ERANGE && value >= 0 &&
      value <= UINT8_MAX) {
    *out = (uint8_t)value;
    return 1;
  }
  fprintf(stderr, "Invalid level: %s (expected integer 0..255)\n", s);
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
