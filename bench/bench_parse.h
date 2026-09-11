#pragma once

#include "dtype.h"
#include "test_data.h"
#include "types.codec.h"
#include "types.lod.h"

#include <stddef.h>
#include <stdint.h>

enum bench_backend
{
  BENCH_GPU,
  BENCH_CPU,
};

// Byte-size parser: "256K", "1M", "8G" etc. Leaves *out alone when the text
// is not a size that fits.
int
parse_bytes(const char* s, uint64_t* out);

// CLI option parsers. parse_fill returns NULL on error.
// The int-returning parsers return non-zero on success, 0 on error.
fill_fn
parse_fill(const char* s);

int
parse_codec(const char* s, struct codec_config* out);

int
parse_shuffle(const char* s, enum codec_shuffle* out);

const char*
bench_shuffle_name(enum codec_shuffle shuffle);

int
parse_level(const char* s, uint8_t* out);

uint8_t
bench_default_level(enum compression_codec codec);

int
parse_reduce(const char* s, enum lod_reduce_method* out);

int
parse_backend(const char* s, enum bench_backend* out);

int
parse_dtype(const char* s, enum dtype* out);
