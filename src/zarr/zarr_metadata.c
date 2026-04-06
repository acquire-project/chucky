#include "zarr/zarr_metadata.h"
#include "defs.limits.h"
#include "dtype.h"
#include "zarr/json_writer.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

int
zarr_root_json(char* buf, size_t cap)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "group");
  jw_key(&jw, "consolidated_metadata");
  jw_null(&jw);
  jw_key(&jw, "attributes");
  jw_object_begin(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}

int
zarr_array_json(char* buf,
                size_t cap,
                uint8_t rank,
                const struct dimension* dimensions,
                enum dtype data_type,
                double fill_value,
                const uint64_t* chunks_per_shard,
                struct codec_config codec)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "array");

  jw_key(&jw, "shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].size);
  jw_array_end(&jw);

  jw_key(&jw, "data_type");
  jw_string(&jw, dtype_zarr_string(data_type));

  // chunk_grid: shard = chunk_size * chunks_per_shard
  jw_key(&jw, "chunk_grid");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "regular");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].chunk_size * chunks_per_shard[d]);
  jw_array_end(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  jw_key(&jw, "chunk_key_encoding");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "default");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "separator");
  jw_string(&jw, "/");
  jw_object_end(&jw);
  jw_object_end(&jw);

  // codecs: sharding_indexed
  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "sharding_indexed");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);

  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].chunk_size);
  jw_array_end(&jw);

  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  if (codec.id != CODEC_NONE) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    if (codec_is_blosc(codec.id))
      jw_string(&jw, "blosc");
    else
      jw_string(&jw, codec.id == CODEC_LZ4_NON_STANDARD ? "lz4" : "zstd");
    jw_key(&jw, "configuration");
    jw_object_begin(&jw);
    if (codec.id == CODEC_ZSTD) {
      jw_key(&jw, "level");
      jw_int(&jw, codec.level);
      jw_key(&jw, "checksum");
      jw_bool(&jw, false);
    }
    if (codec_is_blosc(codec.id)) {
      jw_key(&jw, "cname");
      jw_string(&jw, codec.id == CODEC_BLOSC_LZ4 ? "lz4" : "zstd");
      jw_key(&jw, "clevel");
      jw_int(&jw, codec.level);
      jw_key(&jw, "shuffle");
      jw_string(&jw,
                codec.shuffle == CODEC_SHUFFLE_BIT    ? "bitshuffle"
                : codec.shuffle == CODEC_SHUFFLE_BYTE ? "shuffle"
                                                      : "noshuffle");
      jw_key(&jw, "typesize");
      jw_int(&jw, (int64_t)dtype_bpe(data_type));
      jw_key(&jw, "blocksize");
      jw_int(&jw, 0);
    }
    jw_object_end(&jw);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "index_codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "crc32c");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_array_end(&jw);

  jw_key(&jw, "index_location");
  jw_string(&jw, "end");

  jw_object_end(&jw); // configuration
  jw_object_end(&jw); // sharding_indexed
  jw_array_end(&jw);  // codecs

  jw_key(&jw, "fill_value");
  jw_float(&jw, fill_value);

  jw_key(&jw, "storage_transformers");
  jw_array_begin(&jw);
  jw_array_end(&jw);

  jw_key(&jw, "attributes");
  jw_object_begin(&jw);
  jw_object_end(&jw);

  // dimension_names
  {
    int has_names = 0;
    for (int d = 0; d < rank; ++d) {
      if (dimensions[d].name) {
        has_names = 1;
        break;
      }
    }
    if (has_names) {
      jw_key(&jw, "dimension_names");
      jw_array_begin(&jw);
      for (int d = 0; d < rank; ++d) {
        if (dimensions[d].name)
          jw_string(&jw, dimensions[d].name);
        else
          jw_null(&jw);
      }
      jw_array_end(&jw);
    }
  }

  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}

int
zarr_for_each_intermediate(const char* array_name,
                           int (*fn)(const char* partial, void* ctx),
                           void* ctx)
{
  if (!array_name)
    return 0;

  size_t len = strlen(array_name);
  if (len == 0 || len >= 4096)
    return -1;

  // Reject leading slash, trailing slash, or empty segments (//)
  if (array_name[0] == '/' || array_name[len - 1] == '/')
    return -1;
  if (strstr(array_name, "//"))
    return -1;

  char name[4096];
  memcpy(name, array_name, len + 1);

  for (size_t i = 0; i < len; ++i) {
    if (name[i] == '/') {
      name[i] = '\0';
      int rc = fn(name, ctx);
      name[i] = '/';
      if (rc != 0)
        return rc;
    }
  }
  return 0;
}

int
zarr_shard_key(char* buf,
               size_t cap,
               uint8_t rank,
               const uint64_t* shard_count,
               uint64_t flat)
{
  int pos = snprintf(buf, cap, "c");
  if (pos < 0 || (size_t)pos >= cap)
    return -1;

  uint64_t coords[MAX_ZARR_RANK];
  uint64_t rem = flat;
  for (int d = rank - 1; d >= 0; --d) {
    if (d == 0) {
      coords[d] = rem;
    } else {
      coords[d] = rem % shard_count[d];
      rem /= shard_count[d];
    }
  }

  for (int d = 0; d < rank; ++d) {
    int n = snprintf(
      buf + pos, cap - (size_t)pos, "/%llu", (unsigned long long)coords[d]);
    if (n < 0 || (size_t)(pos + n) >= cap)
      return -1;
    pos += n;
  }
  return 0;
}
