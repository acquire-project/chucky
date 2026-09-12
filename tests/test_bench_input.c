#include "bench_input.h"
#include "test_platform.h"

#include <stdio.h>

struct check_writer
{
  struct writer writer;
  size_t offset;
  size_t limit;
  size_t fail_at;
  int mismatch;
};

static const uint16_t expected[] = { 0, 65535, 231, 32768, 999, 42, 17 };

static struct writer_result
check_append(struct writer* writer, struct slice data)
{
  struct check_writer* check = (struct check_writer*)writer;
  if (check->fail_at && check->offset >= check->fail_at)
    return writer_error();
  const uint16_t* p = data.beg;
  const uint16_t* end = data.end;
  size_t count = (size_t)(end - p);
  if (count > check->limit)
    count = check->limit;
  for (size_t i = 0; i < count; ++i) {
    if (p[i] != expected[check->offset % 7])
      check->mismatch = 1;
    ++check->offset;
  }
  return (struct writer_result){ 0, { p + count, end } };
}

static int
replay(const struct bench_source* source,
       struct writer* writer,
       size_t elements,
       size_t append_elements)
{
  for (size_t offset = 0; offset < elements * 2;) {
    size_t maximum = append_elements * 2;
    if (maximum > elements * 2 - offset)
      maximum = elements * 2 - offset;
    const struct slice data = bench_source_slice(source, offset, maximum);
    if (!data.beg || data.beg == data.end ||
        writer_append_wait(writer, data).error)
      return 1;
    offset += (const unsigned char*)data.end - (const unsigned char*)data.beg;
  }
  return 0;
}

static int
test_load_padding(enum dtype dtype)
{
  char directory[1024];
  char path[1200];
  struct bench_input input = { 0 };
  unsigned char source[2 * 65 * 66 * 4];
  const size_t bpe = dtype_bpe(dtype);
  const size_t source_bytes = 2 * 65 * 66 * bpe;
  int error = 1;

  if (test_tmpdir_create(directory, sizeof(directory)) ||
      snprintf(path, sizeof(path), "%s/input.raw", directory) < 0)
    return 1;
  for (size_t i = 0; i < source_bytes; ++i)
    source[i] = (unsigned char)(i * 37 + 7);
  FILE* file = fopen(path, "wb");
  if (!file)
    goto Cleanup;
  size_t written = fwrite(source, 1, source_bytes, file);
  if (fclose(file) || written != source_bytes)
    goto Cleanup;

  if (bench_input_load(&input, path, dtype, 65, 66, 64, 64) ||
      input.elements != 2 * 128 * 128 || input.frame_elements != 128 * 128 ||
      input.logical_frame_elements != 65 * 66 ||
      input.source_bytes != source_bytes)
    goto Cleanup;
  for (size_t frame = 0; frame < 2; ++frame)
    for (size_t y = 0; y < 128; ++y)
      for (size_t x = 0; x < 128; ++x)
        for (size_t byte = 0; byte < bpe; ++byte) {
          unsigned char value =
            y < 66 && x < 65 ? source[((frame * 66 + y) * 65 + x) * bpe + byte]
                             : 0;
          if (input.data[((frame * 128 + y) * 128 + x) * bpe + byte] != value)
            goto Cleanup;
        }
  if (!bench_input_load(&input, path, dtype, SIZE_MAX, 1, 64, 64) ||
      !bench_input_load(&input, path, dtype, 1, 1, 0, 64) ||
      !bench_input_load(&input, path, dtype_f64, 65, 66, 64, 64))
    goto Cleanup;
  bench_input_free(&input);
  file = fopen(path, "wb");
  if (!file)
    goto Cleanup;
  written = fwrite(source, 1, source_bytes - 1, file);
  if (fclose(file) || written != source_bytes - 1 ||
      !bench_input_load(&input, path, dtype, 65, 66, 64, 64) || input.data)
    goto Cleanup;
  error = 0;

Cleanup:
  bench_input_free(&input);
  if (test_tmpdir_remove(directory))
    error = 1;
  return error;
}

int
main(void)
{
  if (test_load_padding(dtype_u8) || test_load_padding(dtype_u16) ||
      test_load_padding(dtype_f32))
    return 1;
  uint16_t data[7];
  for (size_t i = 0; i < 7; ++i)
    data[i] = expected[i];
  struct bench_source input = { .data = (const unsigned char*)data,
                                .bytes = sizeof(data) };
  const size_t appends[] = { 1, 3, 7, 19, 64 };
  const size_t limits[] = { 1, 4, 64 };
  for (size_t i = 0; i < sizeof(appends) / sizeof(*appends); ++i) {
    for (size_t j = 0; j < sizeof(limits) / sizeof(*limits); ++j) {
      struct check_writer writer = { .writer = { .append = check_append },
                                     .limit = limits[j] };
      if (replay(&input, &writer.writer, 53, appends[i]) ||
          writer.offset != 53 || writer.mismatch) {
        fprintf(stderr, "Replay changed with append or partial consumption\n");
        return 1;
      }
    }
  }
  struct check_writer writer = { .writer = { .append = check_append },
                                 .limit = 3,
                                 .fail_at = 12 };
  if (!replay(&input, &writer.writer, 53, 5) ||
      !replay(&input, &writer.writer, 53, 0))
    return 1;
  input.bytes = 0;
  return !replay(&input, &writer.writer, 53, 5);
}
