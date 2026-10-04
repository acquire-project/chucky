#include "bench_input.h"
#include "test_data.h"
#include "test_platform.h"

#include <stdio.h>
#include <string.h>

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
test_load_packed(enum dtype dtype)
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

  if (bench_input_load(&input, path, dtype, 65, 66) ||
      input.frame_elements != 65 * 66 || input.source_bytes != source_bytes)
    goto Cleanup;
  if (memcmp(input.data, source, source_bytes))
    goto Cleanup;
  bench_input_free(&input);
  if (!bench_input_load(&input, path, dtype, SIZE_MAX, 1) ||
      !bench_input_load(&input, path, dtype, 0, 1) ||
      !bench_input_load(&input, path, dtype_f64, 65, 66))
    goto Cleanup;
  file = fopen(path, "wb");
  if (!file)
    goto Cleanup;
  written = fwrite(source, 1, source_bytes - 1, file);
  if (fclose(file) || written != source_bytes - 1 ||
      !bench_input_load(&input, path, dtype, 65, 66) || input.data)
    goto Cleanup;
  error = 0;

Cleanup:
  bench_input_free(&input);
  if (test_tmpdir_remove(directory))
    error = 1;
  return error;
}

static int
test_generated_prefix(void)
{
  const struct dimension dims[] = { { .size = 100 },
                                    { .size = 7 },
                                    { .size = 13 } };
  const uint8_t rank = 3;
  const size_t period = 16 * 7 * 13;
  const size_t sizes[] = { 1, 90, 91, 92, 1455, 1456, 1457, 4096 };
  uint16_t reference[4096], actual[4096];

  for (int random = 0; random < 2; ++random) {
    fill_fn fill = random ? fill_rand : fill_xor;
    if (random)
      rand_pattern_init(dims, rank, 16);
    else
      xor_pattern_init(dims, rank, 16);
    fill(reference, 4096, 0, 4096);
    rand_pattern_free();
    xor_pattern_free();

    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); ++i) {
      const size_t count = sizes[i];
      const size_t length = count < period ? count : period;
      if (random)
        rand_pattern_init_elements(length);
      else
        xor_pattern_init_elements(dims, rank, length);
      for (size_t offset = 0; offset < count;) {
        const size_t part = count - offset < 17 ? count - offset : 17;
        fill(actual + offset, part, offset, count);
        offset += part;
      }
      rand_pattern_free();
      xor_pattern_free();
      if (memcmp(actual, reference, count * sizeof(*actual))) {
        fprintf(stderr, "Bounded preparation changed the generated source\n");
        return 1;
      }
    }
  }
  return 0;
}

int
main(void)
{
  if (test_generated_prefix() || test_load_packed(dtype_u8) ||
      test_load_packed(dtype_u16) || test_load_packed(dtype_f32))
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
