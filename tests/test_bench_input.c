#include "bench_input.h"

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

int
main(void)
{
  uint16_t data[7];
  for (size_t i = 0; i < 7; ++i)
    data[i] = expected[i];
  struct bench_input input = { .data = data, .elements = 7 };
  const size_t appends[] = { 1, 3, 7, 19, 64 };
  const size_t limits[] = { 1, 4, 64 };
  for (size_t i = 0; i < sizeof(appends) / sizeof(*appends); ++i) {
    for (size_t j = 0; j < sizeof(limits) / sizeof(*limits); ++j) {
      struct check_writer writer = { .writer = { .append = check_append },
                                     .limit = limits[j] };
      if (bench_input_pump(&input, &writer.writer, 53, appends[i]) ||
          writer.offset != 53 || writer.mismatch) {
        fprintf(stderr, "Replay changed with append or partial consumption\n");
        return 1;
      }
    }
  }
  struct check_writer writer = { .writer = { .append = check_append },
                                 .limit = 3,
                                 .fail_at = 12 };
  if (!bench_input_pump(&input, &writer.writer, 53, 5) ||
      !bench_input_pump(&input, &writer.writer, 53, 0))
    return 1;
  input.elements = 0;
  return !bench_input_pump(&input, &writer.writer, 53, 5);
}
