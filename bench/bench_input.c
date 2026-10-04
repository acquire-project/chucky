#include "bench_input.h"
#include "platform/platform.h"

#include <stdio.h>
#include <stdlib.h>

int
bench_input_load(struct bench_input* input,
                 const char* path,
                 enum dtype dtype,
                 size_t width,
                 size_t height)
{
  const uint16_t endian = 1;
  const size_t bpe = dtype_bpe(dtype);
  if (!input || !path ||
      (dtype != dtype_u8 && dtype != dtype_u16 && dtype != dtype_f32) ||
      !width || !height || width > SIZE_MAX / bpe / height ||
      (bpe > 1 && *(const unsigned char*)&endian != 1)) {
    fprintf(stderr,
            "Image replay requires little-endian u8, u16, or f32 frames\n");
    return 1;
  }
  *input = (struct bench_input){ 0 };
  size_t frame_elements = width * height;
  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  FILE* file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "Cannot open image pack: %s\n", path);
    return 1;
  }
  if (fseek(file, 0, SEEK_END))
    goto Fail;
  long length = ftell(file);
  if (length <= 0 || (uint64_t)length > SIZE_MAX ||
      (uint64_t)length % (frame_elements * bpe) != 0 ||
      fseek(file, 0, SEEK_SET))
    goto Fail;
  size_t bytes = (size_t)length;
  input->frame_elements = frame_elements;
  input->source_bytes = bytes;
  input->data = malloc(bytes);
  if (!input->data || fread(input->data, 1, bytes, file) != bytes)
    goto Fail;
  if (fgetc(file) != EOF || ferror(file))
    goto Fail;
  if (fclose(file)) {
    bench_input_free(input);
    return 1;
  }
  input->load_s = platform_toc(&clock);
  return 0;

Fail:
  fprintf(stderr, "Image pack is empty, incomplete, or unreadable: %s\n", path);
  fclose(file);
  bench_input_free(input);
  return 1;
}

struct slice
bench_source_slice(const struct bench_source* source,
                   uint64_t offset,
                   size_t maximum_bytes)
{
  if (!source || !source->data || !source->bytes || !maximum_bytes)
    return (struct slice){ 0 };
  const size_t start = offset % source->bytes;
  size_t count = source->bytes - start;
  if (count > maximum_bytes)
    count = maximum_bytes;
  return (struct slice){ source->data + start, source->data + start + count };
}

void
bench_input_free(struct bench_input* input)
{
  free(input->data);
  *input = (struct bench_input){ 0 };
}
