#include "bench_input.h"
#include "platform/platform.h"

#include <stdio.h>
#include <stdlib.h>

int
bench_input_load(struct bench_input* input,
                 const char* path,
                 size_t width,
                 size_t height)
{
  const uint16_t endian = 1;
  if (!input || !path || !width || !height || width > SIZE_MAX - 255 ||
      height > SIZE_MAX - 255 || width > SIZE_MAX / sizeof(uint16_t) / height ||
      *(const unsigned char*)&endian != 1) {
    fprintf(stderr, "Image replay requires little-endian u16 frames\n");
    return 1;
  }
  *input = (struct bench_input){ 0 };
  size_t padded_width = (width + 255) / 256 * 256;
  size_t padded_height = (height + 255) / 256 * 256;
  if (padded_width > SIZE_MAX / sizeof(uint16_t) / padded_height)
    return 1;
  size_t frame_elements = width * height;
  size_t padded_frame = padded_width * padded_height;
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
      (uint64_t)length % (frame_elements * sizeof(uint16_t)) != 0 ||
      fseek(file, 0, SEEK_SET))
    goto Fail;
  size_t bytes = (size_t)length;
  size_t frames = bytes / (frame_elements * sizeof(uint16_t));
  if (frames > SIZE_MAX / sizeof(uint16_t) / padded_frame)
    goto Fail;
  input->elements = frames * padded_frame;
  input->frame_elements = padded_frame;
  input->source_bytes = bytes;
  input->data = calloc(input->elements, sizeof(uint16_t));
  if (!input->data)
    goto Fail;
  if (padded_frame == frame_elements) {
    if (fread(input->data, 1, bytes, file) != bytes)
      goto Fail;
  } else {
    for (size_t frame = 0; frame < frames; ++frame)
      for (size_t row = 0; row < height; ++row) {
        uint16_t* data =
          input->data + frame * padded_frame + row * padded_width;
        if (fread(data, sizeof(uint16_t), width, file) != width)
          goto Fail;
      }
  }
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

int
bench_input_pump(const struct bench_input* input,
                 struct writer* writer,
                 size_t total_elements,
                 size_t append_elements)
{
  if (!input || !input->data || !input->elements || !writer ||
      !append_elements || !total_elements)
    return 1;
  for (size_t offset = 0; offset < total_elements;) {
    size_t start = offset % input->elements;
    size_t count = input->elements - start;
    if (count > append_elements)
      count = append_elements;
    if (count > total_elements - offset)
      count = total_elements - offset;
    struct slice data = { input->data + start, input->data + start + count };
    if (writer_append_wait(writer, data).error)
      return 1;
    offset += count;
  }
  return 0;
}

void
bench_input_free(struct bench_input* input)
{
  free(input->data);
  *input = (struct bench_input){ 0 };
}
