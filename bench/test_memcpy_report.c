#include "bench_report.h"

#include <string.h>

int
main(int argc, char** argv)
{
  if (argc != 2)
    return 1;
  const int sampled = strcmp(argv[1], "sampled") == 0;
  const int empty = strcmp(argv[1], "empty") == 0;
  struct stream_metrics m = { 0 };
  if (!empty) {
    m.memcpy_calls = 128;
    m.memcpy_bytes = 128 * 512;
    m.memcpy = (struct stream_metric){
      .name = "Memcpy",
      .owner = METRIC_OWNER_PRODUCER,
      .count = sampled ? 2 : 128,
      .ms = 1,
      .best_ms = 0.25f,
      .max_ms = 0.75f,
      .best_input_bytes = 512,
      .best_output_bytes = 512,
      .input_bytes = sampled ? 1024 : 65536,
      .output_bytes = sampled ? 1024 : 65536,
    };
  }
  const struct tile_stream_layout layout = {
    .epoch_elements = 4096,
    .chunk_stride = 4096,
    .chunks_per_epoch = 1,
  };
  const struct bench_memory mem = { 0 };
  const struct sink_stats sink = { 0 };
  print_memcpy_metric(&m);
  print_bench_json_pass(&m,
                        NULL,
                        &layout,
                        dtype_u8,
                        (struct codec_config){ .id = CODEC_NONE },
                        &sink,
                        65536,
                        65536,
                        1,
                        0,
                        0,
                        &mem,
                        1);
  return 0;
}
