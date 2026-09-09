// Compare direct and buffered frame appends using the same tile-stream layout.
#include "bench_gpu.h"
#include "dimension.h"
#include "platform/platform.h"
#include "sink_discard.h"
#include "stream.cpu.h"
#include "writer.buffered.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct observed_writer
{
  struct writer writer;
  struct writer* downstream;
  int64_t* submitted;
  double* downstream_ms;
  double* completion_ms;
  size_t frame;
};

static struct writer_result
observed_append(struct writer* self, struct slice input)
{
  struct observed_writer* o = (struct observed_writer*)self;
  const int64_t start = platform_monotonic_ns();
  struct writer_result r = writer_append_wait(o->downstream, input);
  const int64_t end = platform_monotonic_ns();
  o->downstream_ms[o->frame] = (end - start) * 1e-6;
  o->completion_ms[o->frame] = (end - o->submitted[o->frame]) * 1e-6;
  ++o->frame;
  return r;
}

static struct writer_result
observed_flush(struct writer* self)
{
  return writer_flush(((struct observed_writer*)self)->downstream);
}

static struct writer_result
observed_close(struct writer* self)
{
  return writer_close(((struct observed_writer*)self)->downstream);
}

static int
compare(const void* a, const void* b)
{
  const double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static void
distribution(const char* name, double* values, size_t n)
{
  qsort(values, n, sizeof(*values), compare);
  printf("\"%s_ms\":{\"p50\":%.6f,\"p95\":%.6f,\"p99\":%.6f,\"max\":%.6f}",
         name,
         values[n / 2],
         values[(n - 1) * 95 / 100],
         values[(n - 1) * 99 / 100],
         values[n - 1]);
}

static int
number(const char* text, size_t* value)
{
  char* end;
  errno = 0;
  unsigned long long n = strtoull(text, &end, 10);
  if (errno || *text < '0' || *text > '9' || *end || n > SIZE_MAX)
    return 1;
  *value = (size_t)n;
  return 0;
}

int
main(int argc, char** argv)
{
  int gpu = 0;
  struct codec_config codec = { .id = CODEC_NONE };
  const char* codec_name = "none";
  size_t frames = 2048, fps = 0, slots = 0, side = 1024, threads = 4;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc)
      goto usage;
    const char* key = argv[i];
    const char* value = argv[++i];
    if (!strcmp(key, "--backend")) {
      if (strcmp(value, "cpu") && strcmp(value, "gpu"))
        goto usage;
      gpu = !strcmp(value, "gpu");
    } else if (!strcmp(key, "--codec")) {
      if (strcmp(value, "none") && strcmp(value, "zstd"))
        goto usage;
      codec_name = value;
      codec.id = !strcmp(value, "none") ? CODEC_NONE : CODEC_ZSTD;
    } else {
      size_t* out = !strcmp(key, "--frames")    ? &frames
                    : !strcmp(key, "--fps")     ? &fps
                    : !strcmp(key, "--slots")   ? &slots
                    : !strcmp(key, "--side")    ? &side
                    : !strcmp(key, "--threads") ? &threads
                                                : NULL;
      if (!out || number(value, out))
        goto usage;
    }
  }
  if (!frames || frames > 10000000 || !threads || threads > 1024 ||
      side < 256 || side > 4096 || side % 256 || fps > 1000000)
    goto usage;

  const size_t frame_bytes = side * side * sizeof(uint16_t);
  const size_t pattern_frames = 64;
  uint16_t* pattern = malloc(pattern_frames * frame_bytes);
  int64_t* submitted = calloc(frames, sizeof(*submitted));
  double* caller = calloc(frames, sizeof(*caller));
  double* downstream = calloc(frames, sizeof(*downstream));
  double* completion = calloc(frames, sizeof(*completion));
  double* lateness = calloc(frames, sizeof(*lateness));
  struct tile_stream_cpu* cpu = NULL;
  struct tile_stream_gpu* device = NULL;
  struct buffered_writer* buffered = NULL;
  int error = 1;
  if (!pattern || !submitted || !caller || !downstream || !completion ||
      !lateness)
    goto cleanup;
  // A deterministic spatial/temporal pattern; generation stays outside timing.
  for (size_t t = 0; t < pattern_frames; ++t)
    for (size_t y = 0; y < side; ++y)
      for (size_t x = 0; x < side; ++x)
        pattern[(t * side + y) * side + x] = (uint16_t)(x ^ y ^ (t * 127));

  struct dimension dims[3];
  dims_create(dims, "tyx", (uint64_t[]){ 0, side, side });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 1, 256, 256 });
  dims[0].chunks_per_shard = 32;
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 0, 1, 1 });
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 8 * frame_bytes,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = codec,
    .epochs_per_batch = 32,
    .max_threads = (int)threads,
  };
  struct discard_shard_sink sink;
  discard_shard_sink_init(&sink);
  if (gpu) {
    if (!bench_gpu_enabled() || bench_gpu_context_create())
      goto cleanup;
    device = bench_gpu_create(&config, &sink.base);
    if (!device)
      goto cleanup;
  } else {
    cpu = tile_stream_cpu_create(&config, &sink.base);
    if (!cpu)
      goto cleanup;
  }
  struct observed_writer observed = {
    .writer = { observed_append, observed_flush, observed_close },
    .downstream = gpu ? bench_gpu_writer(device) : tile_stream_cpu_writer(cpu),
    .submitted = submitted,
    .downstream_ms = downstream,
    .completion_ms = completion,
  };
  struct writer* writer = &observed.writer;
  if (slots) {
    buffered = buffered_writer_create(
      writer, &(struct buffered_writer_config){ frame_bytes, slots });
    if (!buffered)
      goto cleanup;
    writer = buffered_writer_as_writer(buffered);
  }
  const int64_t start = platform_monotonic_ns();
  int64_t halfway = start;
  for (size_t frame = 0; frame < frames; ++frame) {
    const int64_t due =
      start + (fps ? (int64_t)(frame * 1000000000ull / fps) : 0);
    int64_t now = platform_monotonic_ns();
    while (fps && now < due) {
      platform_sleep_ns(due - now);
      now = platform_monotonic_ns();
    }
    submitted[frame] = now;
    lateness[frame] = fps ? (now - due) * 1e-6 : 0;
    const unsigned char* data =
      (const unsigned char*)pattern + (frame % pattern_frames) * frame_bytes;
    struct writer_result r =
      writer_append_wait(writer, (struct slice){ data, data + frame_bytes });
    const int64_t end = platform_monotonic_ns();
    caller[frame] = (end - now) * 1e-6;
    if (r.error)
      goto cleanup;
    if (frame + 1 == frames / 2)
      halfway = end;
  }
  const int64_t appended = platform_monotonic_ns();
  if (writer_flush(writer).error || writer_close(writer).error)
    goto cleanup;
  const int64_t done = platform_monotonic_ns();
  struct buffered_writer_stats stats = { 0 };
  if (buffered)
    stats = buffered_writer_get_stats(buffered);
  int destroy_error = buffered_writer_destroy(buffered);
  buffered = NULL;
  if (destroy_error || observed.frame != frames)
    goto cleanup;
  printf("{\"backend\":\"%s\",\"codec\":\"%s\",\"frames\":%zu,\"fps\":%zu,"
         "\"frame_bytes\":%zu,\"slots\":%zu,\"threads\":%zu,"
         "\"wall_s\":%.6f,\"gib_s\":%.6f,\"second_half_gib_s\":%.6f,"
         "\"flush_ms\":%.6f,",
         gpu ? "gpu" : "cpu",
         codec_name,
         frames,
         fps,
         frame_bytes,
         slots,
         threads,
         (done - start) * 1e-9,
         (double)frames * frame_bytes / (done - start) * (1e9 / 1073741824.0),
         (double)(frames - frames / 2) * frame_bytes / (done - halfway) *
           (1e9 / 1073741824.0),
         (done - appended) * 1e-6);
  distribution("caller", caller, frames);
  printf(",");
  distribution("downstream", downstream, frames);
  printf(",");
  distribution("completion", completion, frames);
  printf(",");
  distribution("lateness", lateness, frames);
  printf(",\"peak_occupied_slots\":%zu,\"peak_pending_bytes\":%zu,"
         "\"backpressure_ms\":%.6f,\"queue_mean_ms\":%.6f,"
         "\"queue_max_ms\":%.6f,\"abandoned_bytes\":%llu}\n",
         stats.peak_occupied_slots,
         stats.peak_pending_bytes,
         stats.backpressure_ns * 1e-6,
         stats.completed_slots ? stats.queue_ns * 1e-6 / stats.completed_slots
                               : 0,
         stats.max_queue_ns * 1e-6,
         (unsigned long long)stats.abandoned_bytes);
  error = 0;
cleanup:
  error |= buffered_writer_destroy(buffered);
  if (cpu)
    tile_stream_cpu_destroy(cpu);
  if (device)
    bench_gpu_destroy(device);
  if (gpu)
    bench_gpu_context_destroy();
  free(lateness);
  free(completion);
  free(downstream);
  free(caller);
  free(submitted);
  free(pattern);
  return error;
usage:
  fprintf(
    stderr,
    "Usage: %s [--backend cpu|gpu] [--codec none|zstd] "
    "[--frames N] [--fps N] [--slots N] [--side N] [--threads N]\n"
    "fps=0: unpaced; slots=0: direct; side: multiple of 256, 256..4096.\n",
    argv[0]);
  return 2;
}
