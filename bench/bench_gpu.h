#pragma once

// The GPU pipeline behind an interface with no CUDA in it, so the benchmarks
// build on machines that have no CUDA headers.

#include "types.stream.h"

#include <stddef.h>
#include <stdint.h>

struct shard_sink;
struct tile_stream_gpu;
struct tile_stream_layout;
struct writer;

int
bench_gpu_enabled(void);

// Create the process-wide context on device 0. Returns 0 on success.
int
bench_gpu_context_create(void);

void
bench_gpu_context_destroy(void);

// Free device memory in bytes, or 0 when it can't be read.
size_t
bench_gpu_free_memory(void);

int
bench_gpu_advise_layout(struct tile_stream_configuration* config,
                        size_t target_chunk_bytes,
                        size_t min_chunk_bytes,
                        const int* ratios,
                        size_t budget_bytes,
                        size_t min_shard_bytes,
                        uint32_t target_concurrent_shards,
                        uint32_t min_append_shards,
                        size_t shard_alignment,
                        struct advise_layout_diagnostic* diag);

void
bench_gpu_report_memory(const struct tile_stream_configuration* config,
                        uint64_t* total_chunks,
                        size_t* device_bytes,
                        size_t* pinned_bytes);

void
bench_gpu_report_memory_pair(const struct tile_stream_configuration* config);

struct tile_stream_gpu*
bench_gpu_create(const struct tile_stream_configuration* config,
                 struct shard_sink* sink);

void
bench_gpu_destroy(struct tile_stream_gpu* s);

const struct tile_stream_layout*
bench_gpu_layout(const struct tile_stream_gpu* s);

struct stream_metrics
bench_gpu_get_metrics(const struct tile_stream_gpu* s);

struct tile_stream_status
bench_gpu_status(const struct tile_stream_gpu* s);

int
bench_gpu_worker_threads(const struct tile_stream_gpu* s);

struct writer*
bench_gpu_writer(struct tile_stream_gpu* s);

uint64_t
bench_gpu_cursor(const struct tile_stream_gpu* s);
