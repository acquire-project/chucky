#include "cpu/pipeline.h"
#include "cpu/stream.body.h"
#include "dimension.h"
#include "multiarray.cpu.h"
#include "platform/platform.h"
#include "stream/config.h"
#include "stream/host_output_pool.h"
#include "threadpool/threadpool.h"
#include "zarr/shard_delivery.h"

#include "cpu/compress.h"
#include "cpu/compress_blosc.h"
#include "cpu/transpose.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// ---- Per-array descriptor ----

struct array_descriptor
{
  struct tile_stream_configuration config;
  struct computed_stream_layouts cl;
  struct tile_stream_layout layout;
  struct level_geometry levels;
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];
  struct shard_state shard[LOD_MAX_LEVELS];
  struct shard_sink* sink;
  uint64_t cursor_elements;
  uint64_t total_element_limit;
  uint32_t batch_accumulated;
  uint32_t* batch_active_masks;  // [K], heap-allocated
  uint32_t* pool_epochs_scratch; // [LOD_MAX_LEVELS * K], heap-allocated
  uint32_t append_counts[LOD_MAX_LEVELS];
  void* append_accum;
  struct reduce_csr* csrs; // [nlod-1] CSR reduce LUTs (multiscale only), owned
  uint8_t agg_current;     // next slot to use (0 or 1)
  size_t shard_alignment;  // from sink; 0 = no alignment
  int pool_fully_covered;  // 1 if scatter overwrites every pool position
  int flushed;             // 1 once finalized; no further input is taken
  int closed;              // 1 once close drained and, on success, published
  int close_failed;        // outcome of that close, re-reported by later calls
};

// ---- Main struct ----

struct multiarray_tile_stream_cpu
{
  struct multiarray_writer writer;
  int n_arrays;
  int active;            // -1 = none
  int luts_computed_for; // -1 = none, array index of last LUT computation
  int metrics_enabled;
  struct threadpool* pool; // shared across all arrays in this multiarray

  struct array_descriptor* arrays;

  // Shared pools — sized for max across all arrays.
  void* chunk_pool;
  size_t chunk_pool_bytes;
  void* compressed;
  size_t* comp_sizes;
  struct host_output_pool* output_pool;

  // Shared aggregate workspace: one double-buffered scratch pair across all
  // LODs, sized for the worst-case unified batch across all arrays.
  struct cpu_agg_slot agg_slots[2];

  // Shared LUT storage (recomputed on switch).
  uint32_t* scatter_lut;
  uint64_t* scatter_fixed_dims_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_fixed_dims_offsets[LOD_MAX_LEVELS];

  // Shared LOD buffers (multiscale only).
  void* linear;
  void* lod_values;
  size_t lod_values_bytes;

  struct stream_metrics metrics;
};

// ---- Forward declarations ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data);
static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self);
static struct multiarray_writer_result
close_impl(struct multiarray_writer* self);
static struct cpu_stream_view
make_multiarray_view(struct multiarray_tile_stream_cpu* ms,
                     struct array_descriptor* desc);

// ---- Helpers ----

static inline size_t
max_sz(size_t a, size_t b)
{
  return a > b ? a : b;
}

static inline struct multiarray_writer_result
multiarray_writer_fail_at(const void* beg, const void* end)
{
  return (struct multiarray_writer_result){
    .error = multiarray_writer_fail,
    .rest = { .beg = beg, .end = end },
  };
}

// Shared buffer sizes computed across all arrays (used by create).
struct pool_maxima
{
  size_t chunk_pool_bytes;
  size_t compressed_bytes;
  size_t comp_sizes_count;
  size_t linear_bytes;
  size_t lod_values_bytes;
  size_t host_output_bytes;

  // Unified per-batch maxima across all arrays.
  uint64_t total_batch_chunks_max;   // unified gather/perm len
  uint64_t total_batch_covering_max; // unified offsets/sizes len

  size_t scatter_lut_count;
  size_t scatter_fixed_dims_offsets_count;
  size_t morton_lut_count[LOD_MAX_LEVELS];
  size_t lod_fixed_dims_offsets_count[LOD_MAX_LEVELS];
};

// ---- Per-array init ----

static int
init_array_descriptor(struct array_descriptor* desc,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct pool_maxima* maxima,
                      struct threadpool* pool)
{
  if (config->dtype == dtype_f16)
    return 1;
  if (codec_is_blosc(config->codec.id) &&
      compress_blosc_validate(config->codec))
    return 1;

  desc->config = *config;
  desc->sink = sink;
  desc->shard_alignment = shard_sink_required_shard_alignment(sink);

  if (compute_stream_layouts(config,
                             1,
                             compress_cpu_max_output_size,
                             desc->shard_alignment,
                             &desc->cl))
    return 1;

  desc->layout = desc->cl.layouts[0];
  desc->levels = desc->cl.levels;
  desc->pool_fully_covered =
    (desc->layout.chunk_stride == desc->layout.chunk_elements);

  const uint32_t K = desc->cl.epochs_per_batch;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  const uint64_t total_chunks = desc->levels.total_chunks;

  desc->batch_active_masks = (uint32_t*)calloc(K, sizeof(uint32_t));
  if (!desc->batch_active_masks)
    return 1;
  desc->pool_epochs_scratch =
    (uint32_t*)malloc((size_t)LOD_MAX_LEVELS * (size_t)K * sizeof(uint32_t));
  if (!desc->pool_epochs_scratch)
    return 1;

  // total_element_limit: configured stream length (0 = unbounded)
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&desc->cl.dims);
    if (dims[0].size > 0) {
      desc->total_element_limit = desc->layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        desc->total_element_limit *= ceildiv(dims[d].size, dims[d].chunk_size);
    } else {
      desc->total_element_limit = 0;
    }
  }

  // Update pool maxima.
  maxima->chunk_pool_bytes = max_sz(
    maxima->chunk_pool_bytes,
    (size_t)K * total_chunks * desc->layout.chunk_stride * bytes_per_element);
  maxima->compressed_bytes =
    max_sz(maxima->compressed_bytes,
           (size_t)K * total_chunks * desc->cl.max_output_size);
  maxima->comp_sizes_count =
    max_sz(maxima->comp_sizes_count, (size_t)K * total_chunks);

  // Per-level shard + aggregate state.
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    const struct level_layout_info* li = &desc->cl.per_level[lv];
    desc->agg_layout[lv] = li->agg_layout;
    if (init_shard_state(&desc->shard[lv], li))
      return 1;
  }

  // Worst-case unified batch layout for this array — every LOD active for
  // every epoch of its batch_active_count.
  {
    uint32_t worst[LOD_MAX_LEVELS];
    for (int lv = 0; lv < desc->levels.nlod; ++lv) {
      const uint32_t count = desc->cl.per_level[lv].batch_active_count;
      worst[lv] = count ? count : 1;
    }
    struct batch_aggregate_layout max_batch_layout;
    if (batch_aggregate_layout_init_compact(&max_batch_layout,
                                            desc->agg_layout,
                                            worst,
                                            (uint8_t)desc->levels.nlod))
      return 1;
    size_t run_count = 0;
    size_t required_output_bytes = 0;
    const enum host_batch_storage storage = host_batch_storage_select(
      config->codec.id == CODEC_NONE, desc->shard_alignment);
    if (host_batch_capacity(desc->agg_layout,
                            worst,
                            (uint8_t)desc->levels.nlod,
                            storage,
                            desc->shard_alignment,
                            &required_output_bytes,
                            &run_count))
      return 1;
    (void)run_count;
    size_t output_bytes = 0;
    if (host_output_size(
          required_output_bytes, platform_page_alignment(), &output_bytes))
      return 1;
    maxima->host_output_bytes = max_sz(maxima->host_output_bytes, output_bytes);
    if (max_batch_layout.total_batch_chunks > maxima->total_batch_chunks_max)
      maxima->total_batch_chunks_max = max_batch_layout.total_batch_chunks;
    if (max_batch_layout.total_batch_covering >
        maxima->total_batch_covering_max)
      maxima->total_batch_covering_max = max_batch_layout.total_batch_covering;
  }

  // LOD sizes.
  if (desc->levels.enable_multiscale) {
    maxima->linear_bytes = max_sz(
      maxima->linear_bytes, desc->layout.epoch_elements * bytes_per_element);

    uint64_t total_lod_elements =
      desc->cl.plan.level_spans.ends[desc->cl.plan.levels.nlod - 1];
    maxima->lod_values_bytes =
      max_sz(maxima->lod_values_bytes, total_lod_elements * bytes_per_element);
    maxima->scatter_lut_count = max_sz(maxima->scatter_lut_count,
                                       desc->cl.plan.levels.level[0].lod_nelem);
    maxima->scatter_fixed_dims_offsets_count = max_sz(
      maxima->scatter_fixed_dims_offsets_count, desc->cl.plan.fixed_dims_count);

    for (int lv = 0; lv < desc->levels.nlod; ++lv) {
      maxima->morton_lut_count[lv] = max_sz(
        maxima->morton_lut_count[lv], desc->cl.plan.levels.level[lv].lod_nelem);
      maxima->lod_fixed_dims_offsets_count[lv] =
        max_sz(maxima->lod_fixed_dims_offsets_count[lv],
               desc->cl.plan.levels.level[lv].fixed_dims_count);
    }

    // Append accum: per-array, not shared.
    if (desc->cl.dims.append_downsample) {
      uint64_t append_total = 0;
      for (int lv = 1; lv < desc->cl.plan.levels.nlod; ++lv)
        append_total += desc->cl.plan.fixed_dims_count *
                        desc->cl.plan.levels.level[lv].lod_nelem;
      if (append_total > 0) {
        desc->append_accum = calloc(append_total, bytes_per_element);
        if (!desc->append_accum)
          return 1;
      }
      memset(desc->append_counts, 0, sizeof(desc->append_counts));
    }

    // CSR reduce LUTs: per-array, not shared (plans differ per array).
    int ncsr = desc->cl.plan.levels.nlod - 1;
    if (ncsr > 0) {
      desc->csrs = (struct reduce_csr*)calloc(ncsr, sizeof(struct reduce_csr));
      if (!desc->csrs)
        return 1;
      for (int l = 0; l < ncsr; ++l) {
        const struct level_dims* src_ld = &desc->cl.plan.levels.level[l];
        const struct level_dims* dst_ld = &desc->cl.plan.levels.level[l + 1];
        uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
        uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;
        if (reduce_csr_alloc(&desc->csrs[l], src_total, dst_total))
          return 1;
        if (reduce_csr_build(&desc->csrs[l], &desc->cl.plan, l, pool))
          return 1;
      }
    }
  }

  return 0;
}

// ---- Shared buffer allocation ----

static int
alloc_shared_buffers(struct multiarray_tile_stream_cpu* ms,
                     const struct pool_maxima* mx)
{
  if (mx->chunk_pool_bytes > 0) {
    ms->chunk_pool = calloc(1, mx->chunk_pool_bytes);
    CHECK(Fail, ms->chunk_pool);
    ms->chunk_pool_bytes = mx->chunk_pool_bytes;
  }
  if (mx->compressed_bytes > 0) {
    ms->compressed = malloc(mx->compressed_bytes);
    CHECK(Fail, ms->compressed);
  }
  if (mx->comp_sizes_count > 0) {
    ms->comp_sizes = (size_t*)calloc(mx->comp_sizes_count, sizeof(size_t));
    CHECK(Fail, ms->comp_sizes);
  }
  ms->output_pool =
    host_output_pool_create(mx->host_output_bytes,
                            platform_page_alignment(),
                            (struct host_output_allocator){ 0 });
  CHECK(Fail, ms->output_pool);

  // Unified per-batch aggregate scratch slots, double-buffered.
  {
    const uint64_t cap_chunks = mx->total_batch_chunks_max;
    const uint64_t cap_cov = mx->total_batch_covering_max;
    for (int fc = 0; fc < 2; ++fc) {
      struct cpu_agg_slot* as = &ms->agg_slots[fc];
      if (cap_chunks > 0) {
        as->perm = (uint32_t*)malloc(cap_chunks * sizeof(uint32_t));
        as->gather = (uint32_t*)malloc(cap_chunks * sizeof(uint32_t));
        CHECK(Fail, as->perm && as->gather);
      }
      if (cap_cov > 0) {
        // +LOD_MAX_LEVELS slack for the per-LOD shift applied by the
        // unified aggregate.
        const uint64_t cov_alloc = cap_cov + LOD_MAX_LEVELS;
        as->permuted_sizes = (size_t*)calloc(cov_alloc, sizeof(size_t));
        as->offsets = (size_t*)malloc(cov_alloc * sizeof(size_t));
        as->chunk_sizes = (size_t*)calloc(cov_alloc, sizeof(size_t));
        CHECK(Fail, as->permuted_sizes && as->offsets && as->chunk_sizes);
      }
    }
  }
  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    if (mx->morton_lut_count[lv] > 0) {
      ms->morton_lut[lv] =
        (uint32_t*)malloc(mx->morton_lut_count[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->morton_lut[lv]);
    }
    if (mx->lod_fixed_dims_offsets_count[lv] > 0) {
      ms->lod_fixed_dims_offsets[lv] = (uint64_t*)calloc(
        mx->lod_fixed_dims_offsets_count[lv], sizeof(uint64_t));
      CHECK(Fail, ms->lod_fixed_dims_offsets[lv]);
    }
  }
  if (mx->scatter_lut_count > 0) {
    ms->scatter_lut =
      (uint32_t*)malloc(mx->scatter_lut_count * sizeof(uint32_t));
    CHECK(Fail, ms->scatter_lut);
  }
  if (mx->scatter_fixed_dims_offsets_count > 0) {
    ms->scatter_fixed_dims_offsets =
      (uint64_t*)calloc(mx->scatter_fixed_dims_offsets_count, sizeof(uint64_t));
    CHECK(Fail, ms->scatter_fixed_dims_offsets);
  }

  // LOD buffers.
  if (mx->linear_bytes > 0) {
    ms->linear = calloc(1, mx->linear_bytes);
    CHECK(Fail, ms->linear);
  }
  if (mx->lod_values_bytes > 0) {
    ms->lod_values = calloc(1, mx->lod_values_bytes);
    CHECK(Fail, ms->lod_values);
    ms->lod_values_bytes = mx->lod_values_bytes;
  }

  return 0;

Fail:
  return 1;
}

// ---- Create / Destroy ----

struct multiarray_tile_stream_cpu*
multiarray_tile_stream_cpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[],
  int enable_metrics)
{
  if (n_arrays <= 0 || !configs || !sinks)
    return NULL;

  struct multiarray_tile_stream_cpu* ms =
    (struct multiarray_tile_stream_cpu*)calloc(1, sizeof(*ms));
  if (!ms)
    return NULL;

  ms->n_arrays = n_arrays;
  ms->active = -1;
  ms->luts_computed_for = -1;
  {
    int nthreads = configs[0].max_threads > 0 ? configs[0].max_threads
                                              : platform_default_thread_count();
    ms->pool = threadpool_new(nthreads - 1);
    CHECK(Fail, ms->pool);
  }

  ms->arrays = (struct array_descriptor*)calloc(
    (size_t)n_arrays, sizeof(struct array_descriptor));
  CHECK(Fail, ms->arrays);

  struct pool_maxima maxima = { 0 };
  for (int i = 0; i < n_arrays; ++i)
    CHECK(Fail,
          init_array_descriptor(
            &ms->arrays[i], &configs[i], sinks[i], &maxima, ms->pool) == 0);

  CHECK(Fail, alloc_shared_buffers(ms, &maxima) == 0);

  ms->writer.update = update_impl;
  ms->writer.flush = flush_impl;
  ms->writer.close = close_impl;

  if (enable_metrics) {
    ms->metrics_enabled = 1;
    ms->metrics.scatter = mk_stream_metric("scatter", METRIC_OWNER_COMPUTE);
    ms->metrics.compress = mk_stream_metric("compress", METRIC_OWNER_COMPRESS);
    ms->metrics.aggregate =
      mk_stream_metric("aggregate", METRIC_OWNER_COMPRESS);
    ms->metrics.sink = mk_stream_metric("sink", METRIC_OWNER_DELIVERY);
    int any_multiscale = 0;
    for (int i = 0; i < n_arrays; ++i)
      any_multiscale |= ms->arrays[i].levels.enable_multiscale;
    if (any_multiscale) {
      ms->metrics.lod_gather =
        mk_stream_metric("lod_gather", METRIC_OWNER_COMPUTE);
      ms->metrics.lod_reduce =
        mk_stream_metric("lod_reduce", METRIC_OWNER_COMPUTE);
      ms->metrics.lod_append_fold =
        mk_stream_metric("lod_append_fold", METRIC_OWNER_COMPUTE);
      ms->metrics.lod_morton_chunk =
        mk_stream_metric("lod_morton", METRIC_OWNER_COMPUTE);
    }
  }

  return ms;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  return NULL;
}

void
multiarray_tile_stream_cpu_destroy(struct multiarray_tile_stream_cpu* ms)
{
  if (!ms)
    return;

  // Auto-finalize any unflushed arrays so destroy is a safe commit point
  // for callers that didn't explicitly flush. Errors are logged but not
  // propagated — destroy returns void.
  {
    struct multiarray_writer_result r = flush_impl(&ms->writer);
    if (r.error)
      log_error("CPU multiarray auto-flush failed during destroy");
  }

  if (ms->arrays && close_impl(&ms->writer).error)
    log_error("CPU multiarray close failed during destroy");

  if (ms->arrays) {
    for (int i = 0; i < ms->n_arrays; ++i) {
      struct array_descriptor* desc = &ms->arrays[i];
      for (int lv = 0; lv < desc->levels.nlod; ++lv) {
        shard_state_destroy(&desc->shard[lv]);
      }
      free(desc->append_accum);
      free(desc->batch_active_masks);
      free(desc->pool_epochs_scratch);
      if (desc->csrs) {
        int ncsr = desc->cl.plan.levels.nlod - 1;
        for (int l = 0; l < ncsr; ++l)
          reduce_csr_free(&desc->csrs[l]);
        free(desc->csrs);
      }
      computed_stream_layouts_free(&desc->cl);
    }
    free(ms->arrays);
  }

  free(ms->chunk_pool);
  free(ms->compressed);
  free(ms->comp_sizes);

  for (int fc = 0; fc < 2; ++fc) {
    struct cpu_agg_slot* as = &ms->agg_slots[fc];
    host_batch_destroy(&as->host);
    free(as->perm);
    free(as->gather);
    free(as->permuted_sizes);
    free(as->offsets);
    free(as->chunk_sizes);
  }
  host_output_pool_destroy(ms->output_pool);
  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    free(ms->morton_lut[lv]);
    free(ms->lod_fixed_dims_offsets[lv]);
  }
  free(ms->scatter_lut);
  free(ms->scatter_fixed_dims_offsets);
  free(ms->linear);
  free(ms->lod_values);
  threadpool_free(ms->pool);
  free(ms);
}

struct multiarray_writer*
multiarray_tile_stream_cpu_writer(struct multiarray_tile_stream_cpu* ms)
{
  return ms ? &ms->writer : NULL;
}

struct stream_metrics
multiarray_tile_stream_cpu_get_metrics(
  const struct multiarray_tile_stream_cpu* ms)
{
  return ms->metrics;
}

// ---- LUT recomputation ----

static void
recompute_luts(struct multiarray_tile_stream_cpu* ms, int array_index)
{
  struct array_descriptor* desc = &ms->arrays[array_index];
  struct lut_targets luts = {
    .scatter_lut = ms->scatter_lut,
    .scatter_fixed_dims_offsets = ms->scatter_fixed_dims_offsets,
  };
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    luts.morton_lut[lv] = ms->morton_lut[lv];
    luts.lod_fixed_dims_offsets[lv] = ms->lod_fixed_dims_offsets[lv];
  }
  cpu_pipeline_compute_luts(&desc->cl, &desc->levels, ms->pool, &luts);
  ms->luts_computed_for = array_index;
}

static void
ensure_luts(struct multiarray_tile_stream_cpu* ms, int array_index)
{
  if (ms->luts_computed_for != array_index)
    recompute_luts(ms, array_index);
}

// Forward declaration
static struct cpu_stream_view
make_multiarray_view(struct multiarray_tile_stream_cpu* ms,
                     struct array_descriptor* desc);

// ---- Writer: update helpers ----

// Flush the departing array's pending batch and prepare shared buffers for the
// incoming array.  Returns 0 on success.
static int
switch_to_array(struct multiarray_tile_stream_cpu* ms, int array_index)
{
  if (ms->active >= 0) {
    struct array_descriptor* departing = &ms->arrays[ms->active];

    if (departing->cursor_elements % departing->layout.epoch_elements != 0)
      return multiarray_writer_not_flushable;

    if (departing->batch_accumulated > 0) {
      // Only flush the accumulated batch — NOT the full flush body.
      // Shard finalization and metadata happen in flush_impl.
      struct cpu_stream_view v = make_multiarray_view(ms, departing);
      if (cpu_stream_flush_batch(&v))
        return multiarray_writer_fail;
    }
  }

  ms->active = array_index;
  ensure_luts(ms, array_index);

  memset(ms->chunk_pool, 0, ms->chunk_pool_bytes);

  struct array_descriptor* desc = &ms->arrays[array_index];
  if (desc->levels.enable_multiscale && ms->lod_values)
    memset(ms->lod_values, 0, ms->lod_values_bytes);

  return 0;
}

// ---- View builder ----

static struct cpu_stream_view
make_multiarray_view(struct multiarray_tile_stream_cpu* ms,
                     struct array_descriptor* desc)
{
  struct cpu_stream_view v = {
    .config = &desc->config,
    .sink = desc->sink,
    .cl = &desc->cl,
    .layout = &desc->layout,
    .levels = &desc->levels,
    .cursor_elements = &desc->cursor_elements,
    .total_element_limit = desc->total_element_limit,
    .batch_accumulated = &desc->batch_accumulated,
    .batch_active_masks = desc->batch_active_masks,
    .pool_epochs_scratch = desc->pool_epochs_scratch,
    .pool_fully_covered = desc->pool_fully_covered,
    .shard = desc->shard,
    .agg_layout = desc->agg_layout,
    .csrs = desc->csrs,
    .append_accum = desc->append_accum,
    .append_counts = desc->append_counts,
    .agg_current = &desc->agg_current,
    .chunk_pool = ms->chunk_pool,
    .compressed = ms->compressed,
    .comp_sizes = ms->comp_sizes,
    .agg_slots = ms->agg_slots,
    .output_pool = ms->output_pool,
    .linear = desc->levels.enable_multiscale ? ms->linear : NULL,
    .lod_values = desc->levels.enable_multiscale ? ms->lod_values : NULL,
    .scatter_lut = desc->levels.enable_multiscale ? ms->scatter_lut : NULL,
    .scatter_fixed_dims_offsets =
      desc->levels.enable_multiscale ? ms->scatter_fixed_dims_offsets : NULL,
    .pool = ms->pool,
    .shard_alignment = desc->shard_alignment,
    .metrics = ms->metrics_enabled ? &ms->metrics : NULL,
    .metadata_update_clock = NULL, // multiarray defers metadata to final flush
  };
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    v.morton_lut[lv] = ms->morton_lut[lv];
    v.lod_fixed_dims_offsets[lv] = ms->lod_fixed_dims_offsets[lv];
  }
  return v;
}

// ---- Writer: update ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data)
{
  struct multiarray_tile_stream_cpu* ms =
    container_of(self, struct multiarray_tile_stream_cpu, writer);

  if (array_index < 0 || array_index >= ms->n_arrays)
    return multiarray_writer_fail_at(data.beg, data.end);

  struct array_descriptor* desc = &ms->arrays[array_index];

  if (desc->flushed)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_finished,
      .rest = data,
    };

  // Switch arrays if needed.
  if (array_index != ms->active) {
    int err = switch_to_array(ms, array_index);
    if (err)
      return (struct multiarray_writer_result){ .error = err, .rest = data };
  }

  struct cpu_stream_view v = make_multiarray_view(ms, desc);
  struct writer_result r = cpu_stream_append_body(&v, data);

  // `writer_finished` here means the array is at capacity
  // (total_element_limit); an already-finalized array is refused above.
  return (struct multiarray_writer_result){
    .error = r.error,
    .rest = r.rest,
  };
}

// ---- Writer: flush ----

static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_cpu* ms =
    container_of(self, struct multiarray_tile_stream_cpu, writer);

  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor* desc = &ms->arrays[a];
    // Finalizing twice would re-finalize an already-closed sink, which
    // deadlocks on Windows.
    if (desc->flushed)
      continue;
    if (desc->cursor_elements == 0 && desc->batch_accumulated == 0) {
      desc->flushed = 1;
      continue;
    }

    // Ensure LUTs are computed for this array (shared LUTs may be stale).
    if (a != ms->active)
      ms->active = a;
    ensure_luts(ms, a);

    struct cpu_stream_view v = make_multiarray_view(ms, desc);
    struct writer_result r = cpu_stream_flush_body(&v);
    // Latched even on failure: a flush that died partway may already have
    // closed shards, so taking more input would append past them. Those writes
    // are new, so an earlier close no longer covers them.
    desc->flushed = 1;
    desc->closed = 0;
    if (r.error)
      goto Error;
  }

  return (struct multiarray_writer_result){
    .error = multiarray_writer_ok,
  };

Error:
  return (struct multiarray_writer_result){
    .error = multiarray_writer_fail,
  };
}

static struct multiarray_writer_result
close_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_cpu* ms =
    container_of(self, struct multiarray_tile_stream_cpu, writer);
  int failed = 0;
  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor* desc = &ms->arrays[a];
    if (desc->closed || !desc->flushed || !desc->sink) {
      failed |= desc->close_failed;
      continue;
    }
    struct cpu_stream_view v = make_multiarray_view(ms, desc);
    desc->close_failed = (cpu_stream_close_body(&v).error != 0);
    failed |= desc->close_failed;
    desc->closed = 1;
  }
  return (struct multiarray_writer_result){
    .error = failed ? multiarray_writer_fail : multiarray_writer_ok,
  };
}
