#include "cpu/stream.internal.h"
#include "stream/config.h"

#include "cpu/compress.h"
#include "cpu/compress_blosc.h"
#include "cpu/transpose.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>

// ---- Forward declarations ----

static struct writer_result
cpu_append(struct writer* self, struct slice input);
static struct writer_result
cpu_flush_final(struct writer* self);

// ---- Create / Destroy ----

struct tile_stream_cpu*
tile_stream_cpu_create(const struct tile_stream_configuration* config,
                       struct shard_sink* sink)
{
  if (!config || !sink)
    return NULL;
  if (config->dtype == dtype_f16)
    return NULL;
  if (codec_is_blosc(config->codec.id) &&
      compress_blosc_validate(config->codec))
    return NULL;

  struct tile_stream_cpu* s = (struct tile_stream_cpu*)calloc(1, sizeof(*s));
  if (!s)
    return NULL;

  s->config = *config;
  s->nthreads =
    config->max_threads > 0 ? config->max_threads : omp_get_max_threads();
  s->shard_alignment = shard_sink_required_shard_alignment(sink);
  s->shard_sink = sink;

  // CPU codec alignment is 1 (no nvcomp alignment needed).
  if (compute_stream_layouts(
        config, 1, compress_cpu_max_output_size, s->shard_alignment, &s->cl))
    goto Fail;

  s->layout = s->cl.layouts[0];
  s->levels = s->cl.levels;

  const uint32_t K = s->cl.epochs_per_batch;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  const uint64_t total_chunks = s->levels.total_chunks;
  const size_t chunk_stride_bytes = s->layout.chunk_stride * bytes_per_element;
  const size_t max_out = s->cl.max_output_size;

  // Chunk pool: K epochs' worth across all levels.
  s->chunk_pool = calloc((uint64_t)K * total_chunks, chunk_stride_bytes);
  CHECK(Fail, s->chunk_pool);

  s->pool_fully_covered = (s->layout.chunk_stride == s->layout.chunk_elements);

  // Compressed output buffer (K epochs).
  s->compressed = malloc((uint64_t)K * total_chunks * max_out);
  CHECK(Fail, s->compressed);

  s->comp_sizes = (size_t*)calloc((uint64_t)K * total_chunks, sizeof(size_t));
  CHECK(Fail, s->comp_sizes);

  // Per-level shard + aggregate state.
  {
    uint64_t max_batch_C = 0;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct level_layout_info* li = &s->cl.per_level[lv];
      s->agg_layout[lv] = li->agg_layout;
      const struct aggregate_layout* agg = &s->agg_layout[lv];

      uint32_t K_l = li->batch_active_count;
      s->batch_active_count[lv] = K_l;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = agg->covering_count;
      uint64_t M_lv = agg->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      // Per-level aggregate output buffers (batch-scaled).
      size_t data_lv = agg_pool_bytes((uint64_t)slot_count * M_lv,
                                      agg->max_comp_chunk_bytes,
                                      C_lv,
                                      agg->cps_inner,
                                      agg->page_size);
      {
        struct cpu_agg_slot* as = &s->agg_slots[lv];
        if (batch_C > 0) {
          as->offsets = (size_t*)malloc((batch_C + 1) * sizeof(size_t));
          as->chunk_sizes = (size_t*)calloc(batch_C, sizeof(size_t));
          CHECK(Fail, as->offsets && as->chunk_sizes);
        }
        if (data_lv > 0) {
          as->data = malloc(data_lv);
          as->data_capacity_bytes = data_lv;
          CHECK(Fail, as->data);
        }
      }

      // Batch aggregate LUTs (gather + perm).
      if (M_lv > 0) {
        uint64_t lut_len = (uint64_t)slot_count * M_lv;

        s->batch_gather[lv] = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
        s->batch_chunk_to_shard_map[lv] =
          (uint32_t*)malloc(lut_len * sizeof(uint32_t));
        CHECK(Fail, s->batch_gather[lv] && s->batch_chunk_to_shard_map[lv]);
      }

      CHECK(Fail, init_shard_state(&s->shard[lv], li) == 0);
    }

    // Shared permuted_sizes scratch (sized to max batch C).
    if (max_batch_C > 0) {
      s->shard_order_sizes = (size_t*)calloc(max_batch_C, sizeof(size_t));
      CHECK(Fail, s->shard_order_sizes);
    }
  }

  // LOD buffers (multiscale only).
  if (s->levels.enable_multiscale) {
    // Linear epoch buffer: input is accumulated here before LOD scatter.
    s->linear = calloc(s->layout.epoch_elements, bytes_per_element);
    CHECK(Fail, s->linear);

    uint64_t total_lod_elements =
      s->cl.plan.level_spans.ends[s->cl.plan.levels.nlod - 1];
    s->lod_values = calloc(total_lod_elements, bytes_per_element);
    CHECK(Fail, s->lod_values);

    // Append accumulator: total elements in levels 1+ (packed).
    // Uses the source dtype (not a wider accumulator) for integer mean —
    // overflow_safe_add_shift prevents overflow at the cost of rounding per
    // fold.
    if (s->cl.dims.append_downsample) {
      uint64_t append_total = 0;
      for (int lv = 1; lv < s->cl.plan.levels.nlod; ++lv)
        append_total += s->cl.plan.levels.level[lv].fixed_dims_count *
                        s->cl.plan.levels.level[lv].lod_nelem;
      if (append_total > 0) {
        s->append_accum = calloc(append_total, bytes_per_element);
        CHECK(Fail, s->append_accum);
      }
      memset(s->append_counts, 0, sizeof(s->append_counts));
    }

    // Allocate L0 scatter LUT + batch offsets.
    {
      const struct lod_plan* plan = &s->cl.plan;
      s->scatter_lut =
        (uint32_t*)malloc(plan->levels.level[0].lod_nelem * sizeof(uint32_t));
      CHECK(Fail, s->scatter_lut);

      s->scatter_fixed_dims_offsets =
        (uint64_t*)calloc(plan->fixed_dims_count, sizeof(uint64_t));
      CHECK(Fail, s->scatter_fixed_dims_offsets);
    }

    // Allocate morton-to-chunk LUTs and batch offsets for each level.
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct lod_plan* plan = &s->cl.plan;
      uint64_t lod_count = plan->levels.level[lv].lod_nelem;

      s->morton_lut[lv] = (uint32_t*)malloc(lod_count * sizeof(uint32_t));
      CHECK(Fail, s->morton_lut[lv]);

      s->lod_fixed_dims_offsets[lv] = (uint64_t*)calloc(
        plan->levels.level[lv].fixed_dims_count, sizeof(uint64_t));
      CHECK(Fail, s->lod_fixed_dims_offsets[lv]);
    }

    // CSR reduce LUTs: one per level transition (nlod-1 total).
    {
      const struct lod_plan* plan = &s->cl.plan;
      int ncsr = plan->levels.nlod - 1;
      if (ncsr > 0) {
        s->csrs = (struct reduce_csr*)calloc(ncsr, sizeof(struct reduce_csr));
        CHECK(Fail, s->csrs);
        for (int l = 0; l < ncsr; ++l) {
          const struct level_dims* src_ld = &plan->levels.level[l];
          const struct level_dims* dst_ld = &plan->levels.level[l + 1];
          uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
          uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;
          CHECK(Fail, reduce_csr_alloc(&s->csrs[l], src_total, dst_total) == 0);
          CHECK(Fail, reduce_csr_build(&s->csrs[l], plan, l) == 0);
        }
      }
    }
  }

  // Fill all LUTs.
  {
    struct lut_targets luts = {
      .scatter_lut = s->scatter_lut,
      .scatter_fixed_dims_offsets = s->scatter_fixed_dims_offsets,
    };
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      luts.batch_gather[lv] = s->batch_gather[lv];
      luts.batch_chunk_to_shard_map[lv] = s->batch_chunk_to_shard_map[lv];
      luts.morton_lut[lv] = s->morton_lut[lv];
      luts.lod_fixed_dims_offsets[lv] = s->lod_fixed_dims_offsets[lv];
    }
    cpu_pipeline_compute_luts(&s->cl,
                              &s->levels,
                              s->batch_active_count,
                              s->agg_layout,
                              s->nthreads,
                              &luts);
  }

  // Metrics.
  s->metrics.memcpy = mk_stream_metric("memcpy");
  s->metrics.scatter = mk_stream_metric("scatter");
  s->metrics.compress = mk_stream_metric("compress");
  s->metrics.aggregate = mk_stream_metric("aggregate");
  s->metrics.sink = mk_stream_metric("sink");
  // GPU-only stall metrics: named but never populated on CPU path.
  s->metrics.flush_stall = mk_stream_metric("flush_stall");
  s->metrics.kick_sync_stall = mk_stream_metric("kick_sync");
  s->metrics.io_fence_stall = mk_stream_metric("io_fence");
  s->metrics.backpressure = mk_stream_metric("backpressure");
  if (s->levels.enable_multiscale) {
    s->metrics.lod_gather = mk_stream_metric("lod_gather");
    s->metrics.lod_reduce = mk_stream_metric("lod_reduce");
    s->metrics.lod_append_fold = mk_stream_metric("lod_append_fold");
    s->metrics.lod_morton_chunk = mk_stream_metric("lod_morton");
  }

  // Precompute max_cursor so cpu_append doesn't recompute each call.
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&s->cl.dims);
    if (dims[0].size > 0) {
      s->max_cursor_elements = s->layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        s->max_cursor_elements *= ceildiv(dims[d].size, dims[d].chunk_size);
    } else {
      s->max_cursor_elements = 0;
    }
  }

  s->writer.append = cpu_append;
  s->writer.flush = cpu_flush_final;

  platform_toc(&s->metadata_update_clock);

  return s;

Fail:
  tile_stream_cpu_destroy(s);
  return NULL;
}

void
tile_stream_cpu_destroy(struct tile_stream_cpu* s)
{
  if (!s)
    return;

  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->shard[lv];
    if (ss->shards) {
      for (uint64_t si = 0; si < ss->shard_inner_count; ++si)
        free(ss->shards[si].index);
      free(ss->shards);
    }
    free(s->batch_gather[lv]);
    free(s->batch_chunk_to_shard_map[lv]);
    free(s->morton_lut[lv]);
    free(s->lod_fixed_dims_offsets[lv]);

    struct cpu_agg_slot* as = &s->agg_slots[lv];
    free(as->data);
    free(as->offsets);
    free(as->chunk_sizes);
  }
  free(s->shard_order_sizes);

  free(s->chunk_pool);
  free(s->compressed);
  free(s->comp_sizes);
  free(s->scatter_lut);
  free(s->scatter_fixed_dims_offsets);
  free(s->linear);
  free(s->lod_values);
  free(s->append_accum);

  if (s->csrs) {
    int ncsr = s->cl.plan.levels.nlod - 1;
    for (int l = 0; l < ncsr; ++l)
      reduce_csr_free(&s->csrs[l]);
    free(s->csrs);
  }

  computed_stream_layouts_free(&s->cl);
  free(s);
}

// ---- Accessors ----

struct stream_metrics
tile_stream_cpu_get_metrics(const struct tile_stream_cpu* s)
{
  return s->metrics;
}

const struct tile_stream_layout*
tile_stream_cpu_layout(const struct tile_stream_cpu* s)
{
  return &s->layout;
}

struct writer*
tile_stream_cpu_writer(struct tile_stream_cpu* s)
{
  return &s->writer;
}

uint64_t
tile_stream_cpu_cursor(const struct tile_stream_cpu* s)
{
  return s->cursor_elements;
}

// ---- Memory estimate ----

// Compute memory sizing from pre-computed layouts.
// Used by memory_estimate for reporting.
static void
compute_memory_info(const struct computed_stream_layouts* cl,
                    size_t bytes_per_element,
                    struct tile_stream_cpu_memory_info* info)
{
  const uint32_t K = cl->epochs_per_batch;
  const uint64_t total_chunks = cl->levels.total_chunks;
  const size_t chunk_stride_bytes =
    cl->layouts[0].chunk_stride * bytes_per_element;
  const size_t max_out = cl->max_output_size;

  info->chunk_pool_bytes = (uint64_t)K * total_chunks * chunk_stride_bytes;
  info->compressed_pool_bytes = (uint64_t)K * total_chunks * max_out;
  info->comp_sizes_bytes = (uint64_t)K * total_chunks * sizeof(size_t);

  // Aggregate: per-level perm + batch-scaled slots + LUTs + shared scratch.
  {
    size_t agg = 0;
    uint64_t max_batch_C = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      const struct level_layout_info* li = &cl->per_level[lv];
      const struct aggregate_layout* al = &li->agg_layout;
      uint32_t K_l = li->batch_active_count;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = al->covering_count;
      uint64_t M_lv = al->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      size_t data_lv = agg_pool_bytes((uint64_t)slot_count * M_lv,
                                      al->max_comp_chunk_bytes,
                                      C_lv,
                                      al->cps_inner,
                                      al->page_size);

      agg += data_lv +
             (batch_C + 1) * sizeof(size_t) + // offsets (batch-scaled)
             batch_C * sizeof(size_t);        // chunk_sizes (batch-scaled)

      // Batch LUTs (gather + perm).
      {
        uint64_t lut_len = (uint64_t)slot_count * M_lv;
        agg += 2 * lut_len * sizeof(uint32_t);
      }
    }
    if (max_batch_C > 0)
      agg += max_batch_C * sizeof(size_t); // shared permuted_sizes scratch
    info->aggregate_bytes = agg;
  }

  // LOD buffers (multiscale only).
  {
    size_t lod = 0;
    if (cl->levels.enable_multiscale) {
      lod += cl->layouts[0].epoch_elements * bytes_per_element; // linear
      uint64_t total_lod_elements =
        cl->plan.level_spans.ends[cl->plan.levels.nlod - 1];
      lod += total_lod_elements * bytes_per_element; // lod_values

      if (cl->dims.append_downsample) {
        uint64_t append_total = 0;
        for (int lv = 1; lv < cl->plan.levels.nlod; ++lv)
          append_total += cl->plan.levels.level[lv].fixed_dims_count *
                          cl->plan.levels.level[lv].lod_nelem;
        lod += append_total * bytes_per_element; // append_accum
      }

      lod +=
        cl->plan.levels.level[0].lod_nelem * sizeof(uint32_t); // scatter_lut
      lod += cl->plan.fixed_dims_count *
             sizeof(uint64_t); // scatter_fixed_dims_offsets

      for (int lv = 0; lv < cl->levels.nlod; ++lv) {
        lod +=
          cl->plan.levels.level[lv].lod_nelem * sizeof(uint32_t); // morton_lut
        lod += cl->plan.levels.level[lv].fixed_dims_count *
               sizeof(uint64_t); // fixed_dims_offsets
      }

      // Host CSR reduce LUTs (one per level transition).
      for (int l = 0; l < cl->plan.levels.nlod - 1; ++l) {
        const struct level_dims* src_ld = &cl->plan.levels.level[l];
        const struct level_dims* dst_ld = &cl->plan.levels.level[l + 1];
        uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
        uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;
        if (src_total == 0 || dst_total == 0)
          continue;
        lod += (dst_total + 1) * sizeof(uint64_t); // csrs[l].starts
        lod += src_total * sizeof(uint64_t);       // csrs[l].indices
      }
      lod += (size_t)(cl->plan.levels.nlod - 1) * sizeof(struct reduce_csr);
    }
    info->lod_bytes = lod;
  }

  // Shard state: active_shard arrays + index buffers.
  {
    size_t shards = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      const struct level_layout_info* li = &cl->per_level[lv];
      shards += li->shard_inner_count * sizeof(struct active_shard);
      shards += li->shard_inner_count * li->chunks_per_shard_total * 2 *
                sizeof(uint64_t);
    }
    info->shard_bytes = shards;
  }

  info->heap_bytes = sizeof(struct tile_stream_cpu) + info->chunk_pool_bytes +
                     info->compressed_pool_bytes + info->comp_sizes_bytes +
                     info->aggregate_bytes + info->lod_bytes +
                     info->shard_bytes;

  info->chunks_per_epoch = cl->layouts[0].chunks_per_epoch;
  info->total_chunks = total_chunks;
  info->max_output_size = max_out;
  info->nlod = cl->levels.nlod;
  info->epochs_per_batch = K;
}

int
tile_stream_cpu_memory_estimate(const struct tile_stream_configuration* config,
                                size_t shard_alignment,
                                struct tile_stream_cpu_memory_info* info)
{
  if (!info)
    return 1;

  memset(info, 0, sizeof(*info));

  struct computed_stream_layouts cl;
  if (compute_stream_layouts(
        config, 1, compress_cpu_max_output_size, shard_alignment, &cl))
    return 1;

  compute_memory_info(&cl, dtype_bpe(config->dtype), info);
  computed_stream_layouts_free(&cl);
  return 0;
}

int
tile_stream_cpu_advise_chunk_sizes(struct tile_stream_configuration* config,
                                   size_t target_chunk_bytes,
                                   const uint8_t* ratios,
                                   size_t budget_bytes,
                                   size_t shard_alignment)
{
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  if (bytes_per_element == 0 || budget_bytes == 0)
    return 1;

  for (size_t target = target_chunk_bytes; target >= bytes_per_element;
       target >>= 1) {
    dims_budget_chunk_bytes(
      config->dimensions, config->rank, target, bytes_per_element, ratios);
    struct tile_stream_cpu_memory_info mem;
    if (tile_stream_cpu_memory_estimate(config, shard_alignment, &mem))
      return 1;
    if (mem.heap_bytes <= budget_bytes)
      return 0;
  }
  return 1;
}

// ---- View builder ----

static struct cpu_stream_view
make_view(struct tile_stream_cpu* s)
{
  struct cpu_stream_view v = {
    .config = &s->config,
    .sink = s->shard_sink,
    .cl = &s->cl,
    .layout = &s->layout,
    .levels = &s->levels,
    .cursor_elements = &s->cursor_elements,
    .max_cursor_elements = s->max_cursor_elements,
    .batch_accumulated = &s->batch_accumulated,
    .batch_active_masks = s->batch_active_masks,
    .pool_fully_covered = s->pool_fully_covered,
    .shard = s->shard,
    .agg_layout = s->agg_layout,
    .batch_active_count = s->batch_active_count,
    .csrs = s->csrs,
    .append_accum = s->append_accum,
    .append_counts = s->append_counts,
    .io_done = s->io_done,
    .chunk_pool = s->chunk_pool,
    .chunk_pool_bytes = 0,
    .compressed = s->compressed,
    .comp_sizes = s->comp_sizes,
    .agg_slots = s->agg_slots,
    .shard_order_sizes = s->shard_order_sizes,
    .linear = s->linear,
    .lod_values = s->lod_values,
    .scatter_lut = s->scatter_lut,
    .scatter_fixed_dims_offsets = s->scatter_fixed_dims_offsets,
    .nthreads = s->nthreads,
    .shard_alignment = s->shard_alignment,
    .metrics = &s->metrics,
    .metadata_update_clock = &s->metadata_update_clock,
  };
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    v.batch_gather[lv] = s->batch_gather[lv];
    v.batch_chunk_to_shard_map[lv] = s->batch_chunk_to_shard_map[lv];
    v.morton_lut[lv] = s->morton_lut[lv];
    v.lod_fixed_dims_offsets[lv] = s->lod_fixed_dims_offsets[lv];
  }
  return v;
}

// ---- Writer callbacks ----

static struct writer_result
cpu_append(struct writer* self, struct slice input)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);

  if (s->flushed)
    return writer_finished_at(input.beg, input.end);

  struct cpu_stream_view v = make_view(s);
  return cpu_stream_append_body(&v, input);
}

static struct writer_result
cpu_flush_final(struct writer* self)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);
  struct cpu_stream_view v = make_view(s);
  struct writer_result r = cpu_stream_flush_body(&v);
  s->flushed = 1;
  return r;
}
