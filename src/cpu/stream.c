#include "cpu/stream.internal.h"
#include "stream/config.h"

#include "cpu/compress.h"
#include "cpu/compress_blosc.h"
#include "cpu/transpose.h"
#include "defs.limits.h"
#include "lod/lod_plan.h"
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
static struct cpu_stream_view
make_view(struct tile_stream_cpu* s);

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

  // Per-epoch mask + scratch buffers, sized to K.
  s->batch_active_masks = (uint32_t*)calloc(K, sizeof(uint32_t));
  CHECK(Fail, s->batch_active_masks);
  // Worst case is per-LOD pool_epochs of length K, packed together.
  s->pool_epochs_scratch =
    (uint32_t*)malloc((size_t)LOD_MAX_LEVELS * (size_t)K * sizeof(uint32_t));
  CHECK(Fail, s->pool_epochs_scratch);

  // Per-level shard + aggregate state.
  {
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct level_layout_info* li = &s->cl.per_level[lv];
      s->agg_layout[lv] = li->agg_layout;
      s->batch_active_count[lv] = li->batch_active_count;
      CHECK(Fail, init_shard_state(&s->shard[lv], li) == 0);
    }

    // Worst-case unified batch layout: every LOD active across every epoch
    // of its batch_active_count. Sizes the shared per-slot data buffer +
    // unified scratch arrays.
    {
      uint32_t worst[LOD_MAX_LEVELS];
      for (int lv = 0; lv < s->levels.nlod; ++lv) {
        uint32_t k = s->batch_active_count[lv];
        worst[lv] = k > 0 ? k : 1;
      }
      const size_t page = s->cl.per_level[0].agg_layout.page_size;
      CHECK(Fail,
            batch_aggregate_layout_init(&s->max_batch_layout,
                                        s->agg_layout,
                                        worst,
                                        (uint8_t)s->levels.nlod,
                                        page) == 0);
    }

    // Allocate the two double-buffered slots once each. Each slot owns a
    // page-aligned data buffer and unified scratch arrays. cap_data is
    // already page-aligned by batch_aggregate_layout_init.
    {
      const uint64_t cap_chunks = s->max_batch_layout.total_batch_chunks;
      const uint64_t cap_cov = s->max_batch_layout.total_batch_covering;
      const size_t cap_data = s->max_batch_layout.total_data_bytes;
      const size_t align = platform_page_alignment();
      for (int fc = 0; fc < 2; ++fc) {
        struct cpu_agg_slot* as = &s->agg_slots[fc];
        as->data_capacity_bytes = cap_data;
        if (cap_data > 0) {
          as->data = platform_aligned_alloc(align, cap_data);
          CHECK(Fail, as->data);
        }
        if (cap_chunks > 0) {
          as->perm = (uint32_t*)malloc(cap_chunks * sizeof(uint32_t));
          as->gather = (uint32_t*)malloc(cap_chunks * sizeof(uint32_t));
          CHECK(Fail, as->perm && as->gather);
        }
        if (cap_cov > 0) {
          // +LOD_MAX_LEVELS slack across all three for the per-LOD shift
          // applied by the unified aggregate.
          const uint64_t cov_alloc = cap_cov + LOD_MAX_LEVELS;
          as->permuted_sizes = (size_t*)calloc(cov_alloc, sizeof(size_t));
          as->offsets = (size_t*)malloc(cov_alloc * sizeof(size_t));
          as->chunk_sizes = (size_t*)calloc(cov_alloc, sizeof(size_t));
          CHECK(Fail, as->permuted_sizes && as->offsets && as->chunk_sizes);
        }
      }
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
      luts.morton_lut[lv] = s->morton_lut[lv];
      luts.lod_fixed_dims_offsets[lv] = s->lod_fixed_dims_offsets[lv];
    }
    cpu_pipeline_compute_luts(&s->cl, &s->levels, s->nthreads, &luts);
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

  // Precompute total_element_limit (configured stream length) so the body can
  // detect the at-capacity case without recomputing each call.
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&s->cl.dims);
    if (dims[0].size > 0) {
      s->total_element_limit = s->layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        s->total_element_limit *= ceildiv(dims[d].size, dims[d].chunk_size);
    } else {
      s->total_element_limit = 0;
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

  // Auto-finalize any unwritten data so destroy is a safe commit point for
  // callers that didn't explicitly flush. Errors here are swallowed — the
  // stream is tearing down, there's no one to report to.
  if (!s->flushed) {
    struct cpu_stream_view v = make_view(s);
    (void)cpu_stream_flush_body(&v);
    s->flushed = 1;
  }

  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->shard[lv];
    if (ss->shards) {
      for (uint64_t si = 0; si < ss->shard_inner_count; ++si)
        free(ss->shards[si].index);
      free(ss->shards);
    }
    free(s->morton_lut[lv]);
    free(s->lod_fixed_dims_offsets[lv]);
  }

  for (int fc = 0; fc < 2; ++fc) {
    struct cpu_agg_slot* as = &s->agg_slots[fc];
    platform_aligned_free(as->data);
    free(as->perm);
    free(as->gather);
    free(as->permuted_sizes);
    free(as->offsets);
    free(as->chunk_sizes);
  }

  free(s->chunk_pool);
  free(s->compressed);
  free(s->comp_sizes);
  free(s->batch_active_masks);
  free(s->pool_epochs_scratch);
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

  // Aggregate: unified per-batch double-buffered slots + shared shard-order
  // scratch + batch state. Mirrors the allocation in tile_stream_cpu_create.
  {
    size_t agg = 0;

    struct aggregate_layout per_lod[LOD_MAX_LEVELS];
    uint32_t worst[LOD_MAX_LEVELS];
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      per_lod[lv] = cl->per_level[lv].agg_layout;
      uint32_t k = cl->per_level[lv].batch_active_count;
      worst[lv] = k > 0 ? k : 1;
    }
    const size_t page = cl->per_level[0].agg_layout.page_size;
    struct batch_aggregate_layout layout;
    if (batch_aggregate_layout_init(
          &layout, per_lod, worst, (uint8_t)cl->levels.nlod, page) == 0) {
      const uint64_t cap_chunks = layout.total_batch_chunks;
      const uint64_t cap_cov = layout.total_batch_covering;
      const size_t cap_data = layout.total_data_bytes;
      const uint64_t cov_alloc = cap_cov + LOD_MAX_LEVELS;

      // Per slot: data + perm + gather + permuted_sizes + offsets +
      // chunk_sizes (the last three carry +LOD_MAX_LEVELS shift slack).
      const size_t per_slot = cap_data + cap_chunks * sizeof(uint32_t) + // perm
                              cap_chunks * sizeof(uint32_t) + // gather
                              cov_alloc * sizeof(size_t) +    // permuted_sizes
                              cov_alloc * sizeof(size_t) +    // offsets
                              cov_alloc * sizeof(size_t);     // chunk_sizes
      agg += 2 * per_slot;
    }

    // Batch state: per-epoch active mask + per-LOD pool_epochs scratch.
    agg += (size_t)K * sizeof(uint32_t);
    agg += (size_t)LOD_MAX_LEVELS * K * sizeof(uint32_t);

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
tile_stream_cpu_advise_layout(struct tile_stream_configuration* config,
                              size_t target_chunk_bytes,
                              size_t min_chunk_bytes,
                              const int* ratios,
                              size_t budget_bytes,
                              size_t min_shard_bytes,
                              uint32_t target_concurrent_shards,
                              uint32_t min_append_shards,
                              size_t shard_alignment,
                              struct advise_layout_diagnostic* diag)
{
  if (diag) {
    memset(diag, 0, sizeof(*diag));
    diag->budget_bytes = budget_bytes;
    diag->parts_limit = MAX_PARTS_PER_SHARD;
  }

  const size_t bytes_per_element = dtype_bpe(config->dtype);
  if (bytes_per_element == 0 || budget_bytes == 0) {
    if (diag)
      diag->reason = ADVISE_INVALID_CONFIG;
    return 1;
  }

  const uint32_t user_k = config->epochs_per_batch;
  const size_t floor =
    min_chunk_bytes > bytes_per_element ? min_chunk_bytes : bytes_per_element;
  if (diag)
    diag->floor_chunk_bytes = floor;

  enum advise_layout_reason last_reason = ADVISE_BUDGET_EXCEEDED;
  size_t last_chunk_bytes = 0;
  uint32_t last_k = 0;
  size_t last_heap_bytes = 0;
  uint64_t last_cps_total = 0;

  for (size_t target = target_chunk_bytes; target >= floor; target >>= 1) {
    // Phase 1: fit chunks + K to memory budget. Start with auto-derived K
    // (or user-supplied K if non-zero); if heap_bytes exceeds budget, halve K
    // and retry. User-supplied K is authoritative and isn't reduced.
    if (dims_budget_chunk_bytes(config->dimensions,
                                config->rank,
                                target,
                                bytes_per_element,
                                ratios)) {
      // Non-recoverable input (e.g. pinned dims exceed target at this step).
      // Halving target can only make it worse — bail.
      last_reason = ADVISE_CHUNK_BUDGET_INFEASIBLE;
      last_chunk_bytes = target;
      break;
    }

    uint64_t chunk_vol = 1;
    for (uint8_t d = 0; d < config->rank; ++d)
      chunk_vol *= config->dimensions[d].chunk_size;
    last_chunk_bytes = (size_t)(chunk_vol * bytes_per_element);

    config->epochs_per_batch = user_k;
    int fit = 0;
    for (;;) {
      struct tile_stream_cpu_memory_info mem;
      if (tile_stream_cpu_memory_estimate(config, shard_alignment, &mem)) {
        if (diag) {
          diag->reason = ADVISE_INVALID_CONFIG;
          diag->chunk_bytes = last_chunk_bytes;
          diag->epochs_per_batch = config->epochs_per_batch;
        }
        return 1;
      }
      last_k = mem.epochs_per_batch;
      last_heap_bytes = mem.heap_bytes;
      if (mem.heap_bytes <= budget_bytes) {
        config->epochs_per_batch = mem.epochs_per_batch;
        fit = 1;
        break;
      }
      last_reason = ADVISE_BUDGET_EXCEEDED;
      last_cps_total = 0;
      if (user_k || mem.epochs_per_batch <= 1)
        break;
      config->epochs_per_batch = mem.epochs_per_batch / 2;
    }
    if (!fit)
      continue;

    // Phase 2: shard geometry (parts budget + concurrency target + byte floor).
    // Return 1 = MIN_SHARD_TOO_SMALL (retryable by shrinking chunks);
    // return 2 = PARTS_LIMIT infeasible even with inner fully split — halving
    // chunks only grows inner_cps_prod, so bail immediately.
    int sg = dims_set_shard_geometry(config->dimensions,
                                     config->rank,
                                     min_shard_bytes,
                                     target_concurrent_shards,
                                     min_append_shards,
                                     bytes_per_element);
    if (sg == 1) {
      last_reason = ADVISE_MIN_SHARD_TOO_SMALL;
      last_cps_total = 0;
      continue;
    }
    if (sg == 2) {
      last_reason = ADVISE_PARTS_LIMIT_EXCEEDED;
      uint64_t cps_total = 1;
      for (uint8_t d = 0; d < config->rank; ++d) {
        uint64_t cps = config->dimensions[d].chunks_per_shard;
        if (cps == 0)
          cps = 1;
        cps_total *= cps;
      }
      last_cps_total = cps_total;
      break;
    }
    if (sg != 0) {
      last_reason = ADVISE_INVALID_CONFIG;
      break;
    }

    if (diag) {
      uint64_t cps_total = 1;
      for (uint8_t d = 0; d < config->rank; ++d) {
        uint64_t cps = config->dimensions[d].chunks_per_shard;
        if (cps == 0)
          cps = 1;
        cps_total *= cps;
      }
      uint8_t na = dims_n_append(config->dimensions, config->rank);
      uint64_t inner_shards_prod = 1;
      for (uint8_t d = na; d < config->rank; ++d) {
        uint64_t size = config->dimensions[d].size;
        uint64_t cs = config->dimensions[d].chunk_size;
        uint64_t cps = config->dimensions[d].chunks_per_shard;
        if (cs == 0 || cps == 0)
          continue;
        uint64_t n_chunks = (size + cs - 1) / cs;
        inner_shards_prod *= (n_chunks + cps - 1) / cps;
      }
      diag->reason = ADVISE_OK;
      diag->chunk_bytes = last_chunk_bytes;
      diag->epochs_per_batch = config->epochs_per_batch;
      diag->device_bytes = last_heap_bytes;
      diag->chunks_per_shard_total = cps_total;
      diag->actual_concurrent_shards = inner_shards_prod;
      diag->actual_shard_bytes = last_chunk_bytes * cps_total;
      diag->min_append_shards_overrode_min_shard_bytes =
        (min_append_shards > 1 && min_shard_bytes > 0) ? 1 : 0;
    }
    return 0;
  }

  config->epochs_per_batch = user_k;
  if (diag) {
    diag->reason = last_reason;
    diag->chunk_bytes = last_chunk_bytes;
    diag->epochs_per_batch = last_k;
    diag->device_bytes = last_heap_bytes;
    diag->chunks_per_shard_total = last_cps_total;
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
    .total_element_limit = s->total_element_limit,
    .batch_accumulated = &s->batch_accumulated,
    .batch_active_masks = s->batch_active_masks,
    .pool_epochs_scratch = s->pool_epochs_scratch,
    .pool_fully_covered = s->pool_fully_covered,
    .shard = s->shard,
    .agg_layout = s->agg_layout,
    .batch_active_count = s->batch_active_count,
    .csrs = s->csrs,
    .append_accum = s->append_accum,
    .append_counts = s->append_counts,
    .io_done = s->io_done,
    .agg_current = &s->agg_current,
    .chunk_pool = s->chunk_pool,
    .chunk_pool_bytes = 0,
    .compressed = s->compressed,
    .comp_sizes = s->comp_sizes,
    .agg_slots = s->agg_slots,
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

  struct cpu_stream_view v = make_view(s);
  struct writer_result r = cpu_stream_append_body(&v, input);
  if (s->flushed && r.rest.beg != input.beg)
    s->flushed = 0;
  return r;
}

static struct writer_result
cpu_flush_final(struct writer* self)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);
  // Idempotency: a redundant flush with no intervening appends re-finalizes
  // already-closed sinks (deadlock on Windows, wasted work elsewhere).
  // The flag is reset by cpu_append when new data arrives.
  if (s->flushed)
    return writer_ok();
  struct cpu_stream_view v = make_view(s);
  struct writer_result r = cpu_stream_flush_body(&v);
  if (r.error == 0)
    s->flushed = 1;
  return r;
}
