#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.engine.h"
#include "gpu/stream.flush.h"
#include "gpu/stream.ingest.h"
#include "gpu/stream.lod.h"

#include "defs.limits.h"
#include "gpu/prelude.cuda.h"
#include "multiarray.gpu.h"
#include "stream/config.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

// ---- Per-array descriptor ----
// Extends stream_context with mutable per-array state that is swapped
// into/out of the engine on array switch.

struct array_descriptor_gpu
{
  struct stream_context ctx;
  struct computed_stream_layouts cl; // owned, freed on destroy

  // Per-array LOD state (owns plan, layouts[], layout_gpu[], CSRs, append
  // accumulator device memory, and LOD LUTs — but NOT d_linear/d_morton/timing,
  // which are shared and owned by the engine).
  struct lod_state array_lod;

  // Mutable per-array state (saved/restored via bind/unbind)
  uint32_t batch_accumulated;
  int pools_current;
  struct output_slot_ledger output;
  struct flush_slot_gpu flush_slots[2];
  struct flush_handoff flush_pending_handoff[2];
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];
  uint32_t batch_active_count[LOD_MAX_LEVELS];

  // Per-array unified state. total_shards = sum_lv num_shards[lv]; tail
  // buffers contiguous over all shards in this array. Bind swaps these
  // into compress_agg_stage.{per_lod_agg_layouts, shard, shards}.
  uint64_t u_total_shards;
  uint32_t u_shards_begin[LOD_MAX_LEVELS];
  uint32_t u_n_shards[LOD_MAX_LEVELS];
  size_t u_page_size;
  size_t u_tail_carry_bytes;
  size_t* u_d_tail_bytes;
  CUdeviceptr u_d_tail_carry;
  size_t* u_h_tail_bytes;
  size_t* u_h_shard_capacity;
  struct shard_state u_shard[LOD_MAX_LEVELS];

  int flushed; // 1 once flush body has run for this array
};

// ---- Pool maxima (computed across all arrays) ----

struct pool_maxima
{
  size_t pool_bytes;
  size_t buffer_capacity;
  size_t compressed_bytes;
  uint64_t codec_batch;
  size_t chunk_bytes;
  uint32_t epochs_per_batch;
  int max_nlod;
  size_t max_output_size;
  size_t lod_linear_bytes; // max across arrays
  size_t lod_morton_bytes; // max across arrays
  int any_multiscale;      // 1 if any array uses multiscale

  // Unified-pipeline maxima (max across arrays).
  uint64_t u_max_total_batch_chunks;
  uint64_t u_max_total_batch_covering;
  size_t u_max_total_data_bytes;
  uint64_t u_max_total_shards;
};

// ---- Main struct ----

struct multiarray_tile_stream_gpu
{
  struct multiarray_writer writer;
  struct stream_engine engine;
  int n_arrays;
  int active; // -1 = none
  int max_nlod;
  struct array_descriptor_gpu* arrays;
};

// ---- Forward declarations ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data);
static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self);

// ---- Helpers ----

static inline size_t
max_sz(size_t a, size_t b)
{
  return a > b ? a : b;
}

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
  return a > b ? a : b;
}

static inline uint32_t
max_u32(uint32_t a, uint32_t b)
{
  return a > b ? a : b;
}

static int
init_output_ledger(struct output_slot_ledger* ledger,
                   const struct aggregate_slot* slot)
{
  struct output_slot_capacity capacity = {
    .data_bytes = slot->slot_capacity_bytes,
    .desc_entries = slot->slot_desc_capacity,
    .batch_records = slot->batches_per_slot_cap,
  };
  return output_slot_ledger_init(ledger, capacity) == OUTPUT_LEDGER_OK ? 0 : 1;
}

// ---- Bind / Unbind ----
// Copy per-array mutable state between descriptor and engine sub-structs.

// Quiesce the shared d2h delivery pipeline against the departing array
// before another array binds in. Without this, the next array would inherit
// stale fences from a different sink (deadlock — fences only retire on the
// sink that issued them) or reuse aggregate buffers the prior sink is still
// reading.
static void
drain_d2h_for_array(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  cuStreamSynchronize(e->streams.d2h);
  // Drain the unified slot's io_done fence with the departing array's sink
  // so the next-bound array doesn't wait on a fence that was issued by a
  // different sink.
  for (int fc = 0; fc < 2; ++fc) {
    struct aggregate_slot* agg = &e->compress_agg.output[fc];
    if (agg->io_done.seq > 0 && desc->ctx.sink->wait_fence)
      desc->ctx.sink->wait_fence(desc->ctx.sink, agg->io_done);
    agg->io_done.seq = 0;
  }
}

static void
bind_context(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  // Set batch K from the array's own config, not the engine max.
  // Shared buffers are sized to max K, but each array should fill/flush at
  // its own K so that batch_active_count and active_count agree.
  e->batch.epochs_per_batch = desc->cl.epochs_per_batch;
  e->batch.accumulated = desc->batch_accumulated;
  e->pools.current = desc->pools_current;
  e->flush.output = desc->output;
  for (int i = 0; i < 2; ++i) {
    e->flush.slot[i] = desc->flush_slots[i];
    e->flush.pending_handoff[i] = desc->flush_pending_handoff[i];
  }
  e->d2h_deliver.shard_alignment = desc->ctx.shard_alignment;

  // Unified-pipeline bind: swap per-array state into the engine. Engine
  // buffers (agg[2], d_batch_gather/perm, h_lut_* scratch,
  // d_*_offsets/_tps_group/_capacity) are sized to max and shared.
  e->compress_agg.nlod = (uint8_t)desc->ctx.levels.nlod;
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    e->compress_agg.per_lod_agg_layouts[lv] = desc->agg_layout[lv];
    e->compress_agg.shard[lv] = desc->u_shard[lv];
  }
  e->compress_agg.shards.total_shards = desc->u_total_shards;
  e->compress_agg.shards.page_size = desc->u_page_size;
  e->compress_agg.shards.tail_carry_bytes = desc->u_tail_carry_bytes;
  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    e->compress_agg.shards.shards_begin[lv] = desc->u_shards_begin[lv];
    e->compress_agg.shards.n_shards[lv] = desc->u_n_shards[lv];
  }
  e->compress_agg.shards.h_tail_bytes = desc->u_h_tail_bytes;
  e->compress_agg.shards.d_tail_bytes = desc->u_d_tail_bytes;
  e->compress_agg.shards.d_tail_carry = desc->u_d_tail_carry;
  // Per-array shard_capacity table is constant; re-upload on bind so the
  // device-side d_shard_capacity reflects the active array's shard sizes.
  // (h_base_offsets / h_tps_group / h_offsets_base are per-batch scratch,
  // refreshed by the kick.)
  if (desc->u_total_shards > 0 && desc->u_h_shard_capacity) {
    cuMemcpyHtoD((CUdeviceptr)e->compress_agg.shards.d_shard_capacity,
                 desc->u_h_shard_capacity,
                 desc->u_total_shards * sizeof(size_t));
  }
  // Invalidate the LUT cache: per-array layouts differ.
  e->compress_agg.lut_cache_valid = 0;

  // Per-array LOD state is now a single struct assignment.  Shared LOD
  // resources (d_linear, d_morton, timing) live in e->lod_shared and are
  // untouched by bind/unbind — the type system enforces this.
  e->lod = desc->array_lod;
}

static void
unbind_context(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  drain_d2h_for_array(e, desc);

  desc->batch_accumulated = e->batch.accumulated;
  desc->pools_current = e->pools.current;
  desc->output = e->flush.output;
  for (int i = 0; i < 2; ++i) {
    desc->flush_slots[i] = e->flush.slot[i];
    desc->flush_pending_handoff[i] = e->flush.pending_handoff[i];
  }
  // Snapshot per-array state back into the descriptor. shard_state mutates
  // over a batch (epoch_in_shard, shard_epoch, writer pointers); preserve it.
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv)
    desc->u_shard[lv] = e->compress_agg.shard[lv];
  // Clear engine-side per-array pointers so the next bind establishes them
  // unambiguously. We zero everything that bind_context fills in so a stale
  // pointer can't survive an unbind/destroy that isn't followed by a bind.
  e->compress_agg.shards.d_tail_bytes = NULL;
  e->compress_agg.shards.d_tail_carry = 0;
  e->compress_agg.shards.h_tail_bytes = NULL;
  e->compress_agg.shards.total_shards = 0;
  e->compress_agg.shards.tail_carry_bytes = 0;
  e->compress_agg.shards.page_size = 0;
  memset(e->compress_agg.shards.shards_begin,
         0,
         sizeof(e->compress_agg.shards.shards_begin));
  memset(e->compress_agg.shards.n_shards,
         0,
         sizeof(e->compress_agg.shards.n_shards));
  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    memset(&e->compress_agg.per_lod_agg_layouts[lv],
           0,
           sizeof(e->compress_agg.per_lod_agg_layouts[lv]));
  }
  e->compress_agg.nlod = 0;

  // Save per-array LOD state. counts[] tracks running append-accumulator
  // state across epochs; element_capacity is the fixed accumulator buffer
  // size set at init.
  if (desc->ctx.levels.enable_multiscale) {
    memcpy(desc->array_lod.append_accum.counts,
           e->lod.append_accum.counts,
           sizeof(desc->array_lod.append_accum.counts));
    desc->array_lod.append_accum.element_capacity =
      e->lod.append_accum.element_capacity;
  }
}

// ---- Per-array init ----

static int
init_array_descriptor(struct array_descriptor_gpu* desc,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct pool_maxima* mx)
{
  if (!codec_is_gpu_supported(config->codec.id))
    return 1;

  desc->ctx.config = *config;
  desc->ctx.sink = sink;
  desc->ctx.shard_alignment = shard_sink_required_shard_alignment(sink);

  if (compute_stream_layouts(config,
                             codec_alignment(config->codec.id),
                             codec_max_output_size,
                             desc->ctx.shard_alignment,
                             &desc->cl))
    return 1;

  desc->ctx.layout = desc->cl.layouts[0];
  desc->ctx.levels = desc->cl.levels;
  desc->ctx.dims = desc->cl.dims;

  // Initialize per-array LOD state: transfer plan from cl, copy layouts,
  // upload level layouts + LUTs + CSRs. Does NOT allocate d_linear/d_morton
  // or timing events — those are engine-owned shared resources.
  desc->array_lod.plan = desc->cl.plan;
  desc->cl.plan = (struct lod_plan){ 0 }; // ownership transferred
  for (int lv = 0; lv < desc->cl.levels.nlod; ++lv)
    desc->array_lod.layouts[lv] = desc->cl.layouts[lv];

  if (lod_state_init(&desc->array_lod, &desc->ctx.levels, &desc->ctx.config))
    return 1;

  // Alias L0 layout GPU pointers from array_lod into ctx (for scatter).
  desc->ctx.layout_gpu = desc->array_lod.layout_gpu[0];

  if (desc->ctx.levels.enable_multiscale && desc->ctx.dims.append_downsample) {
    if (lod_state_init_accumulators(&desc->array_lod, &desc->ctx.config))
      return 1;
  }

  const uint32_t K = desc->cl.epochs_per_batch;
  const size_t bpe = dtype_bpe(config->dtype);
  const uint64_t total_chunks = desc->ctx.levels.total_chunks;
  const uint64_t chunk_stride = desc->ctx.layout.chunk_stride;

  // Per-array, per-slot batch_active_masks. Bind/unbind copies the pointer
  // into the engine's flush slots; the storage lives here for the array's
  // lifetime.
  for (int fc = 0; fc < 2; ++fc) {
    desc->flush_slots[fc].batch_active_masks =
      (uint32_t*)calloc(K, sizeof(uint32_t));
    if (!desc->flush_slots[fc].batch_active_masks)
      return 1;
  }

  // total_element_limit: configured stream length (0 = unbounded)
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&desc->ctx.dims);
    if (dims[0].size > 0) {
      desc->ctx.total_element_limit = desc->ctx.layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        desc->ctx.total_element_limit *=
          ceildiv(dims[d].size, dims[d].chunk_size);
    }
  }

  // Buffer capacity (page-aligned)
  desc->ctx.config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // Update pool maxima
  mx->pool_bytes =
    max_sz(mx->pool_bytes, (size_t)K * total_chunks * chunk_stride * bpe);
  mx->buffer_capacity =
    max_sz(mx->buffer_capacity, desc->ctx.config.buffer_capacity_bytes);
  mx->compressed_bytes = max_sz(
    mx->compressed_bytes, (size_t)K * total_chunks * desc->cl.max_output_size);
  mx->codec_batch = max_u64(mx->codec_batch, (uint64_t)K * total_chunks);
  mx->chunk_bytes = max_sz(mx->chunk_bytes, chunk_stride * bpe);
  mx->epochs_per_batch = max_u32(mx->epochs_per_batch, K);
  mx->max_output_size = max_sz(mx->max_output_size, desc->cl.max_output_size);

  if (desc->ctx.levels.nlod > mx->max_nlod)
    mx->max_nlod = desc->ctx.levels.nlod;

  // LOD buffer sizes (for engine's shared d_linear / d_morton).
  if (desc->ctx.levels.enable_multiscale) {
    mx->any_multiscale = 1;
    size_t linear_bytes = desc->ctx.layout.epoch_elements * bpe;
    mx->lod_linear_bytes = max_sz(mx->lod_linear_bytes, linear_bytes);
    uint64_t total_vals = desc->array_lod.plan.level_spans
                            .ends[desc->array_lod.plan.levels.nlod - 1];
    size_t morton_bytes = total_vals * bpe;
    mx->lod_morton_bytes = max_sz(mx->lod_morton_bytes, morton_bytes);
  }

  // Unified-pipeline per-array sizing.
  desc->u_total_shards = 0;
  desc->u_page_size = desc->cl.per_level[0].agg_layout.page_size;
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    desc->u_shards_begin[lv] = (uint32_t)desc->u_total_shards;
    desc->u_n_shards[lv] =
      (uint32_t)desc->cl.per_level[lv].agg_layout.num_shards;
    desc->u_total_shards += desc->cl.per_level[lv].agg_layout.num_shards;
  }
  if (desc->u_total_shards > 0) {
    desc->u_h_shard_capacity =
      (size_t*)malloc(desc->u_total_shards * sizeof(size_t));
    desc->u_h_tail_bytes =
      (size_t*)calloc(desc->u_total_shards, sizeof(size_t));
    if (!desc->u_h_shard_capacity || !desc->u_h_tail_bytes)
      return 1;
    for (uint64_t i = 0; i < desc->u_total_shards; ++i) {
      for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
        if (i >= desc->u_shards_begin[lv] &&
            i < desc->u_shards_begin[lv] + desc->u_n_shards[lv]) {
          desc->u_h_shard_capacity[i] =
            desc->cl.per_level[lv].agg_layout.shard_capacity;
          break;
        }
      }
    }
    if (cuMemAlloc((CUdeviceptr*)&desc->u_d_tail_bytes,
                   desc->u_total_shards * sizeof(size_t)) != CUDA_SUCCESS)
      return 1;
    if (cuMemsetD8((CUdeviceptr)desc->u_d_tail_bytes,
                   0,
                   desc->u_total_shards * sizeof(size_t)) != CUDA_SUCCESS)
      return 1;
    if (desc->u_page_size > 0) {
      desc->u_tail_carry_bytes = desc->u_total_shards * desc->u_page_size;
      if (cuMemAlloc(&desc->u_d_tail_carry, desc->u_tail_carry_bytes) !=
          CUDA_SUCCESS)
        return 1;
      if (cuMemsetD8(desc->u_d_tail_carry, 0, desc->u_tail_carry_bytes) !=
          CUDA_SUCCESS)
        return 1;
    }
  }

  // Pool-maxima inputs for the unified pipeline. Compute this array's
  // max-batch layout and feed the maxima accumulator below.
  struct batch_aggregate_layout array_max_layout;
  {
    struct aggregate_layout per_lod_layouts[LOD_MAX_LEVELS];
    uint32_t per_lod_max[LOD_MAX_LEVELS] = { 0 };
    for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
      per_lod_layouts[lv] = desc->cl.per_level[lv].agg_layout;
      per_lod_max[lv] = desc->cl.per_level[lv].batch_active_count;
    }
    if (batch_aggregate_layout_init(&array_max_layout,
                                    per_lod_layouts,
                                    per_lod_max,
                                    (uint8_t)desc->ctx.levels.nlod,
                                    desc->u_page_size))
      return 1;
  }
  mx->u_max_total_batch_chunks =
    max_u64(mx->u_max_total_batch_chunks, array_max_layout.total_batch_chunks);
  mx->u_max_total_batch_covering = max_u64(
    mx->u_max_total_batch_covering, array_max_layout.total_batch_covering);
  mx->u_max_total_data_bytes =
    max_sz(mx->u_max_total_data_bytes, array_max_layout.total_data_bytes);
  mx->u_max_total_shards =
    max_u64(mx->u_max_total_shards, desc->u_total_shards);

  // Per-LOD shard_state + agg_layout snapshot.
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    const struct level_layout_info* li = &desc->cl.per_level[lv];
    desc->agg_layout[lv] = li->agg_layout;
    desc->batch_active_count[lv] = li->batch_active_count;
    if (init_shard_state(&desc->u_shard[lv], li))
      return 1;
  }

  return 0;
}

static void
destroy_array_descriptor(struct array_descriptor_gpu* desc)
{
  if (!desc)
    return;
  for (int fc = 0; fc < 2; ++fc) {
    free(desc->flush_slots[fc].batch_active_masks);
    desc->flush_slots[fc].batch_active_masks = NULL;
  }
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    shard_state_destroy(&desc->u_shard[lv]);
    aggregate_layout_destroy(&desc->agg_layout[lv]);
  }
  free(desc->u_h_shard_capacity);
  free(desc->u_h_tail_bytes);
  cu_mem_free((CUdeviceptr)desc->u_d_tail_bytes);
  cu_mem_free(desc->u_d_tail_carry);
  // array_lod owns everything except d_linear/d_morton/timing (which stay 0
  // in the per-array struct). ctx.layout_gpu aliases array_lod.layout_gpu[0]
  // and is freed via array_lod destroy.
  lod_state_destroy(&desc->array_lod);
  computed_stream_layouts_free(&desc->cl);
}

// ---- Shared resource allocation ----

static int
init_shared_resources(struct multiarray_tile_stream_gpu* ms,
                      const struct pool_maxima* mx)
{
  struct stream_engine* e = &ms->engine;

  // CUDA streams
  CU(Fail, cuStreamCreate(&e->streams.h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&e->streams.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&e->streams.compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&e->streams.d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i)
    CU(Fail, cuEventCreate(&e->pools.ready[i], CU_EVENT_DEFAULT));

  CHECK(Fail,
        ingest_init(&e->stage, mx->buffer_capacity, e->streams.compute) == 0);

  e->pool_bytes = mx->pool_bytes;
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&e->pools.buf[i], mx->pool_bytes));
    CU(Fail,
       cuMemsetD8Async(e->pools.buf[i], 0, mx->pool_bytes, e->streams.compute));
  }

  e->batch.epochs_per_batch = mx->epochs_per_batch;
  CU(Fail, cuEventCreate(&e->batch.pool_ready, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(e->batch.pool_ready, e->streams.compute));

  // Per-slot batch_active_masks live on each array descriptor (bind/unbind
  // copies them in). Stage scratch is shared and sized to the max K.
  e->compress_agg.pool_epochs_scratch =
    (uint32_t*)malloc((size_t)mx->epochs_per_batch * sizeof(uint32_t));
  CHECK(Fail, e->compress_agg.pool_epochs_scratch);

  CHECK(Fail,
        codec_init(&e->compress_agg.codec,
                   ms->arrays[0].ctx.config.codec.id,
                   mx->chunk_bytes,
                   mx->codec_batch) == 0);

  for (int fc = 0; fc < 2; ++fc) {
    size_t comp_sz = mx->codec_batch * e->compress_agg.codec.max_output_size;
    if (comp_sz > 0)
      CU(Fail, cuMemAlloc(&e->compress_agg.d_compressed[fc], comp_sz));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.d_measurement[fc],
                  sizeof(struct aggregate_append_measurement)));
    CU(Fail,
       cuMemHostAlloc((void**)&e->compress_agg.h_measurement[fc],
                      sizeof(struct aggregate_append_measurement),
                      0));
    memset((void*)e->compress_agg.h_measurement[fc],
           0,
           sizeof(struct aggregate_append_measurement));
    CU(Fail,
       cuEventCreate(&e->compress_agg.measurement_ready[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&e->compress_agg.t_compress_start[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&e->compress_agg.t_compress_end[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&e->compress_agg.t_aggregate_end[fc], CU_EVENT_DEFAULT));
  }

  CHECK(Fail, d2h_deliver_init(&e->d2h_deliver, 0, e->streams.compute) == 0);
  e->d2h_deliver.metrics = &e->metrics;

  // Unified-pipeline shared resources. Sized to maxima across arrays.
  e->compress_agg.max_total_batch_chunks = mx->u_max_total_batch_chunks;
  e->compress_agg.max_total_batch_covering = mx->u_max_total_batch_covering;
  e->compress_agg.max_total_data_bytes = mx->u_max_total_data_bytes;
  if (mx->u_max_total_batch_chunks > 0) {
    const uint64_t slot_chunk_cap =
      mx->u_max_total_batch_covering + (uint64_t)ms->max_nlod;
    // Multiarray binds arrays of any codec — including CODEC_NONE pass-through
    // whose slot data area is sized to one worst-case batch (W). Two batches
    // would not fit. Keep cap=1 until per-bind cap is plumbed.
    const uint32_t batches_per_slot_cap = 1;
    for (int fc = 0; fc < 2; ++fc) {
      CHECK(Fail,
            aggregate_batch_slot_init(&e->compress_agg.output[fc],
                                      slot_chunk_cap,
                                      mx->u_max_total_data_bytes,
                                      batches_per_slot_cap,
                                      mx->u_max_total_shards) == 0);
      CU(Fail,
         cuEventRecord(e->compress_agg.output[fc].ready, e->streams.compute));
      CU(Fail,
         cuEventRecord(e->compress_agg.output[fc].host_func_done,
                       e->streams.compute));
    }
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.d_write_desc,
                  sizeof(struct aggregate_write_desc)));
    CU(Fail,
       cuMemsetD8((CUdeviceptr)e->compress_agg.d_write_desc,
                  0,
                  sizeof(struct aggregate_write_desc)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.d_tail_sum_bytes,
                  sizeof(size_t)));
    CU(Fail,
       cuMemHostAlloc((void**)&e->compress_agg.h_write_desc,
                      sizeof(struct aggregate_write_desc),
                      0));
    memset((void*)e->compress_agg.h_write_desc,
           0,
           sizeof(struct aggregate_write_desc));
    for (int i = 0; i < 4; ++i) {
      e->compress_agg.cb_args_ring[i].slots[0] = &e->compress_agg.output[0];
      e->compress_agg.cb_args_ring[i].slots[1] = &e->compress_agg.output[1];
      e->compress_agg.cb_args_ring[i].h_write_desc =
        e->compress_agg.h_write_desc;
    }
    {
      const uint64_t temp_count =
        mx->u_max_total_batch_covering + (uint64_t)LOD_MAX_LEVELS;
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&e->compress_agg.d_temp_offsets,
                    temp_count * sizeof(size_t)));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&e->compress_agg.d_temp_perm_sizes,
                    temp_count * sizeof(size_t)));
    }
    CU(Fail,
       cuMemAlloc(&e->compress_agg.d_batch_gather,
                  mx->u_max_total_batch_chunks * sizeof(uint32_t)));
    CU(Fail,
       cuMemAlloc(&e->compress_agg.d_batch_perm,
                  mx->u_max_total_batch_chunks * sizeof(uint32_t)));
    e->compress_agg.h_lut_gather_scratch =
      (uint32_t*)malloc(mx->u_max_total_batch_chunks * sizeof(uint32_t));
    e->compress_agg.h_lut_perm_scratch =
      (uint32_t*)malloc(mx->u_max_total_batch_chunks * sizeof(uint32_t));
    CHECK(Fail,
          e->compress_agg.h_lut_gather_scratch &&
            e->compress_agg.h_lut_perm_scratch);
  }
  // Engine-shared per-shard scratch + device buffers, sized to max shards.
  if (mx->u_max_total_shards > 0) {
    e->compress_agg.shards.h_base_offsets =
      (size_t*)malloc(mx->u_max_total_shards * sizeof(size_t));
    e->compress_agg.shards.h_tps_group =
      (uint64_t*)malloc(mx->u_max_total_shards * sizeof(uint64_t));
    e->compress_agg.shards.h_offsets_base =
      (uint64_t*)malloc(mx->u_max_total_shards * sizeof(uint64_t));
    CHECK(Fail,
          e->compress_agg.shards.h_base_offsets &&
            e->compress_agg.shards.h_tps_group &&
            e->compress_agg.shards.h_offsets_base);
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.shards.d_base_offsets,
                  mx->u_max_total_shards * sizeof(size_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.shards.d_shard_capacity,
                  mx->u_max_total_shards * sizeof(size_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.shards.d_tps_group,
                  mx->u_max_total_shards * sizeof(uint64_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&e->compress_agg.shards.d_offsets_base,
                  mx->u_max_total_shards * sizeof(uint64_t)));
  }
  // Override the shared scratch sized at engine-init: the unified kick
  // needs LOD_MAX_LEVELS * max_K entries to carve per-LOD pool_epochs slices.
  // Allocate the cache to the same shape so the LUT-steady check can compare
  // each LOD's slice at a stable stride.
  free(e->compress_agg.pool_epochs_scratch);
  e->compress_agg.pool_epochs_stride = mx->epochs_per_batch;
  e->compress_agg.pool_epochs_scratch = (uint32_t*)malloc(
    (size_t)LOD_MAX_LEVELS * mx->epochs_per_batch * sizeof(uint32_t));
  e->compress_agg.cached_pool_epochs = (uint32_t*)malloc(
    (size_t)LOD_MAX_LEVELS * mx->epochs_per_batch * sizeof(uint32_t));
  CHECK(Fail,
        e->compress_agg.pool_epochs_scratch &&
          e->compress_agg.cached_pool_epochs);

  CU(Fail, cuEventRecord(e->pools.ready[0], e->streams.compute));
  CU(Fail, cuEventRecord(e->pools.ready[1], e->streams.compute));

  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail,
       cuEventRecord(e->compress_agg.t_compress_start[fc], e->streams.compute));
    CU(Fail,
       cuEventRecord(e->compress_agg.t_compress_end[fc], e->streams.compute));
    CU(Fail,
       cuEventRecord(e->compress_agg.t_aggregate_end[fc], e->streams.compute));
  }

  // Shared LOD buffers (sized to max across arrays). Only allocated if any
  // array uses multiscale.  The struct lod_shared_state / lod_state split
  // keeps engine-owned resources separate from per-array state, so bind/unbind
  // never touches these fields.
  if (mx->any_multiscale) {
    CHECK(Fail,
          lod_shared_state_init(&e->lod_shared,
                                mx->lod_linear_bytes,
                                mx->lod_morton_bytes,
                                e->streams.compute) == 0);
  }

  CU(Fail, cuStreamSynchronize(e->streams.compute));

  return 0;

Fail:
  return 1;
}

// ---- Array switching ----

static int
switch_to_array(struct multiarray_tile_stream_gpu* ms, int array_index)
{
  struct stream_engine* e = &ms->engine;

  if (ms->active >= 0) {
    struct array_descriptor_gpu* departing = &ms->arrays[ms->active];

    // Reject switch mid-epoch
    if (departing->ctx.cursor_elements % departing->ctx.layout.epoch_elements !=
        0)
      return multiarray_writer_not_flushable;

    // Flush departing array's accumulated batch. With sync_flush=1,
    // stream_flush_body uses the synchronous path (no pool swap or
    // pending state), so it's safe to call during switch.
    if (e->batch.accumulated > 0) {
      struct writer_result r = flush_accumulated_sync(e, &departing->ctx);
      if (r.error)
        return multiarray_writer_fail;
    }

    unbind_context(e, departing);
  }

  ms->active = array_index;
  bind_context(e, &ms->arrays[array_index]);

  // Zero both pools for the incoming array. This is the correctness-critical
  // zero: it ensures no stale data from the departing array leaks into the
  // incoming array's scatter. (flush_accumulate_epoch's sync path also zeros
  // the per-array portion of the current pool as an optimization for the
  // common batch-boundary case, but that only covers one pool and only the
  // per-array size — this full zero covers both pools at the max size.)
  for (int i = 0; i < 2; ++i)
    CU(Fail,
       cuMemsetD8Async(e->pools.buf[i], 0, e->pool_bytes, e->streams.compute));

  return 0;

Fail:
  return multiarray_writer_fail;
}

// ---- Writer: update ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);

  if (array_index < 0 || array_index >= ms->n_arrays)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_fail,
      .rest = data,
    };

  struct array_descriptor_gpu* desc = &ms->arrays[array_index];

  // Switch arrays if needed
  if (array_index != ms->active) {
    int err = switch_to_array(ms, array_index);
    if (err)
      return (struct multiarray_writer_result){ .error = err, .rest = data };
  }

  struct writer_result r = stream_append_body(&ms->engine, &desc->ctx, data);
  if (desc->flushed && r.rest.beg != data.beg)
    desc->flushed = 0;

  // `writer_finished` here means "stream is at capacity (total_element_limit)";
  // finalization happens on explicit `flush()` or on destroy, not here.
  return (struct multiarray_writer_result){
    .error = r.error,
    .rest = r.rest,
  };
}

// ---- Writer: flush ----

static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);

  // Save current array's state
  if (ms->active >= 0)
    unbind_context(&ms->engine, &ms->arrays[ms->active]);

  // Flush each array that has data
  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    // Idempotency: a redundant flush with no intervening updates re-finalizes
    // already-closed sinks. The flag is reset by update_impl when new data
    // arrives.
    if (desc->flushed)
      continue;
    if (desc->ctx.cursor_elements == 0 && desc->batch_accumulated == 0) {
      desc->flushed = 1;
      continue;
    }

    ms->active = a;
    bind_context(&ms->engine, desc);

    struct writer_result r = stream_flush_body(&ms->engine, &desc->ctx);
    if (r.error)
      goto Error;

    unbind_context(&ms->engine, desc);
    desc->flushed = 1;
  }

  ms->active = -1;

  // Each array's stream_flush_body already drained its sink as a commit
  // point; no additional drain needed here.

  return (struct multiarray_writer_result){ .error = multiarray_writer_ok };

Error:
  if (ms->active >= 0)
    unbind_context(&ms->engine, &ms->arrays[ms->active]);
  ms->active = -1;
  return (struct multiarray_writer_result){ .error = multiarray_writer_fail };
}

// ---- Create / Destroy ----

static void
sync_all(struct gpu_streams* streams)
{
  if (streams->h2d)
    cuStreamSynchronize(streams->h2d);
  if (streams->compute)
    cuStreamSynchronize(streams->compute);
  if (streams->compress)
    cuStreamSynchronize(streams->compress);
  if (streams->d2h)
    cuStreamSynchronize(streams->d2h);
}

void
multiarray_tile_stream_gpu_destroy(struct multiarray_tile_stream_gpu* ms)
{
  if (!ms)
    return;

  // Auto-finalize any unflushed arrays so destroy is a safe commit point
  // for callers that didn't explicitly flush. Errors are logged but not
  // propagated — destroy returns void.
  {
    struct multiarray_writer_result r = flush_impl(&ms->writer);
    if (r.error)
      log_error("GPU multiarray auto-flush failed during destroy");
  }

  sync_all(&ms->engine.streams);

  if (ms->arrays) {
    for (int a = 0; a < ms->n_arrays; ++a)
      destroy_array_descriptor(&ms->arrays[a]);
    free(ms->arrays);
  }

  struct stream_engine* e = &ms->engine;

  cu_event_destroy(e->batch.pool_ready);

  d2h_deliver_destroy(&e->d2h_deliver);

  codec_free(&e->compress_agg.codec);
  free(e->compress_agg.pool_epochs_scratch);
  free(e->compress_agg.cached_pool_epochs);
  e->compress_agg.pool_epochs_scratch = NULL;
  e->compress_agg.cached_pool_epochs = NULL;
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(e->compress_agg.d_compressed[fc]);
    cu_mem_free((CUdeviceptr)e->compress_agg.d_measurement[fc]);
    if (e->compress_agg.h_measurement[fc])
      cuMemFreeHost((void*)e->compress_agg.h_measurement[fc]);
    cu_event_destroy(e->compress_agg.measurement_ready[fc]);
    cu_event_destroy(e->compress_agg.t_compress_start[fc]);
    cu_event_destroy(e->compress_agg.t_compress_end[fc]);
    cu_event_destroy(e->compress_agg.t_aggregate_end[fc]);
  }

  // Unified-pipeline shared resources (allocated in init_shared_resources).
  for (int fc = 0; fc < 2; ++fc)
    aggregate_slot_destroy(&e->compress_agg.output[fc]);
  cu_mem_free((CUdeviceptr)e->compress_agg.d_write_desc);
  cu_mem_free((CUdeviceptr)e->compress_agg.d_tail_sum_bytes);
  if (e->compress_agg.h_write_desc)
    cuMemFreeHost((void*)e->compress_agg.h_write_desc);
  cu_mem_free((CUdeviceptr)e->compress_agg.d_temp_offsets);
  cu_mem_free((CUdeviceptr)e->compress_agg.d_temp_perm_sizes);
  cu_mem_free(e->compress_agg.d_batch_gather);
  cu_mem_free(e->compress_agg.d_batch_perm);
  free(e->compress_agg.h_lut_gather_scratch);
  free(e->compress_agg.h_lut_perm_scratch);
  free(e->compress_agg.shards.h_base_offsets);
  free(e->compress_agg.shards.h_tps_group);
  free(e->compress_agg.shards.h_offsets_base);
  cu_mem_free((CUdeviceptr)e->compress_agg.shards.d_base_offsets);
  cu_mem_free((CUdeviceptr)e->compress_agg.shards.d_shard_capacity);
  cu_mem_free((CUdeviceptr)e->compress_agg.shards.d_tps_group);
  cu_mem_free((CUdeviceptr)e->compress_agg.shards.d_offsets_base);
  memset(&e->compress_agg.shards, 0, sizeof(e->compress_agg.shards));

  // The per-array fields of e->lod are views of the last-bound descriptor
  // (freed above by destroy_array_descriptor).  Engine-owned shared LOD
  // resources live in e->lod_shared.
  lod_shared_state_destroy(&e->lod_shared);

  for (int i = 0; i < 2; ++i) {
    cu_mem_free(e->pools.buf[i]);
    cu_event_destroy(e->pools.ready[i]);
  }

  ingest_destroy(&e->stage);

  cu_stream_destroy(e->streams.h2d);
  cu_stream_destroy(e->streams.compute);
  cu_stream_destroy(e->streams.compress);
  cu_stream_destroy(e->streams.d2h);

  free(ms);
}

struct multiarray_tile_stream_gpu*
multiarray_tile_stream_gpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[],
  int enable_metrics)
{
  // enable_metrics is ignored: CUDA events are recorded for stream sync
  // regardless, so metrics collection has no meaningful opt-out on the GPU
  // path. See multiarray.gpu.h.
  (void)enable_metrics;

  if (n_arrays <= 0)
    return NULL;

  struct multiarray_tile_stream_gpu* ms =
    (struct multiarray_tile_stream_gpu*)calloc(1, sizeof(*ms));
  if (!ms)
    return NULL;

  ms->n_arrays = n_arrays;
  ms->active = -1;
  ms->writer.update = update_impl;
  ms->writer.flush = flush_impl;

  ms->arrays = (struct array_descriptor_gpu*)calloc(
    n_arrays, sizeof(struct array_descriptor_gpu));
  CHECK(Fail, ms->arrays);

  // Phase 1: compute per-array layouts and pool maxima
  struct pool_maxima mx;
  memset(&mx, 0, sizeof(mx));

  for (int a = 0; a < n_arrays; ++a)
    CHECK(Fail,
          init_array_descriptor(&ms->arrays[a], &configs[a], sinks[a], &mx) ==
            0);

  ms->max_nlod = mx.max_nlod;

  // Validate: all arrays must use the same codec (shared codec instance).
  for (int a = 1; a < n_arrays; ++a) {
    if (ms->arrays[a].ctx.config.codec.id !=
        ms->arrays[0].ctx.config.codec.id) {
      log_error("GPU multiarray: all arrays must use the same codec");
      goto Fail;
    }
  }

  // Phase 2: upload per-array aggregate layouts to GPU
  for (int a = 0; a < n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv)
      CHECK(Fail, aggregate_layout_upload(&desc->agg_layout[lv]) == 0);
  }

  // Phase 3: allocate shared GPU resources
  // (Per-array L0 layout_gpu is aliased from array_lod.layout_gpu[0], which
  // was uploaded during lod_state_init in init_array_descriptor.)
  CHECK(Fail, init_shared_resources(ms, &mx) == 0);
  for (int a = 0; a < n_arrays; ++a)
    CHECK(Fail,
          init_output_ledger(&ms->arrays[a].output,
                             &ms->engine.compress_agg.output[0]) == 0);

  // Use synchronous flush path — the double-buffered pipeline doesn't
  // compose across array switches.
  ms->engine.sync_flush = 1;

  // Label scatter as "Copy" only when every array uses multiscale (matches
  // single-array GPU).  When any array is non-multiscale, the scatter kernel
  // runs directly into the chunk pool, so keep the generic label.
  int all_multiscale = 1;
  for (int a = 0; a < n_arrays; ++a) {
    if (!ms->arrays[a].ctx.levels.enable_multiscale) {
      all_multiscale = 0;
      break;
    }
  }
  ms->engine.metrics = stream_engine_init_metrics(all_multiscale);
  ms->engine.metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&ms->engine.metadata_update_clock);

  return ms;

Fail:
  multiarray_tile_stream_gpu_destroy(ms);
  return NULL;
}

// ---- Accessors ----

struct multiarray_writer*
multiarray_tile_stream_gpu_writer(struct multiarray_tile_stream_gpu* ms)
{
  return &ms->writer;
}

struct stream_metrics
multiarray_tile_stream_gpu_get_metrics(
  const struct multiarray_tile_stream_gpu* ms)
{
  return ms->engine.metrics;
}
