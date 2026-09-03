#include "gpu/ordering.h"

#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/metric.h"

#include <assert.h>
#include <string.h>

static const struct gpu_edge_desc DESC[GPU_EDGE_COUNT] = {
  [GPU_EDGE_STAGING_SCATTER_DONE] = { "staging_scatter_done",
                                      GPU_STREAM_COMPUTE,
                                      GPU_STREAM_NONE,
                                      GPU_STREAM_H2D,
                                      "staging d_in slot reuse",
                                      GPU_EDGE_EVENT,
                                      1,
                                      1,
                                      -1,
                                      0,
                                      GPU_STREAM_NONE },
  [GPU_EDGE_STAGING_H2D_DONE] = { "staging_h2d_done",
                                  GPU_STREAM_H2D,
                                  GPU_STREAM_NONE,
                                  GPU_STREAM_COMPUTE,
                                  "staging d_in contents",
                                  GPU_EDGE_EVENT,
                                  1,
                                  1,
                                  -1,
                                  0,
                                  GPU_STREAM_NONE },
  [GPU_EDGE_STAGING_FREE] = { "staging_free",
                              GPU_STREAM_H2D,
                              GPU_STREAM_NONE,
                              GPU_STREAM_HOST,
                              "staging h_in refill",
                              GPU_EDGE_EVENT,
                              1,
                              1,
                              GPU_EDGE_STAGING_H2D_DONE,
                              0,
                              GPU_STREAM_NONE },
  [GPU_EDGE_POOL_FILLED] = { "pool_filled",
                             GPU_STREAM_COMPUTE,
                             GPU_STREAM_NONE,
                             GPU_STREAM_COMPRESS,
                             "chunk-pool batch contents",
                             GPU_EDGE_EVENT,
                             0,
                             1,
                             -1,
                             0,
                             GPU_STREAM_NONE },
  [GPU_EDGE_LOD_DONE] = { "lod_done",
                          GPU_STREAM_COMPUTE,
                          GPU_STREAM_NONE,
                          GPU_STREAM_COMPRESS,
                          "LOD chunks in pool",
                          GPU_EDGE_EVENT,
                          1,
                          1,
                          -1,
                          1,
                          GPU_STREAM_NONE },
  [GPU_EDGE_AGG_DONE] = { "agg_done",
                          GPU_STREAM_COMPRESS,
                          GPU_STREAM_NONE,
                          GPU_STREAM_D2H,
                          "aggregate slot outputs",
                          GPU_EDGE_EVENT,
                          1,
                          1,
                          -1,
                          0,
                          GPU_STREAM_PAYLOAD_COPY },
  [GPU_EDGE_POOL_CONSUMED] = { "pool_consumed",
                               GPU_STREAM_COMPRESS,
                               GPU_STREAM_NONE,
                               GPU_STREAM_COMPUTE,
                               "chunk pool buf[fc] reuse",
                               GPU_EDGE_EVENT,
                               1,
                               1,
                               GPU_EDGE_AGG_DONE,
                               0,
                               GPU_STREAM_NONE },
  [GPU_EDGE_SLOT_COPY_DONE] = { "slot_copy_done",
                                GPU_STREAM_D2H,
                                GPU_STREAM_PAYLOAD_COPY,
                                GPU_STREAM_COMPRESS,
                                "aggregate slot agg[fc] reuse",
                                GPU_EDGE_EVENT,
                                1,
                                1,
                                -1,
                                0,
                                GPU_STREAM_NONE },
  [GPU_EDGE_D2H_DONE] = { "d2h_done",
                          GPU_STREAM_D2H,
                          GPU_STREAM_PAYLOAD_COPY,
                          GPU_STREAM_HOST,
                          "leased host output stable for sink",
                          GPU_EDGE_EVENT,
                          1,
                          1,
                          GPU_EDGE_SLOT_COPY_DONE,
                          0,
                          GPU_STREAM_NONE },
  [GPU_EDGE_CHUNK_INDEX_READY] = { "chunk_index_ready",
                                   GPU_STREAM_D2H,
                                   GPU_STREAM_NONE,
                                   GPU_STREAM_HOST,
                                   "h_offsets/h_permuted_sizes",
                                   GPU_EDGE_EVENT,
                                   1,
                                   1,
                                   -1,
                                   0,
                                   GPU_STREAM_NONE },
  [GPU_EDGE_DELIVER_BEFORE_REKICK] = { "deliver_before_rekick",
                                       GPU_STREAM_HOST,
                                       GPU_STREAM_NONE,
                                       GPU_STREAM_HOST,
                                       "pending handoff + aggregate inputs",
                                       GPU_EDGE_HOST_RULE,
                                       1,
                                       0,
                                       -1,
                                       0,
                                       GPU_STREAM_NONE },
  [GPU_EDGE_DELIVER_OLDEST_FIRST] = { "deliver_oldest_first",
                                      GPU_STREAM_HOST,
                                      GPU_STREAM_NONE,
                                      GPU_STREAM_HOST,
                                      "batch delivery order",
                                      GPU_EDGE_HOST_RULE,
                                      1,
                                      0,
                                      -1,
                                      0,
                                      GPU_STREAM_NONE },
};

const struct gpu_edge_desc*
gpu_edge_describe(enum gpu_edge e)
{
  return &DESC[e];
}

static int
n_inst(enum gpu_edge e)
{
  return DESC[e].per_fc ? 2 : 1;
}

// Owner edge whose event backs `e`.
static enum gpu_edge
owner_of(enum gpu_edge e)
{
  return DESC[e].alias_of >= 0 ? (enum gpu_edge)DESC[e].alias_of : e;
}

#ifndef NDEBUG
static void
check_stream(const struct gpu_ordering* ord,
             enum gpu_edge e,
             enum gpu_stream_id want,
             enum gpu_stream_id want_alt,
             CUstream got,
             const char* role)
{
  if (want != GPU_STREAM_NONE && ord->streams[want] &&
      ord->streams[want] == got)
    return;
  if (want_alt != GPU_STREAM_NONE && ord->streams[want_alt] &&
      ord->streams[want_alt] == got)
    return;
  // Unregistered declared stream(s): cannot check (test harnesses).
  if ((want == GPU_STREAM_NONE || !ord->streams[want]) &&
      (want_alt == GPU_STREAM_NONE || !ord->streams[want_alt]))
    return;
  log_error(
    "gpu_ordering: edge %s %s on undeclared stream", DESC[e].name, role);
  assert(!"gpu_ordering: stream does not match edge declaration");
}
#endif

int
gpu_ordering_init(struct gpu_ordering* ord, CUstream seed_stream)
{
  memset(ord, 0, sizeof(*ord));
  for (int e = 0; e < GPU_EDGE_COUNT; ++e) {
    const struct gpu_edge_desc* d = &DESC[e];
    if (d->kind != GPU_EDGE_EVENT || d->alias_of >= 0 || d->external)
      continue;
    for (int i = 0; i < n_inst((enum gpu_edge)e); ++i) {
      CU(Fail, cuEventCreate(&ord->edge[e].ev[i], CU_EVENT_DEFAULT));
      ord->edge[e].owned[i] = 1;
      if (d->seeded)
        CU(Fail, cuEventRecord(ord->edge[e].ev[i], seed_stream));
    }
  }
  return 0;

Fail:
  gpu_ordering_destroy(ord);
  return 1;
}

void
gpu_ordering_destroy(struct gpu_ordering* ord)
{
  if (!ord)
    return;
#ifndef NDEBUG
  for (int e = 0; e < GPU_EDGE_COUNT; ++e) {
    const struct gpu_edge_desc* d = &DESC[e];
    if (d->kind != GPU_EDGE_EVENT)
      continue;
    const struct gpu_edge_state* owner = &ord->edge[owner_of((enum gpu_edge)e)];
    for (int i = 0; i < n_inst((enum gpu_edge)e); ++i) {
      if (owner->records[i] > 0 && ord->edge[e].waits[i] == 0)
        log_warn("gpu_ordering: dead edge %s[%d]: %llu records, 0 waits",
                 d->name,
                 i,
                 (unsigned long long)owner->records[i]);
    }
  }
#endif
  for (int e = 0; e < GPU_EDGE_COUNT; ++e) {
    for (int i = 0; i < 2; ++i) {
      if (ord->edge[e].owned[i])
        cu_event_destroy(ord->edge[e].ev[i]);
      ord->edge[e].ev[i] = NULL;
      ord->edge[e].owned[i] = 0;
    }
  }
}

void
gpu_ordering_register_stream(struct gpu_ordering* ord,
                             enum gpu_stream_id id,
                             CUstream stream)
{
  ord->streams[id] = stream;
}

void
gpu_ordering_bind(struct gpu_ordering* ord, enum gpu_edge e, int i, CUevent ev)
{
  assert(DESC[e].external && DESC[e].alias_of < 0);
  assert(!ord->edge[e].owned[i]);
  ord->edge[e].ev[i] = ev;
}

CUevent
gpu_ordering_event(const struct gpu_ordering* ord, enum gpu_edge e, int i)
{
  return ord->edge[owner_of(e)].ev[i];
}

int
gpu_edge_record(struct gpu_ordering* ord,
                enum gpu_edge e,
                int i,
                CUstream stream)
{
  struct gpu_edge_state* st = &ord->edge[e];
#ifndef NDEBUG
  assert(DESC[e].kind == GPU_EDGE_EVENT && DESC[e].alias_of < 0);
  check_stream(
    ord, e, DESC[e].producer, DESC[e].producer_alt, stream, "record");
  st->records[i]++;
#endif
  CU(Error, cuEventRecord(st->ev[i], stream));
  return 0;

Error:
  return 1;
}

int
gpu_edge_wait(struct gpu_ordering* ord, enum gpu_edge e, int i, CUstream stream)
{
  CUevent ev = ord->edge[owner_of(e)].ev[i];
  if (!ev) {
    // Returning success here would silently drop an ordering rule; callers
    // must gate on the binding (e.g. lod_active), so fail in release too.
    log_error("gpu_ordering: wait on unbound edge %s[%d]", DESC[e].name, i);
    assert(!"gpu_ordering: wait on unbound edge");
    return 1;
  }
#ifndef NDEBUG
  assert(DESC[e].kind == GPU_EDGE_EVENT);
  check_stream(ord, e, DESC[e].consumer, DESC[e].consumer_alt, stream, "wait");
  // #141 class: a wait whose record only happens via a future host action.
  // Seeded edges start live; external edges are seeded by their owner.
  if (!DESC[owner_of(e)].seeded && !DESC[owner_of(e)].external)
    assert(ord->edge[owner_of(e)].records[i] > 0 &&
           "gpu_ordering: wait without live record");
  ord->edge[e].waits[i]++;
#endif
  CU(Error, cuStreamWaitEvent(stream, ev, 0));
  return 0;

Error:
  return 1;
}

static int
event_query_ready(CUevent ev, enum gpu_edge e, int i, int* ready)
{
  const CUresult r = cuEventQuery(ev);
  if (r == CUDA_SUCCESS) {
    *ready = 1;
    return 0;
  }
  if (r == CUDA_ERROR_DEINITIALIZED) {
    // Context torn down (shutdown); data needed by this poll site is already
    // on the host, so a clean exit is correct. Logged so the swallow is
    // observable outside teardown.
    log_debug("gpu_ordering: %s[%d] poll saw DEINITIALIZED", DESC[e].name, i);
    *ready = 1;
    return 0;
  }
  if (r == CUDA_ERROR_NOT_READY) {
    *ready = 0;
    return 0;
  }
  handle_curesult(LOG_ERROR, r, __FILE__, __LINE__, "cuEventQuery");
  return 1;
}

static void
metric_wait_begin(struct stream_metric* m,
                  struct platform_clock* clk,
                  int* timed)
{
  if (m && !*timed) {
    platform_toc(clk);
    *timed = 1;
  }
}

static void
metric_wait_end(struct stream_metric* m, struct platform_clock* clk, int timed)
{
  if (m && timed)
    accumulate_metric_ms(m, (float)(platform_toc(clk) * 1000.0), 0, 0);
}

int
gpu_edge_host_wait(struct gpu_ordering* ord, enum gpu_edge e, int i)
{
  CUevent ev = ord->edge[owner_of(e)].ev[i];
#ifndef NDEBUG
  assert(DESC[e].kind == GPU_EDGE_EVENT && DESC[e].consumer == GPU_STREAM_HOST);
  ord->edge[e].waits[i]++;
#endif
  struct stream_metric* m = ord->edge[e].stall;
  if (m)
    m->wait_calls++;
  struct platform_clock clk = { 0 };
  int timed = 0;
  for (;;) {
    int ready = 0;
    if (event_query_ready(ev, e, i, &ready))
      return 1;
    if (ready)
      break;
    metric_wait_begin(m, &clk, &timed);
    platform_sleep_ns(50000); // 50 us
  }
  // Only blocked polls become samples — a zero sample per call would turn
  // the stall rows into poll counts rather than stall time.
  metric_wait_end(m, &clk, timed);
  return 0;
}

int
gpu_edge_host_wait_split(struct gpu_ordering* ord,
                         enum gpu_edge e,
                         enum gpu_edge prerequisite,
                         int i,
                         struct stream_metric* before,
                         struct stream_metric* after)
{
  const CUevent ev = ord->edge[owner_of(e)].ev[i];
  const CUevent prerequisite_ev = ord->edge[owner_of(prerequisite)].ev[i];
#ifndef NDEBUG
  assert(DESC[e].kind == GPU_EDGE_EVENT && DESC[e].consumer == GPU_STREAM_HOST);
  assert(DESC[prerequisite].kind == GPU_EDGE_EVENT);
  ord->edge[e].waits[i]++;
#endif
  struct stream_metric* inclusive = ord->edge[e].stall;
  if (inclusive)
    inclusive->wait_calls++;
  if (before)
    before->wait_calls++;
  if (after)
    after->wait_calls++;

  struct platform_clock inclusive_clk = { 0 };
  struct platform_clock phase_clk = { 0 };
  int inclusive_timed = 0;
  int phase_timed = 0;
  int prerequisite_ready = 0;

  for (;;) {
    int ready = 0;
    if (event_query_ready(ev, e, i, &ready))
      return 1;
    if (ready)
      break;

    metric_wait_begin(inclusive, &inclusive_clk, &inclusive_timed);
    if (!prerequisite_ready) {
      if (event_query_ready(
            prerequisite_ev, prerequisite, i, &prerequisite_ready))
        return 1;
      if (prerequisite_ready) {
        metric_wait_end(before, &phase_clk, phase_timed);
        phase_timed = 0;
      } else {
        metric_wait_begin(before, &phase_clk, &phase_timed);
      }
    }
    if (prerequisite_ready)
      metric_wait_begin(after, &phase_clk, &phase_timed);
    platform_sleep_ns(50000); // 50 us
  }

  metric_wait_end(inclusive, &inclusive_clk, inclusive_timed);
  if (prerequisite_ready)
    metric_wait_end(after, &phase_clk, phase_timed);
  else
    metric_wait_end(before, &phase_clk, phase_timed);
  return 0;
}

void
gpu_ordering_attach_stall_metric(struct gpu_ordering* ord,
                                 enum gpu_edge e,
                                 struct stream_metric* m)
{
  ord->edge[e].stall = m;
}

#ifndef NDEBUG
void
gpu_edge_host_rule_check(struct gpu_ordering* ord,
                         enum gpu_edge e,
                         int cond,
                         const char* file,
                         int line)
{
  (void)ord;
  if (cond)
    return;
  log_log(
    LOG_ERROR, file, line, "gpu_ordering: host rule %s violated", DESC[e].name);
  assert(!"gpu_ordering: host rule violated");
}
#endif
