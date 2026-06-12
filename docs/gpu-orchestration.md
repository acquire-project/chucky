# GPU stream orchestration: target design and migration plan

Status: accepted direction, migration in progress (step 1).
Base: main + #142. Related: #140, #141, PR #142, `gpu-output-slot-ledger.md`.

## Why

Three findings from the 2026-06 investigation drive this:

1. **Ordering is implicit and has shipped two silent-corruption races.**
   Cross-stream rules live as `cuEventRecord`/`cuStreamWaitEvent` pairs spread
   across six files, plus host call-order assumptions ("drain fc before
   re-kicking fc", "deliver oldest-first") written nowhere. #140 was a missing
   pool-reuse edge; #141 was a dependency an event *cannot* express (wait on a
   future host action). Several events are recorded but never waited — dead
   edges are indistinguishable from load-bearing ones.
2. **Both races were resource-lifetime bugs, not stage-logic bugs.** A buffer
   generation was reused (pool re-zero) or read (tail state) while the
   previous reader/writer was still in flight. Stage code forgot an edge; the
   architecture let it.
3. **The L40 stage baseline says the next optimizations are structural.**
   Wall time is ~100% host/producer-side (ingest staging memcpy is the
   ×1.65–1.94 lever; D2H-free buys ≤×1.01 — see PR #139's retrospective).
   Decoupling data movement from orchestration must not become another race
   factory; ordering has to be auditable first.

## End state

Five concepts, four of which exist today in partial form:

### 1. Resources: slotted pools with generation-carried ordering

Every device-side buffer that cycles (staging slots, chunk-pool epochs,
compressed buffers, aggregate output slots, tail state) becomes a slot in a
pool with an acquire/release protocol:

- `acquire(pool, slot, stream)` — queues a wait on the slot's previous
  generation's release before handing out the pointer.
- `release(pool, slot, stream)` — records the generation's completion event.
- Host-published resources (tail state) release via generation counter +
  `cuStreamWaitValue64` instead of an event (the #142 mechanism), behind the
  same API.

A stage cannot observe a raw pointer without its synchronization. Forgetting
an edge stops being possible at the call site; #140 and #141 become
unwritable. Teardown is `pool_release_all` per pool — the #142 hang class
(F2) dies structurally.

### 2. Edges: one declared table (`src/gpu/ordering.{h,c}`)

The audit layer, and step 1 of the migration. Every cross-stream /
host-stream rule is a named entry:

```c
enum gpu_edge { EDGE_POOL_FILLED, EDGE_POOL_CONSUMED, EDGE_AGG_DONE,
                EDGE_SLOT_DRAINED, EDGE_TAIL_PUBLISHED, EDGE_STAGING_FREE,
                /* ... */ EDGE_COUNT };

struct gpu_edge_desc {
  const char*        name;
  enum gpu_stream_id producer, consumer;  // stream id or HOST
  const char*        guards;              // the resource this edge protects
  enum edge_kind     kind;                // EVENT | GEN_COUNTER | HOST_RULE
  uint8_t            per_fc;              // instanced per pipeline slot
};
```

Call sites go through `edge_record` / `edge_wait` / `edge_publish` /
`edge_release_all`. Debug builds assert: consumer stream matches the
declaration; no wait without a live record/publish (the #141 class); at
destroy, edges with records but zero waits over the run are flagged dead.
Timing-only events (`t_compress_start/end`, lod timing) are excluded or
tagged TIMING — metrics events must not masquerade as ordering. Host-rule
edges (drain-before-rekick, oldest-first delivery — the invariant #142's GEQ
gate relies on) get debug asserts even though no GPU primitive backs them.

Per-edge wait/stall accounting comes for free, closing the unmetered
staging-slot busy-poll gap found in the baseline (`stream.c` append loop).

When step 3 lands, most EVENT edges collapse into pool acquire/release; the
table remains as the registry the pools draw from and the audit surface.

### 3. Stages: config in, state owned, handoff out, edges declared

A stage is independently testable when its boundary is data:

- created from a small config; owns its state; no reaching into engine
  globals.
- inputs/outputs are explicit handoff structs (`flush_handoff` is the
  existing model) plus declared edges.
- any neighbor can be faked in one line (the `test_compress_agg` harness
  publish is the proven pattern — that property becomes a requirement).

Variations that change the *dependency shape* (memops-unsupported fallback
dropping pipeline depth, codec NONE skipping compress, page-aligned tail
carry) move out of intra-stage branches and into the schedule (below).

### 4. Scheduler: one owner for streams, depth, and fallbacks

A single module owns: stream creation, pipeline depth, which stages run for
the active configuration, and degraded schedules (e.g. depth-1 host-ordered
when stream memops are unavailable). Orchestration stays single-threaded —
the producer thread *decides*; it stops *moving bytes*:

### 5. Workers: data movement behind queues

Ingest staging copies and sink delivery become queue-fed workers. This is
the baseline's top lever (×1.65–1.94) and the most race-prone change on the
roadmap — it lands last, when every handoff it touches is declared (edges)
or carried (pools). CPU and GPU backends become the same orchestration over
different stage implementations; multiarray becomes composition (N pipelines
sharing pools) instead of the bind/copy checklist in
`src/multiarray/stream.gpu.c`.

## Current edge inventory (2026-06 audits)

Known load-bearing edges (full table with file:line is step 1's first
deliverable; counts below from grep of main + #142):

| edge (name to assign)   | producer → consumer            | guards            | kind        |
|-------------------------|--------------------------------|-------------------|-------------|
| pool_filled (`pool_ready`)       | h2d/compute → compress | chunk pool epoch  | EVENT       |
| pool_consumed (`t_aggregate_end`)| compress → compute     | chunk pool epoch  | EVENT (#140)|
| agg_done (`t_aggregate_end`)     | compress → d2h         | aggregate slot    | EVENT       |
| slot_drained (`ready`/`prev_d2h_done`) | d2h → compress  | aggregate slot    | EVENT       |
| chunk_index_ready (`h_chunk_index_ready`) | d2h → HOST poll | drain copy source | EVENT+poll |
| staging_free (`t_h2d_end`)       | h2d → HOST poll        | staging slot      | EVENT+poll (unmetered busy-wait) |
| tail_published (`d_tail_seq`)    | HOST → compress        | tail state        | GEN_COUNTER (#142) |
| ingest cross-waits (stream.ingest.c ×4) | h2d ↔ compute | staging/pool      | EVENT       |
| drain-before-rekick; oldest-first delivery | HOST order  | slots, gate GEQ   | HOST_RULE (undeclared) |

~30 `cuEventRecord` sites, ~9 `cuStreamWaitEvent` sites, 2
`cuStreamWaitValue64`, 3 `cuEventQuery` poll loops, plus multiarray's own
record sites. Dead-edge candidates from the c3885f9 review (`pools.ready[2]`
et al.) must be re-verified against current main before deletion.

## Migration: five steps, shippable at every point

The end state is reached **in place** — each step replaces one subsystem
behind an existing seam. No flag day, no second engine.

1. **Ordering table.** `ordering.{h,c}`; migrate every record/wait/publish
   call site; re-derive the full edge table; tag timing events; add debug
   asserts + per-edge stall metrics. Separate commit deletes verified-dead
   edges. Behavior-neutral.
   *Exit: suite green; edge table in code matches a hand audit; dead edges
   gone; stall metrics visible in bench output.*
2. **Unify init/teardown and multiarray binding.** Single engine init/destroy
   shared by single-array and multiarray; kill the field-by-field copy
   checklist and the hand-mirrored memory estimate. Behavior-neutral.
   *Exit: suite green; multiarray bind is one struct handoff; memory estimate
   derived, not duplicated.*
3. **Resource pools with generations.** Introduce the pool API; move staging
   slots, chunk pool, aggregate slots, tail state into it; EVENT edges
   guarding those resources collapse into acquire/release; #142's gate
   becomes the pool's GEN_COUNTER release kind.
   *Exit: suite green; no raw cycled-buffer pointer crosses a stage boundary
   outside the pool API; determinism + tail-carry + round-trip still red on
   deliberately-broken builds (mutation check).*
4. **Extract the scheduler.** May be written greenfield and dropped in behind
   the now-clean stage interfaces in one PR — by then it moves wiring, not
   behavior. Dependency-shape variations (fallbacks, codec NONE, page-aligned)
   become schedule selections.
   *Exit: suite green; `stream.flush.c` orchestration state replaced; depth/
   fallback decisions in one module.*
5. **Workers.** Ingest staging copies and delivery move behind queues; the
   producer thread stops doing large memcpys.
   *Exit: suite green; L40 baseline rerun shows the ingest lever realized
   without zstd-path regressions; no new edges outside the table/pools.*

### Gates at every step

`ctest -E "(s3)"` full suite; `gpu_zstd_determinism`, `gpu_zstd_round_trip`,
`gpu_page_aligned_tail_carry` (≥8 reps); zero new build warnings; L40
baseline sweep rerun (`launch.sh` harness) at steps 3 and 5.

### Escape hatch

If during steps 2–3 adapter shims start outweighing the code they adapt, or
the seam inventory proves wrong, flip the remainder to a greenfield engine
behind the same stage interfaces — the landed steps still pay, having
sharpened the seams and the test spec a rewrite needs. The criterion is
mechanical: adapter LOC > adapted LOC in a step's diff ⇒ stop and reassess.

## Non-goals

- No change to layout math (`stream/config.c`), kernels (`aggregate.cu`,
  `lod.cu`), codec backends, or the zarr/sink layer — they keep their
  signatures throughout.
- No CUDA Graphs adoption: the pipeline is small enough that a declared
  table + pools gives the auditability without a graph executor (revisit
  only if the scheduler step shows dispatch overhead that matters).
- No attempt to preserve PR #139's cap-stacking; the baseline showed D2H
  batching is not a lever (≤×1.01).
