# GPU output slot ledger

## Problem

The GPU compressed-output path currently uses output slots as both aggregation
reservoirs and D2H ownership markers. The compressed macro-aggregation path
also has a fixed `batches_per_slot_cap` policy. That cap is a shortcut: it
limits how many batches may be stacked, but it is not the real resource limit.

The intended behavior is capacity-bound:

1. Compress a batch.
2. Measure the exact compressed byte count and descriptor count.
3. Append it to the current output slot if it fits.
4. Otherwise close the current slot, start D2H for that slot, and append into a
   reusable empty slot.

For compressed data, closing a slot should start payload D2H. The sink delivery
step may happen later.

The ledger needs exact compressed sizes, but the writer path should not block
on a foreground host synchronization. Measurement should be an asynchronous GPU
stage that copies a tiny result to pinned host memory. A CPU coordinator can
consume ready measurements, update the host ledger, and launch aggregation into
the reserved output slot while later input/compression work continues.

## Ownership Boundary

Slot selection should be host-owned. CUDA kernels may measure byte counts and
write into a reservation, but they should not decide whether another slot is
safe to reuse.

Introduce a small host-side output-slot ledger. It owns:

- output slot state
- payload byte capacity and cursor
- descriptor capacity and cursor
- batch-record capacity and count
- close sequence ordering
- close hazards such as shard tail rollforward boundaries

It does not own:

- compression
- aggregation kernels
- CUDA streams or events
- shard sink delivery
- codec internals

## Slot States

Use explicit slot lifetime states:

- `EMPTY`: slot may be opened for aggregation
- `OPEN`: slot is accumulating one or more batches
- `D2H_IN_FLIGHT`: slot is closed and D2H owns the device payload
- `HOST_READY`: D2H is complete; host buffers may be delivered to the sink
- `DELIVERING`: sink delivery is reading host buffers

The current `CLOSED` state can initially stand in for `D2H_IN_FLIGHT`, but its
meaning should become "payload D2H has been queued", not only "metadata D2H has
been queued".

## Reservation Model

The ledger should answer whether a measured batch can append to the current
open slot. If not, it returns the slot to close and a reservation in an empty
slot. The first version can be pure C and unit-tested without CUDA.

Append is allowed only when:

```text
state == OPEN
data_cursor + request.data_bytes <= data_capacity
desc_cursor + request.desc_entries <= desc_capacity
batch_count < batch_record_capacity
request does not violate tail-rollforward constraints
```

The batch cap becomes a metadata allocation bound, not the performance policy.

## Concrete Ledger Design

### Data Model

The ledger tracks two output slots. It does not store pointers to aggregate
buffers; the caller maps the returned slot index to `aggregate_slot`.

```c
enum output_ledger_state {
  OUTPUT_LEDGER_EMPTY = 0,
  OUTPUT_LEDGER_OPEN = 1,
  OUTPUT_LEDGER_D2H_IN_FLIGHT = 2,
  OUTPUT_LEDGER_HOST_READY = 3,
  OUTPUT_LEDGER_DELIVERING = 4,
};

struct output_slot_capacity {
  size_t data_bytes;
  uint64_t desc_entries;
  uint32_t batch_records;
};

struct output_slot_entry {
  enum output_ledger_state state;
  size_t data_cursor;
  uint64_t desc_cursor;
  uint32_t batch_count;
  uint64_t close_seq;
};

struct output_slot_ledger {
  struct output_slot_capacity capacity;
  struct output_slot_entry slot[2];
  int current;
  uint64_t next_close_seq;
};
```

Capacity is uniform across the two GPU output slots. If multiarray later needs
per-array capacities, bind/unbind should save and restore the whole ledger
state rather than rebuilding policy from scattered fields.

### Requests and Reservations

A request describes a measured batch before aggregation writes into an output
slot. The measurement is produced by the GPU after compression and copied to
pinned host memory asynchronously.

```c
struct output_slot_request {
  size_t data_bytes;
  uint64_t desc_entries;
  int closes_after_append;
  int tail_rollforward_blocked;
};

struct output_slot_reservation {
  int slot;
  size_t data_base;
  uint64_t desc_base;
  uint32_t batch_index;

  int close_before_append;
  int close_slot;
  int close_after_append;
};
```

`closes_after_append` means the batch may be written, but the resulting slot
must close immediately. This covers explicit flush/EOS and shard finalization
cases that cannot safely share a slot with later batches.

`tail_rollforward_blocked` means the request cannot append to a non-empty open
slot. If the current slot is empty, the request may use it; otherwise the
ledger must close the current slot before reserving space elsewhere.

`data_bytes` must be bytes consumed in the output slot, not merely the sum of
compressed chunk sizes. In contiguous mode this is the compressed scan total.
In page/tail-carry mode it must include leading tail bytes for the shards that
will be copied into the slot.

### Measurement Handoff

The pipeline boundary should become:

```text
compress batch
  -> measure exact output-slot requirements on GPU
  -> async copy tiny measurement record to pinned host memory
  -> coordinator plans and commits ledger reservation
  -> aggregate writes into the reserved slot range
```

The coordinator should wait or poll on measurement-ready events outside the hot
writer path. Avoid launching CUDA work from CUDA host callbacks; use a normal
CPU coordinator path that owns ledger mutation and CUDA submission.

The initial implementation should keep the current double-buffered compressed
staging depth. Measurement and ledger state should add no new full compressed
batch buffers:

```text
compressed slot 0: measured / waiting for reservation and aggregation
compressed slot 1: next batch compressing
```

If both compressed slots are occupied, apply backpressure before reusing one.
Increasing compressed staging depth can be considered later as a performance
tuning option, but it is not part of the initial design.

### Operations

The minimal API should be:

```c
enum output_ledger_error output_slot_ledger_init(
  struct output_slot_ledger* ledger,
  struct output_slot_capacity capacity);

enum output_ledger_error output_slot_ledger_plan_append(
  const struct output_slot_ledger* ledger,
  const struct output_slot_request* request,
  struct output_slot_reservation* out);

enum output_ledger_error output_slot_ledger_commit_append(
  struct output_slot_ledger* ledger,
  const struct output_slot_request* request,
  const struct output_slot_reservation* plan);

enum output_ledger_error output_slot_ledger_close(
  struct output_slot_ledger* ledger,
  int slot,
  uint64_t* out_seq);

enum output_ledger_error output_slot_ledger_mark_host_ready(
  struct output_slot_ledger* ledger,
  int slot);

enum output_ledger_error output_slot_ledger_begin_delivery(
  struct output_slot_ledger* ledger,
  int slot);

enum output_ledger_error output_slot_ledger_finish_delivery(
  struct output_slot_ledger* ledger,
  int slot);

enum output_ledger_error output_slot_ledger_reset_empty(
  struct output_slot_ledger* ledger,
  int slot);

enum output_ledger_error output_slot_ledger_oldest_closed(
  const struct output_slot_ledger* ledger,
  int* out_slot);

int output_slot_ledger_has_work(const struct output_slot_ledger* ledger);
```

`plan_append` does not mutate the ledger. It returns the exact base offsets
that the aggregate path must use, plus any close required before that append.
If `close_before_append` is set, the caller must queue D2H for `close_slot`
and call `output_slot_ledger_close()` successfully before committing the append.

`commit_append` mutates the target slot cursors and batch count. It should be
called after any required pre-append close has succeeded and before launching
aggregation into the returned reservation. If the following CUDA launch fails,
the stream is already in an error path, so keeping the reservation committed is
acceptable.

`close_after_append` is not committed by `commit_append`. It tells the caller
to close the target slot after aggregation has populated the slot metadata and
the D2H stage can wait on the aggregate completion event.

### Append Planning Rules

`plan_append` should enforce these rules in order:

1. Reject a request that can never fit an empty slot.
2. Treat an `EMPTY` current slot as an open empty candidate.
3. If the request fits the current candidate and tail rollforward allows it,
   plan an append there.
4. Otherwise require the current slot to be `OPEN`, plan it as
   `close_before_append`, select the alternate slot, and require the alternate
   to be `EMPTY`.
5. Plan an append into the alternate at base `(0, 0, 0)`.
6. If `closes_after_append` is set, mark `close_after_append` in the returned
   reservation so the caller closes the target after aggregation completes.

The ledger should not silently deliver or reset a non-empty alternate slot. If
the alternate is not `EMPTY`, `plan_append` returns a backpressure error. The
caller must drain host-ready/closed slots and retry.

`commit_append` should:

1. Open an `EMPTY` target slot if needed.
2. Verify the plan still matches the current ledger state.
3. Advance `data_cursor`, `desc_cursor`, and `batch_count`.
4. Set `current` to the target slot.

### Close and Delivery Rules

Closing an `OPEN` slot assigns `close_seq = next_close_seq++` and transitions
to `D2H_IN_FLIGHT`.

Host-ready and delivery transitions are explicit:

```text
OPEN -> D2H_IN_FLIGHT -> HOST_READY -> DELIVERING -> EMPTY
```

During the transitional integration before close-time compressed D2H exists,
delivery may begin from `D2H_IN_FLIGHT`; the drain step performs the sync and
payload copy. Once close-time D2H is implemented, delivery should begin from
`HOST_READY`.

`oldest_closed` returns the lowest `close_seq` among slots in
`D2H_IN_FLIGHT`, `HOST_READY`, or `DELIVERING`, depending on the integration
stage:

- before close-time D2H exists, it can return `D2H_IN_FLIGHT` slots for drain
- after close-time D2H exists, sink delivery should wait for `HOST_READY`

The exact predicate can change at Stage 4, but the sequence number should stay
the ordering mechanism.

### Error Model

Use small integer status codes rather than `writer_result` in the pure module:

```c
enum output_slot_error {
  OUTPUT_LEDGER_OK = 0,
  OUTPUT_LEDGER_INVALID = 1,
  OUTPUT_LEDGER_TOO_LARGE = 2,
  OUTPUT_LEDGER_BACKPRESSURE = 3,
};
```

`BACKPRESSURE` means the request could fit in principle, but both slots are
currently occupied. The caller should drain and retry.

### Invariants

The tests should assert these invariants after every operation:

- `data_cursor <= capacity.data_bytes`
- `desc_cursor <= capacity.desc_entries`
- `batch_count <= capacity.batch_records`
- only `EMPTY` slots have zero cursors
- `slot[current]` is either `EMPTY` or `OPEN` unless both slots are closed or
  pending
- close sequence numbers are strictly increasing
- a slot cannot transition directly from `D2H_IN_FLIGHT` to `EMPTY`

## First Unit Tests

The initial ledger test should be CPU-only and cover:

1. First reservation opens slot 0 at base `(0, 0, 0)`.
2. Multiple reservations append while bytes, descriptors, and records fit.
3. Byte overflow closes current and reserves alternate.
4. Descriptor overflow closes current and reserves alternate.
5. Batch-record overflow closes current and reserves alternate.
6. Tail-rollforward block closes a non-empty current slot.
7. A too-large request fails without mutating state.
8. Both slots occupied returns `BACKPRESSURE`.
9. Close sequence ordering drains oldest first.
10. Delivery completion resets cursors and makes the slot reusable.

## Implementation Stages

### Stage 1: Extract the Ledger

Add a small module, likely `src/gpu/output_slot.{h,c}`, with focused unit tests.
Wire it into the existing path without changing behavior yet. This stage should
replace ad hoc host-side state transitions in `stream.flush.c`, but it should
not change aggregate kernels or D2H semantics.

Success criteria:

- output slot state transitions are tested without CUDA
- current GPU orchestration tests still pass
- no broad rewrite of compression, aggregation, or delivery code

### Stage 2: Make Capacity Dynamic

Replace the fixed compressed `batches_per_slot_cap` policy with capacities:

- payload bytes available in `d_aggregated` and `h_aggregated`
- descriptor entries available in `d_offsets` and `d_permuted_sizes`
- batch records available in `slot_batches`

Keep a conservative batch-record capacity if needed for allocation, but close
because a real resource is exhausted, not because a fixed batch count was hit.

Success criteria:

- no magic compressed batch cap
- tests cover byte-full, descriptor-full, batch-record-full, and explicit flush
- passthrough and compressed paths keep their existing correctness behavior

### Stage 3: Reserve Before Aggregate Write

Split measurement from writing without adding a foreground host sync:

1. Compression and sizing produce exact byte and descriptor requirements.
2. A tiny measurement record is copied to pinned host memory asynchronously.
3. The CPU coordinator reserves `(slot, data_base, desc_base, batch_index)`.
4. Aggregation kernels write only into the reserved range.

At this point the CUDA routing kernel should no longer select target slots.

Success criteria:

- aggregate target selection is host-owned
- kernels receive an already-reserved destination
- alternate slot reuse is controlled only by ledger state
- measurement/ledger handoff does not require a third compressed batch buffer

### Stage 4: Close-Time Compressed D2H

Move compressed payload D2H to the slot close transition. Closing a compressed
slot should queue metadata D2H, determine exact payload ranges, queue payload
D2H, and record a host-ready event. Draining should wait for host readiness and
perform sink delivery in close sequence order.

Success criteria:

- closed compressed slots have payload D2H in flight
- sink delivery is separated from device-slot reuse
- the next aggregation kick does not synchronously perform the prior slot's
  compressed payload copy

## Non-Goals

- Do not rewrite codec compression.
- Do not rewrite shard delivery.
- Do not redesign sink IO.
- Do not fold this into multiarray behavior until the single-array ownership
  boundary is stable.
- Do not perform broad cleanup before the ledger is in place.
