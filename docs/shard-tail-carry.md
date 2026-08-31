# Hybrid GPU shard delivery

Chucky writes Zarr v3 sharded arrays through sinks that may require aligned
writes. Local unbuffered I/O typically requires the file offset, byte count,
and source address to be multiples of a page. Compressed chunk extents are
variable, so the GPU pipeline selects a delivery policy from two facts already
known at materialization time: the aggregate extent kind and the sink's
required shard alignment.

GPU aggregation itself is identical for every policy. It emits a compact,
tail-free device buffer containing only real chunk payload. Padding and fixed
tail placement are host delivery concerns and are never counted as D2H bytes.

## Policy

| aggregate extent | sink alignment | host and shard layout |
|---|---:|---|
| fixed | nonzero | prepend the committed host tail and page-floor writes, keeping the shard packed |
| fixed | zero | write the exact compact payload |
| indexed | nonzero | independently align each physical-shard update and zero-pad non-final updates |
| indexed | zero | write the exact compact payload |

An empty indexed update has no host reservation, padding, writer open, or data
write. It still advances the logical epoch cursor so a later nonempty update
lands in the correct index slot.

This makes indexed shard bytes sink-dependent. A buffered or object-store sink
gets compact payload. An aligned filesystem sink may retain internal zero gaps,
but every chunk index entry continues to name the exact compressed frame and
size. Decoded arrays are identical across those layouts; whole shard files need
not be byte-identical.

## Compact device aggregation

`src/gpu/aggregate.cu` performs one size permutation, one exclusive scan, and
one compact gather. Offsets are absolute in that compact device aggregate.
There are no per-shard capacity regions, device tail buffers, tail H2D uploads,
or page-padding bytes.

Fixed output constructs its offset/size shadow from the permutation geometry.
Indexed output first lands the device-generated offset/size metadata. The D2H
materializer then partitions either form into physical-shard generation runs.
Every nonempty run produces one exact payload span.

## Host materialization

The pinned host slot is larger than the compact device slot because physical
runs need independent source alignment.

Fixed aligned runs use:

```
aligned host base: [committed tail | fresh D2H payload]
```

Indexed aligned runs use:

```
aligned host base: [fresh D2H payload | zero trailing slack]
```

Indexed unaligned runs use exact extents with no host padding. Capacity checks
reserve real worst-case payload plus, for every possible physical run, the
policy-specific prefix or rounding slack and the slack needed to align the run
base. Every multiplication, addition, and round-up is checked before the D2H
copy is submitted.

The pinned slot remains double-buffered in this change. Increasing host slot
depth and accumulating several small compressed updates on the GPU are separate
optimizations.

## Fixed packed delivery

For a non-final fixed run:

1. `total = committed_tail + payload`.
2. Write `floor(total / page) * page` bytes.
3. Copy the sub-page remainder into persistent `active_shard.tail_buf`.
4. Advance the physical data cursor only by bytes actually written.

The next ordered materialization copies that committed remainder before its
fresh D2H payload. Chunk offsets therefore remain packed across batch
boundaries.

For a final fixed run, complete pages are written directly and the remainder
is placed immediately before the index and CRC in the existing footer buffer.
The aligned footer write is truncated to its logical size, leaving no gap.

## Indexed padded delivery

For a non-final aligned indexed run with nonzero payload:

1. Materialize payload at an aligned host base.
2. Zero `[payload_bytes, align_up(payload_bytes, page))`.
3. Write the rounded byte count.
4. Advance the physical cursor by that rounded count.

The next update's chunk offsets begin after the retained gap. Padding is always
zero, smaller than one alignment unit, and follows complete chunk frames; an
index entry never describes a split frame.

A final indexed run does not retain a new gap. Complete payload pages are
written directly. Its sub-page payload remainder is copied into the footer
buffer immediately before `[index | CRC]`, and the temporary footer alignment
slack is truncated. A final empty run can still emit the index/footer needed to
close a shard generation, but an empty non-final update emits no write.

Indexed unaligned delivery writes every nonempty run exactly and uses the same
compact final footer without internal padding.

## Pull-style shard drainer

`shard_drain_begin()` borrows one materialized `host_batch` and the currently
committed per-LOD shard states. `shard_drain_next()` then yields, in run order:

- data commands;
- footer commands;
- truncate commands;
- finalize commands.

Each command identifies its LOD and physical shard, file offset, source range,
physical byte count, direct-write eligibility, and buffer lease. Footer source
bytes are prepared only after the executor waits for that shard's footer-buffer
lease.

The sink executor is intentionally thin. It opens and optionally pre-sizes the
writer, executes the command, manages sink fences, and calls
`shard_drain_accept()` only after the sink accepts it. Accept commits index,
cursor, tail, and generation changes. `shard_drain_abort()` is sticky and does
not roll back commands already accepted by the sink.

This preserves oldest-first batch delivery. Multiarray streams retain their
immediate-drain rule, so swapping per-array state never leaves a command stream
borrowing another array's shard state.

## Buffer lifetimes and fences

`write_direct` borrows its source until the sink fence retires.

- Data commands borrow the aggregate host slot. The existing per-slot fence is
  recorded after delivery, including after a partial failure, and waited before
  the slot is materialized again.
- Footer commands borrow `active_shard.footer_buf`. The executor waits that
  shard's prior footer fence before preparing the bytes and records a new fence
  after a direct footer write.
- Unaligned footer commands use the copying writer path and a transient buffer.

Finalize-generation fences still guard publication of the readable append
extent.

## CPU pipeline

The CPU aggregator and `deliver_to_shards_batch()` keep their existing layout
and tail-carry behavior. The new host policy and pull drainer are GPU delivery
mechanisms; CPU shard bytes and public stream configuration are unchanged.

## Write-layout statistics

GPU delivery optionally reports:

- logical payload bytes;
- internal padding bytes retained in shard data regions;
- nonempty physical-shard update count;
- padded update count.

`physical_data_region_bytes` is derived as logical payload plus internal
padding. `padding_ratio` is internal padding divided by that physical data
region. Footer alignment slack is excluded because it is truncated rather than
retained. D2H payload bytes continue to equal the resolved payload spans and
exclude both fixed tails and indexed padding.

Small, highly compressible updates can consequently have a high filesystem
padding ratio. That measurement motivates a future GPU update-accumulation
policy; this implementation deliberately adds no size threshold.

## Code map

| concept | file |
|---|---|
| compact host-run materialization | `src/zarr/shard_delivery.c` |
| pull drainer and sink executor | `src/zarr/shard_drainer.{h,c}` |
| shard state | `src/zarr/shard_delivery.h` |
| D2H policy selection and copies | `src/gpu/d2h.materializer.{h,c}` |
| compact GPU aggregation | `src/gpu/aggregate.cu` |
| scheduling and oldest-first failure handling | `src/gpu/schedule.c` |
| CPU aggregation and delivery | `src/cpu/aggregate.c`, `src/zarr/shard_delivery.c` |
| writer and sink interfaces | `src/writer.h` |
