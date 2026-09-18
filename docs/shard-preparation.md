# Preparing shard files ahead of use

Set `config.prepare_shards = 1` before creating a CPU or GPU stream to
prepare shard files ahead of use. This is opt-in and currently supported by
the filesystem store, including Zarr arrays and NGFF multiscales.

Creation waits until the first shard generation is ready. When a stream
adopts one of those files, a separate worker creates and, where supported,
sets the capacity of the next file for that slot. There is at most one unused
prepared file per slot. A layout with 15 spatial shards therefore holds at
most 15 additional files, independently of the stream's length. A finite
stream does not prepare files beyond its configured capacity.

Preparation uses one worker separate from the normal write workers. Future
preparation is excluded from the fences used by writes, metadata publication,
and footer reuse. If preparation has not finished when a file is needed,
opening that shard waits for it.

`writer_flush` stops scheduling future preparation, drains accepted input,
and waits for preparation before removing unused files. `writer_close`
reports cleanup failures, and destruction also attempts cleanup when explicit
flush/close was omitted. Files already adopted follow the ordinary shard
lifecycle. Preparation creates files exclusively: it fails if a target
already exists, without truncating or deleting that file. Cleanup removes
only unused files and empty directories created by preparation; it preserves
pre-existing directories and directories containing used shards.

Store backends expose preparation through the shard pool. `prepare`,
`wait_prepared`, and `cancel_prepared` form one capability. Supporting
preparation requires supporting cancellation: wait for outstanding work,
reclaim unused owned resources, report failures, and allow cancellation to
be repeated. Opening a prepared shard transfers it to the ordinary writer.
Preparation and open calls are serialized by the stream; cancellation runs
after delivery stops using the prepared resources.

Custom sinks need the matching `prepare_shards`, `stop_preparing`, and
`cancel_prepared` hooks. Enabling the option on an unsupported sink, including
the current S3 backend, fails stream creation. With the option left at zero,
existing sinks retain their ordinary open-on-demand behavior.
