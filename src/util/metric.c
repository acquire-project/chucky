#include "util/metric.h"

void
reset_stream_metrics(struct stream_metrics* m)
{
  const struct stream_metrics old = *m;
  *m = (struct stream_metrics){ 0 };
  m->memcpy = mk_stream_metric(old.memcpy.name, old.memcpy.owner);
  m->h2d = mk_stream_metric(old.h2d.name, old.h2d.owner);
  m->lod_gather = mk_stream_metric(old.lod_gather.name, old.lod_gather.owner);
  m->lod_reduce = mk_stream_metric(old.lod_reduce.name, old.lod_reduce.owner);
  m->lod_append_fold =
    mk_stream_metric(old.lod_append_fold.name, old.lod_append_fold.owner);
  m->lod_morton_chunk =
    mk_stream_metric(old.lod_morton_chunk.name, old.lod_morton_chunk.owner);
  m->scatter = mk_stream_metric(old.scatter.name, old.scatter.owner);
  m->compress = mk_stream_metric(old.compress.name, old.compress.owner);
  m->aggregate = mk_stream_metric(old.aggregate.name, old.aggregate.owner);
  m->d2h = mk_stream_metric(old.d2h.name, old.d2h.owner);
  m->sink = mk_stream_metric(old.sink.name, old.sink.owner);
  m->flush_stall =
    mk_stream_metric(old.flush_stall.name, old.flush_stall.owner);
  m->delivery_dispatch =
    mk_stream_metric(old.delivery_dispatch.name, old.delivery_dispatch.owner);
  m->footer_buffer_stall = mk_stream_metric(old.footer_buffer_stall.name,
                                            old.footer_buffer_stall.owner);
  m->append_extent_stall = mk_stream_metric(old.append_extent_stall.name,
                                            old.append_extent_stall.owner);
  m->flush_writes_stall =
    mk_stream_metric(old.flush_writes_stall.name, old.flush_writes_stall.owner);
  m->backpressure =
    mk_stream_metric(old.backpressure.name, old.backpressure.owner);
  m->indexed_aggregate_wait = mk_stream_metric(
    old.indexed_aggregate_wait.name, old.indexed_aggregate_wait.owner);
  m->chunk_metadata_wait = mk_stream_metric(old.chunk_metadata_wait.name,
                                            old.chunk_metadata_wait.owner);
  m->chunk_metadata_copy = mk_stream_metric(old.chunk_metadata_copy.name,
                                            old.chunk_metadata_copy.owner);
  m->edge_stall[0] =
    mk_stream_metric(old.edge_stall[0].name, old.edge_stall[0].owner);
  m->edge_stall[1] =
    mk_stream_metric(old.edge_stall[1].name, old.edge_stall[1].owner);
  m->edge_stall[2] =
    mk_stream_metric(old.edge_stall[2].name, old.edge_stall[2].owner);
}
