# dev log

## TODO

- [ ] multiple-array streaming (HCS use case)
- [x] s3 writer
- [x] make a report to characterize performance/memory by chunk size
- [ ] interface for streaming from device, integrating with a cuda stream
- [x] u8, u32, i8, i16, i32
- [x] u64, i64, double?
- [x] fp16
- [x] cpu impl
- [ ] whitepaper
- [ ] coverage
- [ ] ci/cd
- [ ] metadata
- [x] fix "temporal" vs "spatial" naming.
- [x] fix uses of the work chunk, then fix uses of the word tile.
- [x] boundary conditions for dim0 downsampling - it's a ceil like it should be
- [x] lod for dim0
- [x] move benchmark suite out of tests
- [x] optimize buffering for compression stage - may need more than one epoch
- [x] "append dimension" semantics for dim0 - can be infinitely sized, updates over time 
- [x] uniform handling of tiles across lods for compress and aggregate
- [x] uniform handling of tiles for shard writer
- [x] shard writer handles lods
- [x] replicate boundary condition
- [x] min, max, median, suppressed max/min
- [x] verify the multiscale zarr is visualizable
- [x] make sure we're using the right condition to stop downsampling. all
      dims need to be bigger than tile size
- [x] add metrics, bench
- [x] optimize lod scattner, lod gather kernels.
- [x] within epoch transpose
- [x] unbuffered io
- [x] cleanup tests vs experiments
- [x] evaluate gather vs scatter for non-lod stream
- [ ] bench 2 streams, 1 gpu
- [ ] api: multiple append dimensions. how does that work for lod? These
      would be all left dims where chunk shape is 1.
Cleanup
- [ ] make sure everything has extern c guards
- [ ] comments at the top of each test
- [ ] look into j8 failures

## 2026-03-24

Need to move to soa internal representation for dims. Doing some refactoring
of the public/private interfaces first, so that I can clearly distinguish
the soa representation as private.

## 2026-03-23

Looking at aqz integrtion more. Need to think about what it takes to
redistribute this thing.

Will need to wheel variants - one with the cuda driver dependency and one
without.

Can use a `docker-compose.yml` to spin up the `minio` service that my
`Dockerfile` can use. Made some fixes to the `Dockerfile` as well.

I can think more about creating `install` targets; one with and with out cuda.
But first I need to figure out the approach around HCS support.

I've described the problem before (below).

Might be able to constrain: use cases tend to be single snapshot at each fov
and folks just want to have a `tiff` replacement. Could force thin epochs/an
epoch fill before switching streams.

- [ ] api: multiple append dimensions. how does that work for lod? These
      would be all left dims where chunk shape is 1.

Could tier the epoch pool.

So the ideas:

1. everything compress and downstream is stateless. would need to track the
   target array.
2. if we have "thin" epochs, we can just serialize. Should think more about what
   to do to enable this. In particular, if we have several left-dimensions where
   the chunk shape is 1, we can generalize the concept of append dimension to apply
   across them. Has some lod implications but otherwise could be straightforward.
   Constraining streams to complete epochs before switching is probably workable
   for these users.
3. if we have to support "thick" epochs, we can think about tiered storage
   across gpu, host ram, local persist.

Best to force a single chunk shape across arrays. Keeps everything simple.
Probably also need to force only the right-most append dimension to be
downsamplable.

For generalizing to multiple append dimensions, I think I just enforce a
constraint that only the right-most append dimension is downsamplable.

## 2026-03-20

I've been trying to use git worktrees and they are a nightmare. I keep loosing
code - and devlog entries! I've been working on getting benchmark sweeks up.
Results get recorded in jsons in `bench/results/` and I can aggregte across
commits and hosts. There's a `report.py` that generates some visualizations
to browse things and it's all very very useful.

- [x] sweep doesn't include io at the moment
- [ ] medfmt and smallepoch scenarios need some analysis/optimization

The other thing I've been working on is s3 integration. It's pretty easy to
test with a `minio` server launched via `docker`. Need to think about
multipart upload failures; appending to shards is handled a one big multipart.
Also need to think about the part budget. Wrote an s3 guide.

Adapting the benchmark sweep to do s3. Testing it on minio on auk. May need to
do 10 Gbps and 100 Gpbs runs - I don't have a variable for that yet. Also
need to make sure the right metadata is captured in the results.

I'm still thinking about the problem of writing to multiple arrays at once...
The use case is how to have 100's of non-performant streams at once. Could
think about a suspend state.

- maybe separate zarr v ngff metadata
- distinguish public vs private api more
- should probably get struct slice out of the public api, it's not necessary
- document the benchmark sweeps and how they work. reporting, schema, aggregation


## 2026-03-18

- [x] Should make dims_advise part of the tile_stream api since it depends on the
      config.

Alan's been working on this an interesting use case. People will have several
arrays they are streaming to at the same time, possibly with different sizes? If
you just open many independent streams you'll run out of resources - memory is
the most precious one. It's usually for regimes that don't need performance and
are more resource constrained.

If you stream in an epoch before switching streams than theoretically you could
reuse a bunch of the same resources. After you have a tiled epoch (with lods
etc) everything downstream is just how to address the chunks to the correct
shards. There could be a demuxing before the compress/aggregate followed
by the usual muxing to shards. It's N-streams in to 1 compressor to M-shards
out.

I don't particularly like it. What does it save? It saves the overhead of the
compress/aggregate step per stream. If you constrain callers to finish an
epoch before switching, that can save a lot, but it's so constraining.

## 2026-03-17

Still need to do the dim0 fold for cpu. Need to do some review passes. Then I
want to see how well it works on betchmarks.

Initial benchmarks show 0.37 GB/s on cpu for 256cube w no compression. So
pretty slow. Cpu activity is pretty bursty. Initial implementation was single
stream, so that probably needs fixing.

Also, I turned off batching for the cpu compression, but that might be necessary
when epochs are roughly a single tile in size. This would be important for
when people are doing particularly high frame rate imaging with small fields.

Ok, the scatter was slowing it down. Now that I fixed that getting pretty
decent speeds on auk (2-3 GB/s for non-lod).

Made sizing the chunk a little easier for the benchmarks. Looks at available
memory and shrinks the requested chunk size as necessary.

The cpu implementation is quite fast, especially for non-lod.

Next step is s3 streaming and  to figure out how to do a panel of benchmarks.
Now that I think the implementations might not change much, it'd be good to
characterize performance by chunk size, compression, cpu/gpu, etc.

## 2026-03-16

Looking at api examples and cleaning up a little bit.

Resolving spatial/temporal naming. Now "outer"/"dim0"/"append" and "inner."

Fixing uses of the word chunk that are a bit off target.

Analyzing acquire-zarr integration.

When running tests with `-j8` noticed a bunch of weird bugs - even managed to
crash the driver at one point.

Started doing the cpu implementation. I moved a bunch of the source files around
...I should have done the same for tests - will have to clean that up.

Got tests to pass. Adding a test to validate the results are the same between
the cpu and gpu pipeline. Adding the cpu backend to the benchmarks.


## 2026-03-15

Refactoring tests and isolating pipeline stages more.

Spent a good amount of time rewriting the design doc which I think will
basically turn into a whitepaper. Or at least I can use the design doc to get
some useful text down.

Added support for many different scalar types. Apparently zarr supports fp16.

## 2026-03-14

Trying to refactor to isolate pipeline stages.

## 2026-03-13

Confirmed using a non-trivial storage order doesn't effect performance.

Realized I was hardcoding zstd in the zarr metadata. Fixing. Probably need to
double check that part of the code again.

I've been just adding stuff to the code as I go. I need to break up a bunch of
files/functions. So doing some cleanup...

## 2026-03-12

Finishing unbuffered io on posix.

Benching on livescreen again (fixed Sink timing):
```
=== multiscale_dim0 ===
  GPU memory:  16.59 GiB device, 3.26 GiB pinned
    staging:   256.00 MiB   tile_pool: 3.01 GiB
    comp_pool: 3.01 GiB   aggregate: 3.01 GiB
    lod:       3676.68 MiB   codec:     3813.82 MiB
    tiles:     2304/epoch, 3080 total (6 LOD levels, batch=1)
  total:       1757.81 GiB (943718400000 elements, 1563 epochs)
  tile:        262144 elements = 512 KiB  (stride=262144)
  epoch:       2304 slots, 1152 MiB pool
  compress:    max_output=524322 comp_pool=1540 MiB
  LOD levels:  6

  --- Benchmark Results ---
  Input:        1757.81 GiB (943718400000 elements)
  Compressed:   232.89 GiB (ratio: 0.132)
  Tiles:        3601152 (2304/epoch x 1563 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          25.76    26.98       2.43       2.32
  H2D             53.74    53.81       1.16       1.16
  Copy           930.24  1417.36       0.07       0.04
  LOD Gather     662.46   671.97       1.70       1.67
  LOD Reduce     999.41   999.76       1.50       1.50
  Dim0 Fold      539.54   702.38       1.39       1.07
  LOD to tiles    37.92   853.87      39.66       1.76
  Compress        14.10    19.87     106.65      75.70
  Aggregate     3682.52  4815.59       0.41       0.31
  D2H             62.41    71.38      24.10      21.07
  Sink             5.50 744264.83      27.07       0.00

  Init time:     1.273 s
  Wall time:     370.876 s
  Throughput:    4.74 GiB/s
  PASS  
```

I think I'm about done with all the major features besides maybe the in-epoch
transpose. I need to look at that pr in acquire-zarr some more ... ok, it's
compatible. It restricts dim0 and adds some other restrictions that are
artifacts of how aqz's frame-based streaming. API is just specifying the
permutation as a vector of integers.

I'm at about 20ksloc in the code right now. ~2ksloc cuda and ~15ksloc in c.
A lot of that is in `tests`. In `src`, it's a total of 8ksloc (2k cuda, 5k c).
Worst offenders are `stream.c` and `lod.cu`. `aggregate.cu` has gotten a bit
beefy too.

 - Review - separate static (config) state from mutable state

Hmm, I think I might not have been timing things right... had a bug with how
I was setting up the unbuffered io on windows. Fixed that added some timing
around the flush step. Getting 5.7 GB/s on livescreen-1 for no compression
no lod. Compression actually speeds up throughput: 7.6 GB/s lz4, 6.4 GB/s zstd.

Spent a fair amount of the day debugging stray alignment issues and fixing
up transpose (storage order) support. I decided to make it part of the
dimension description which significantly simplifies reasoning about whether
the `dims` array has been permuted to storage order or not.

Noticed I was transferring the uncompressed data size for the D2H. Using
the compressed size means transmitting the size back to the host and a sync
but it's on the `d2h` stream and should be quick.

Added an api for setting up the dims array.

## 2026-03-11

Benchmarking on oreb (5090)

```
GB/s
6.10  bench_stream_256cube_single
5.50  bench_stream_256cube_multiscale 
5.43  bench_stream_256cube_multiscale_dim0
2.90  bench_stream_orca2_single
2.81  bench_stream_orca2_multiscale
Fail  bench_stream_orca2_multiscale_dim0
```

(I fixed the failure but didn't get the timing)

Exploring tile size vs performance with `bench_stream_orca2_multiscale`.
`auk` doesn't have much memory and I keep running into memory blowing up,
especially for smaller tile sizes. LOD scales particularly badly - the number
of lod's goes up.

Benchmarking on livescreen-1 (A6000)

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71                 Driver Version: 595.71         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...  WDDM  |   00000000:E1:00.0  On |                  Off |
| 30%   32C    P8             16W /  600W |    1676MiB /  97887MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+  
```

Using 64x1x64x64 zcyx tiles on bench_stream_orca2_multiscale_dim0. There's
300+ GB of RAM on that machine, so IO times are really just RAM bandwidth
here.

```
=== multiscale_dim0 ===
  GPU memory:  16.59 GiB device, 3.26 GiB pinned
    staging:   256.00 MiB   tile_pool: 3.01 GiB
    comp_pool: 3.01 GiB   aggregate: 3.01 GiB
    lod:       3676.68 MiB   codec:     3813.82 MiB
    tiles:     2304/epoch, 3080 total (6 LOD levels, batch=1)
  total:       17.58 GiB (9437184000 elements, 16 epochs)
  tile:        262144 elements = 512 KiB  (stride=262144)
  epoch:       2304 slots, 1152 MiB pool
  compress:    max_output=524322 comp_pool=1540 MiB
  LOD levels:  6

  --- Benchmark Results ---
  Input:        17.58 GiB (9437184000 elements)
  Compressed:   2.38 GiB (ratio: 0.132)
  Tiles:        36864 (2304/epoch x 16 epochs)
  
  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          25.49   106.32       2.45       0.59
  H2D             53.71    53.81       1.16       1.16
  Copy           928.99  1387.16       0.07       0.05
  LOD Gather     657.86   670.32       1.71       1.68
  LOD Reduce    1000.00  1001.98       1.50       1.50
  Dim0 Fold      406.52   699.80       1.84       1.07
  LOD to tiles    23.41   855.75      64.25       1.76
  Compress        14.01    19.79     107.37      76.00
  Aggregate     3276.13  4799.85       0.46       0.31
  D2H             62.40    71.37      24.10      21.07
  Sink            41.98 2420882.97      28.83       0.00

  Init time:     1.314 s
  Wall time:     3.997 s
  Throughput:    4.40 GiB/s
  PASS  
```

Looking into unbuffered io - at the very least to get more accurate io numbers.
There's some performance to eke out there too. That comes with some alignment
constraints. I probably need to aggregate shards to page-aligned boundaries.

Also working on making sure I'm treating dim0 right. `dim[0].size=0` should
enable infinite streaming. For a finite size, the stream should terminate. And
the zarr shape should update in an eventually consistent way.

## 2026-03-10

The streaming pipeline prepares an epoch worth of tiles for compression. The
compression works best when there are many tiles to compress. However, depending
on the stream config, sometimes there may not be many tiles. I could buffer
multiple epochs worth of tiles if necessary so I could guarantee >1000 tiles
are ready for compression. Typically I'm designing for 1MB/tile so this would
represent ~1GB of buffer, which seems reasonable.

I do some double-buffering before the compress stage - this might need to go
to a ring buffer. Need to adjust the aggregate step too so it knows which
epoch tiles came from. It'll use that info to map to shard and lod id.

On auk, `bench_stream_256cube_multiscale_dim0` looks like it improved from
(yesterdays) 1.88 GB/s to 1.99 GB/s. Need to play around with the chunking
and see if robustness improved there.

Reorganized the tests.

## 2026-03-09

Could think about a lut for the scatter step on the non-lod path.
Also on auk, the avg/best is pretty bad for th"Scatter" and "Copy" steps.

Overall getting 2.19 GB/s and 1.83 GB/s for without and with LOD, respectively,
on auk. After the compression, the next most significant times are  D2H and
what looks like a delayed H2D. The "delayed H2D" looks like the H2D is actually
showing up in the scatter/copy times.

Time to look into LOD along the append dimension. This is tricky. First, I
can't accumulate in memory all the data needed for all the lod's. For 5 levels,
this would mean 2^5=32 epochs and I'm designing for ~1e9 element epochs.

This means I'll need to use an accumulator and a different reduce method for
dim0 than what we specify for the dims inside an epoch.

It also means that different lods will emit data at different times. lod0 will
emit every epoch, lod1 every other epoch, lod2 every 4th, and so on. I suspect
this can be handled by adjusting how much data needs to be moved during the
"LOD to tiles" step.

Reasonable throughput on auk (laptop, 5070):

```
=== bench_multiscale ===
  GPU memory:  2.51 GiB device, 0.68 GiB pinned
    staging:   256.00 MiB   tile_pool: 0.43 GiB
    comp_pool: 0.43 GiB   aggregate: 0.43 GiB
    lod:       493.70 MiB   codec:     502.54 MiB
    tiles:     12288/epoch, 14043 total (5 LOD levels)
  total:       93.75 GiB (50331648000 elements, 500 epochs)
  tile:        8192 elements = 16 KiB  (stride=8192)
  epoch:       12288 slots, 192 MiB pool
  compress:    max_output=16395 comp_pool=219 MiB
  LOD levels:  5

  --- Benchmark Results ---
  Input:        93.75 GiB (50331648000 elements)
  Compressed:   11.60 GiB (ratio: 0.124)
  Tiles:        6144000 (12288/epoch x 500 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          19.37    21.27       3.23       2.94
  H2D             13.30    13.46       4.70       4.64
  Copy           140.12   165.42       0.45       0.38
  LOD Gather      47.66    48.21       3.93       3.89
  LOD Reduce      81.83    96.21       2.62       2.23
  LOD to tiles    90.07    94.35       2.38       2.27
  Compress         3.02     3.81      71.07      56.22
  Aggregate      675.20   745.93       0.32       0.29
  D2H             15.08    15.42      14.22      13.90
  Sink         58510.70 70249.84       0.00       0.00

  Wall time:     47.651 s
  Throughput:    1.97 GiB/s
  PASS
=== bench_multiscale_dim0 ===
  GPU memory:  2.58 GiB device, 0.68 GiB pinned
    staging:   256.00 MiB   tile_pool: 0.43 GiB
    comp_pool: 0.43 GiB   aggregate: 0.43 GiB
    lod:       562.26 MiB   codec:     502.54 MiB
    tiles:     12288/epoch, 14043 total (5 LOD levels)
  total:       93.75 GiB (50331648000 elements, 500 epochs)
  tile:        8192 elements = 16 KiB  (stride=8192)
  epoch:       12288 slots, 192 MiB pool
  compress:    max_output=16395 comp_pool=219 MiB
  LOD levels:  5

  --- Benchmark Results ---
  Input:        93.75 GiB (50331648000 elements)
  Compressed:   11.92 GiB (ratio: 0.127)
  Tiles:        6144000 (12288/epoch x 500 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          19.43    21.25       3.22       2.94
  H2D             13.26    13.46       4.71       4.64
  Copy           123.04   165.00       0.51       0.38
  LOD Gather      47.71    48.30       3.93       3.88
  LOD Reduce      81.74    96.35       2.62       2.22
  Dim0 Fold      102.94   181.21       0.52       0.30
  LOD to tiles    79.06    93.38       2.71       2.29
  Compress         2.90    15.34      73.82      13.97
  Aggregate      601.49  4583.26       0.36       0.05
  D2H             14.21  5080.15      15.09       0.04
  Sink         55068.34 63923.94       0.00       0.00

  Wall time:     49.942 s
  Throughput:    1.88 GiB/s  
```

Factoring benchmarks out of the tests - it takes too long to run the tests
and I don't always want to see all three benchmarks, manage scenarios etc.



## 2026-03-08

Using look up tables. xor pattern. 1000x256^3 2channel data. zstd w mean for
lod.

Current benchmark on oreb (5090)

```
=== bench ===
  GPU memory:  2.01 GiB device, 0.63 GiB pinned
    staging:   256.00 MiB   tile_pool: 0.38 GiB
    comp_pool: 0.38 GiB   aggregate: 0.38 GiB
    lod:       0.00 MiB   codec:     650.29 MiB
    tiles:     12288/epoch, 12288 total (1 LOD levels)
  total:       93.75 GiB (50331648000 elements, 500 epochs)
  tile:        8192 elements = 16 KiB  (stride=8192)
  epoch:       12288 slots, 192 MiB pool
  compress:    max_output=16395 comp_pool=192 MiB

  --- Benchmark Results ---
  Input:        93.75 GiB (50331648000 elements)
  Compressed:   11.60 GiB (ratio: 0.124)
  Tiles:        6144000 (12288/epoch x 500 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          22.53    25.39       2.77       2.46
  H2D             36.55    53.83       1.71       1.16
  Scatter         64.95    95.68       0.96       0.65
  Compress         9.71     9.82      19.31      19.10
  Aggregate     1822.56  1880.47       0.10       0.10
  D2H             53.37    53.41       3.52       3.51
  Sink         98384.48     0.00       0.00       0.00

  Wall time:     15.382 s
  Throughput:    6.09 GiB/s
  PASS
=== bench_multiscale ===
  GPU memory:  2.70 GiB device, 0.68 GiB pinned
    staging:   256.00 MiB   tile_pool: 0.43 GiB
    comp_pool: 0.43 GiB   aggregate: 0.43 GiB
    lod:       493.70 MiB   codec:     695.05 MiB
    tiles:     12288/epoch, 14043 total (5 LOD levels)
  total:       93.75 GiB (50331648000 elements, 500 epochs)
  tile:        8192 elements = 16 KiB  (stride=8192)
  epoch:       12288 slots, 192 MiB pool
  compress:    max_output=16395 comp_pool=219 MiB
  LOD levels:  5

  --- Benchmark Results ---
  Input:        93.75 GiB (50331648000 elements)
  Compressed:   13.29 GiB (ratio: 0.142)
  Tiles:        6144000 (12288/epoch x 500 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          23.14    26.60       2.70       2.35
  H2D             37.58    53.83       1.66       1.16
  Copy           428.14   535.98       0.15       0.12
  LOD Gather     104.97   107.45       1.79       1.75
  LOD Reduce     212.96   216.13       1.01       0.99
  LOD to tiles   204.81   207.49       1.05       1.03
  Compress         9.88     9.98      21.70      21.47
  Aggregate      762.34   969.01       0.28       0.22
  D2H             53.26    53.30       4.03       4.02
  Sink         69235.75     0.00       0.00       0.00

  Wall time:     18.376 s
  Throughput:    5.10 GiB/s
  PASS
```

With writing to disk enabled

```
Without LOD  5.84 GB/s  
With LOD     4.76 GB/s
```

For the Orca Quest 2 scenario:

```
Without LOD  3.31 GB/s  
With LOD     2.82 GB/s
```

I think that's because of the fewer tiles/epoch: 576/epoch for this scenario
compared to 12288/epoch for the previous one. Should try buffering tiles before
compression.

## 2026-03-02

Adding more reduce methods - min, max, median, min suppressed, max suppressed.

Re-valuating some of the past todo's

Something very wrong with the lod scatter kernel. Was thinking there's probably
a way for threads to share work when computing the compacted morton code, and
started looking at the scatter kernel. I started fixing it but it needs more
work...

## 2026-03-01

Continuing to refactor. Got the end-to-end working. Was seeing very
slow compression times, but that was because the benchmark had too few
chunks per epoch. Assembling the morton order and then scattering
back out to tiles is still slow. My guess is the compute is too high,
but I'm also not convinced the loads are right.

Current benchmark on oreb (5090)

```
=== test_bench_multiscale ===
  total:       9.38 GiB (5033164800 elements, 50 epochs)
  tile:        8192 elements = 16 KiB  (stride=8192)
  epoch:       12288 slots, 192 MiB pool
  compress:    max_output=16395 comp_pool=219 MiB
  LOD levels:  9

  --- Benchmark Results ---
  Input:        9.38 GiB (5033164800 elements)
  Compressed:   1.33 GiB (ratio: 0.142)
  Tiles:        614400 (12288/epoch x 50 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          23.10    26.45       2.71       2.36
  H2D             39.52    53.82       1.58       1.16
  Copy           381.37  2929.69       0.16       0.02
  LOD Scatter      9.30    17.74      20.15      10.57
  LOD Reduce     234.29   260.46       0.91       0.82
  LOD Gather      15.68    16.00      13.68      13.40
  Compress         9.71     9.95      22.08      21.55
  Aggregate      389.50   506.64       0.55       0.42
  D2H             53.15    53.23       4.04       4.03
  Sink         10447.33     0.00       0.00       0.00

  Wall time:     3.427 s
  Throughput:    2.74 GiB/s
```

I confirmed I could write ngff zarr's compatible with neuroglancer. Needed
to fix transforms

Added Lz4 compression.

I'm definitely not having trouble with IO.

Playing with the dimension config - it really likes small tiles. It's
hard to balance that with memory usage for larger fields. Trying to
test conditions similar to typical microscope camera streams is tricky.

I don't completely understand it yet, but it might be useful to buffer
a bit more than an epoch so we have enough data for compression. Right
now it feels like maybe I need a defree of freedom there.

was making too many lod's...fixed. Didn't effect throughput too much.

```c
// Orca Quest 2, splitting the fov into two color channels along y
const int downsample[] = { 2, 3 };
struct dimension dims[] = {
  { .size = 10000, .tile_size = 16,   .tiles_per_shard = 128,.name = "t" },
  { .size = 2,     .tile_size = 1,    .tiles_per_shard = 2, .name = "c" },
  { .size = 2048,  .tile_size = 128,  .tiles_per_shard = 9, .name = "y" },
  { .size = 2304,  .tile_size = 128,  .tiles_per_shard = 9, .name = "x" },
};
```

On oreb (5090):

```
=== bench ===
  GPU memory:  2.97 GiB device, 0.81 GiB pinned
    staging:   256.00 MiB   tile_pool: 0.56 GiB
    comp_pool: 0.56 GiB   aggregate: 0.56 GiB
    lod:       0.00 MiB   codec:     1053.35 MiB
    tiles:     576/epoch, 576 total (1 LOD levels)
  total:       175.78 GiB (94371840000 elements, 625 epochs)
  tile:        262144 elements = 512 KiB  (stride=262144)
  epoch:       576 slots, 288 MiB pool
  compress:    max_output=524322 comp_pool=288 MiB

  --- Benchmark Results ---
  Input:        175.78 GiB (94371840000 elements)
  Compressed:   22.03 GiB (ratio: 0.125)
  Tiles:        360000 (576/epoch x 625 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          21.90    46.76       2.57       1.20
  H2D             34.07    96.75       1.65       0.58
  Scatter         97.47   218.70       0.60       0.27
  Compress         4.82     4.87      58.34      57.71
  Aggregate     1672.22  1733.66       0.17       0.16
  D2H             53.34    53.45       5.27       5.26
  Sink             6.71   162.51       1.30       0.05

  Wall time:     52.821 s
  Throughput:    3.33 GiB/s
  PASS
=== bench_multiscale ===
  GPU memory:  4.44 GiB device, 1.02 GiB pinned
    staging:   256.00 MiB   tile_pool: 0.77 GiB
    comp_pool: 0.77 GiB   aggregate: 0.77 GiB
    lod:       684.00 MiB   codec:     1229.00 MiB
    tiles:     576/epoch, 792 total (13 LOD levels)
  total:       175.78 GiB (94371840000 elements, 625 epochs)
  tile:        262144 elements = 512 KiB  (stride=262144)
  epoch:       576 slots, 288 MiB pool
  compress:    max_output=524322 comp_pool=396 MiB
  LOD levels:  13

  --- Benchmark Results ---
  Input:        175.78 GiB (94371840000 elements)
  Compressed:   28.40 GiB (ratio: 0.162)
  Tiles:        360000 (576/epoch x 625 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Memcpy          21.86    45.43       2.57       1.24
  H2D             35.61    96.75       1.58       0.58
  Copy           434.98  1425.06       0.13       0.04
  LOD Scatter     16.54    16.68      17.01      16.86
  LOD Reduce     360.08   377.28       1.04       0.99
  LOD Gather      15.78    15.93      24.50      24.27
  Compress         5.83     5.89      66.32      65.63
  Aggregate      454.46   529.06       0.85       0.73
  D2H             53.15    53.27       7.28       7.26
  Sink             4.35  1657.26       0.65       0.00

  Wall time:     89.814 s
  Throughput:    1.96 GiB/s
  PASS  
```


## 2026-02-28

Cleaning up and moving things around.

## 2026-02-27

Cleaned up lod kernels. Benchmarking from `test_lod` on `auk`:

```
--- gpu_lod_3d_256 ---
  ds_mask=0x7  ds_ndim=3  batch_ndim=0  batch_count=1  nlev=9
  scatter    4.221 ms   15.90 GB/s
  pyramid    1.884 ms   35.62 GB/s
  total      6.105 ms   10.99 GB/s
  PASS
--- gpu_lod_3d_mixed_large ---
  ds_mask=0x6  ds_ndim=2  batch_ndim=1  batch_count=64  nlev=9
  scatter    0.838 ms   20.02 GB/s
  pyramid    0.254 ms   65.92 GB/s
  total      1.093 ms   15.35 GB/s
  PASS
```

Completely deleted the lod code I had (sans kernels) and started over.
Not supporting downsampling on the append dimension for the moment.

Got back to an end-to-end implementation mostly.

## 2026-02-26

Cleaning up the compacted morton algorithm a little bit. Making sure we're not
bit-limited on the morton code.

I think this will make the most sense before the chunk scatter. This doesn't
benefit from chunking, and it effectively needs it's own epoch size. That's
epoch is only 2 "deep". After we compute the lod's we need to scatter out of
the compacted morton order anyway, and that might as well be the chunk scatter.
After that we're just dealing with chunks and need to do some lod bookkeeping.

Need to think about generalizing the morton algorithm where there are dimensions
that are not downsampled... Also need to think about what happens when dim 0
(the append dimension) is downsampled.

When downsampling on dim0, then I think we need to aggregate 2^p elements along
dim0 before kicking off the lod, where p is the depth of the lod tree. That way
we can kick off the full lod reduction at once.

Then we scatter that to tiles. The tiles will fill at different rates based
on the lod. We'll need to track and spill epochs appropriately.

Oh wait, that 2^p dim0 elements is 2^p epochs. That's way too much.

## 2026-02-25

Considering multiscale using morton to order 2x2x..x2 regions.

Given a multidimensional array of rank d and shape s, I want to compute
level-of-detail representations by progressive reduction over 2x2x...x2 blocks
with replicate boundary conditions. I was thinking of an algorithm that first
scatters elements of the array according to their morton order. This way I can
just reduce each run of 2^d elements to compute the next lod level, then repeat
to do the next, etc.

In general though, the array shape is a small volume relative to the 2^p sized
d-dimensional box required to cover it; that box being the domain over which
the coordinates corresponding to each morton index would range.

To deal with that, I want to do the scatter but in a way that omits the
morton indices that are out-of-bounds. This means the runs of elements that we
need to reduce might be different sizes, that we need a way to efficiently
compute the indices to which array elements should scatter, and we need an
"index" array to track the range of each successive run of elements.

To scatter elements appropriately, I need to compute a compacted morton order.

A coordinate vector r from our array has components r_{d-1},...,r_1,r_0. A
morton index for this coordinates would be formed by interleaving bits of the
coordinates of r. For example, if r were 3d and r=(z,y,x) and we denote x(i) as
i'th bit of x, then morton(r)=z(p)y(p)x(p)...z(0)y(0)x(0) - the product there
representing bitwise concatenation, and p being some sufficiently large power
of 2.

If we iterate over morton indexes, they correspond to coordinates in a
d-dimensional box of size 2^p on each side. We choose p to be the smallest value
such that the box contains our array's shape. In general, many of these
coordinate are outside that 2^p-sized box.

Let's say I'm given a k such that k=morton(r). How can I efficiently compute
the number of indices from 0..k-1 that are within the array bounds.

I was thinking the algorithm would work by subtracting through the lowest d
bits, evaluating the intersection of the bounding box at that scale, and then
shifting off those bits and increasing the scale. That ends up being O(p*d^2).

## 2026-02-16

Adding multiscale. Starting with mean.

Also pulling out common error handling macros and trimming some of the old
unused kernels/tests that I added to when I was just trying to establish the
repo.

Downsampling kernel turned out pretty complicated. Seems like there's got to
be a way to simplify.

Each epoch we yield chunks across potentially several scales. It would be nice
if these could be handled uniformly to downstream compress/aggregation steps.
At that point, we're mostly doing the same things to the tiles, we just need
some more complicated shard addressing. But the mapping of tile indices to
lods/shards should be easy to compute.

We may need an extra epoch buffer just to keep the data moving since we'll need
to have two reserved for the downsampling.

Thinking about it a little more, we only emit on the odd epochs. So we need
to store all the lod's for the odd epoch, but the even epoch is effectively a
spare.

I may not need two epoch buffers if I'm not downsampling on the append
dimension?

Need to think about what the boundary condition for lods is. If we imagine
visualizing the array near the boundary as we zoom out, it's nice if there's not
a gap at the edge when we transition lods. So we want to pad a pixel.

In acquire-zarr, replicate padding is used. Odd dimensions are handled by
conceptually padding the array by 1 pixel (duplicating the edge), performing
the 2x2 average, and producing a ceil(N/2) output. The edge pixels get averaged
with copies of themselves rather than with zeros or neighbors, so there's no
darkening or artifact at the boundary.



TODO
- [x] cleanup
- [x] uniform handling of tiles across lods for compress and aggregate
- [x] uniform handling of tiles for shard writer
- [x] shard writer handles lods
  - could still improve this a bit probably
- [x] replicate boundary condition
- [x] support floats
- [x] min, max, median, 2-max
- [x] add metrics, bench
- [x] verify the multiscale zarr is visualizable
- [x] make sure we're using the right condition to stop downsampling. all
      dims need to be bigger than tile size



## 2026-02-15

Testing on 5090 (oreb). New data distribution hits pretty hard :(

```
=== test_bench_zarr ===
  output: .\build\test_output.zarr
  shape:       (4264, 2048, 2048, 3)  tiles: (32, 128, 128, 1)
  total:       99.94 GiB (53653536768 elements, 134 epochs)
  tile:        524288 elements = 1024 KiB  (stride=524288)
  epoch:       768 slots, 768 MiB pool
  compress:    max_chunk=1048636 comp_pool=768 MiB

  --- Benchmark Results ---
  Input:        99.94 GiB (53653536768 elements)
  Compressed:   40.86 GiB (ratio: 0.407)
  Tiles:        102912 (768/epoch x 134 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Source           2.18    22.00      28.70       2.84
  H2D             34.66    53.90       1.81       1.16
  Scatter         75.03   119.36       0.83       0.52
  Compress         3.74     8.19     200.53      91.57
  Aggregate       41.38    50.66      18.13      14.81
  D2H             45.15    53.45      16.61      14.03
  Sink             3.28  2976.18       5.44       0.01

  Wall time:     46.107 s
  Throughput:    2.17 GiB/s
```

Trying to see if I can view the zarr in neuroglancer.

It's not quite as simple as

```powershell
uv run python -m http.server 8080
```

The server also needs to set CORS headers, and use the `RangeHTTPServer`.

After that, you can open [neuroglancer] and set source to  `zarr://http://
localhost:8080/test_output.zarr/0`. You have to point it at an array.

I added a "visual" option to `test_bench` that outputs something a little
easier to look at, and confirmed it works in napari.

Next is probably to start thinking about multiscale. 

I'm still a little annoyed by the performance. The compression is pretty heavily
the bottleneck.

Found another unnecessary sync in the compression pipeline. Removing that 
and putting compress+downstream on another stream got it to 2.9 GB/s (without
the write to disk).

[neuroglancer]: https://neuroglancer-demo.appspot.com/

## 2026-02-14

### Profiling

Trying to look at nsight on the 5090 (oreb).

I noticed I was doing 2-byte loads for the transpose. Changed to 4-bytes.
Can do larger, but that just hits more cache lines and uses more registers.

```
           GB/s   avg     best   
Scatter         78.09   115.40 -- Before
Scatter         78.28   115.94 -- Before
Scatter         79.50   118.56 -- After
Scatter         79.69   119.00 -- After
Scatter         69.62   118.87 -- After
```

Throughput by wall time is 12 GB/s.

So faster but not a lot faster. nsight predictably complains about non-coalesced
stores.

### Sharding

I'll want to write a cuda kernel to aggregate the compressed chunks by shard. My
guess is that will look like a segmented scan. This will necessrily generate the
list of offsets for each chunk. The chunks will get reordered to group them by
shard. That reorder is conceptually the same kind of lifting/transpose that was
done to scatter the input  into the chunks. This time though, we're transposing
the logical tiles into something tiled by the `tiles_by_shard`.

I'm thinking that array of offsets + the aggregated bytes are enough to transmit
back to the host. The host side will have to do some index operations to map
each chunk to a shard and figure out how to update the chunk index for that
shard.

We'll probably want to keep the index for each shard in memory. That'll help
with the crc32 checksum (which just checksums the index). Shards will complete
in layers/epochs, just like tiles, so we can limit the amount of memory
required.

Added the kernel for aggregating the compressed chunks into shards. It's a
three step process where we permute the compressed sizes, do a scan to compute
offsets and then a gather to copy the compressed chunks into shard order.

The aggregated compressed bytes and the offsets buffer then get copied back
to host. The host side figures out how to dispatch those ranges to shards, and
maintain the shard indices. I abstract the shard writer for now.

Wrote a test that uses a coordinate encoding to:
- verify the right chunks are getting copied into the right places (and values)
- verify the chunk index for the shard looks right

Now I think it's time to do the zarr writing. This is tricky because it needs a
few things all at once.

1. Need to write json
2. Need to create the directory structure, probably epoch by epoch. That
   should use a background thread, a job queue, and some barrier.
3. Need to implement the shard store.

Ok, most of that is done. Need to review in the morning, but it looks like
things are working.

Changed the input distribution on the benchmark to be:
- 1/3 12-bit gaussian
- 1/3 uniformly random
- 1/3 constant

On auk (laptop, 5070)

```
Stage        avg GB/s best GB/s     avg ms    best ms
Source           1.78    20.69      35.08       3.02
H2D              9.86    13.47       6.35       4.64
Scatter         26.08    48.16       2.40       1.30
Compress         2.59     3.40     290.05     220.35
Aggregate       11.98    12.96      62.58      57.86
D2H             12.46    13.32      60.21      56.30
Sink             3.97  5867.83       3.24       0.00

Wall time:     56.510 s
Throughput:    1.77 GiB/s  
```

### Thoughts

It occurred to me that if the tile shape has 1's in the larger dimensions we
can yield epochs more quickly - should check on that.

Are we waiting until the first epoch needs to flush to alloc the codec ctx.

## 2026-02-13

Thinking about aggregating by shard.

Could scatter to shard, but shards can be sparse and shouldn't be zero padded.
So I don't think this is the right approach.

In the end, we'd like to have 1 write of a contiguous set of bytes to update
a shard, and then we'd like to do 1 write to update that shards index, which
uses an implicitly indexed `(offset, length)` format with a checksum over the
indices.

Logically, sharding is another lifting from an array shape, though because
shards are sparse, the shape we're talking about can be much larger than the
shape of the input array - it must cover though.

If we specify

```c
struct dimension
{
  uint64_t size;
  uint64_t tile_size;
  uint64_t tiles_per_shard; // new
};
```

The the size of the covering array is `tile_size*tiles_per_shard` along a
dimension. Let's call that $e_d$. $e_d$ can be factored into $s_d * t_d * n_d$
where $s_d$ is the shard index, $t_d$ is the tile index, and $n_d$ is the
pixel index. Coordinates in the covering array are lifted:

$r_d = (s_{d-1},t_{d-1},n_{d-1},...,s_0,t_0,n_0)$

and (virtually) transposed

$r'_d = (s_{d-1},...,s_0,t_{d-1},...,t_0,n_{d-1}...n_0)$

Focusing on the output side, I just need to prepare the data for updating
shards following the [zarr sharding codec].

[zarr sharding codec]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html 

## 2026-02-12

Not a lot of memory on auk (8GB). Kind of forces me to thin chunks so the
epochs aren't too thick.

Trying to make sure the timings make sense....

Overall wall tie looking better - 6.3 GB/s on auk.

Could think about file io. Probably need to do coalescing across chunks first.
Should think about sharding. Might be a kv abstraction there.

There's also the question of multiscale. Getting file io first might clarify.

I think my todo list at this point is:

0. clean up
1. try to get async compression working again
2. aggregate by shard? compaction.
3. ...

Got through a fair amount of clean up.

Looks like I can remove the excessive sync point now. Still has to sync before
the d2h, of course.

## 2026-02-11

Looked back over streaming code and marked some todo's etc. One thing is that
I haven't really thought cleanly about draining data out and how that affects
the calling thread.

Also reviewing the benchmark...
Changing the chunk size and front side buffer. Front side buffer was small.

Need to keep adjusting and double check things make sense. Still not quite
happy with the chunk size - would be nice to get more epochs through for
stats on timings.

## 2026-02-09

Getting the build to work on windows.

On my 5090:

```
--- Benchmark Results ---
Input:        49.97 GiB (26826768384 elements)
Compressed:   16.07 GiB (ratio: 0.322)
Tiles:        51168 (96/epoch x 533 epochs)

GPU step     avg GB/s best GB/s     avg ms    best ms
H2D             33.69        -       2.78          -
scatter        834.68  1202.66       0.11       0.08
compress         6.42    15.39      14.61       6.09
D2H             47.64    53.36       1.97       1.76

Source fill:   3.31 GB/s  (15099.5 ms total)
Sink:          11319.2 ms total  (11 epochs verified)

Wall time:     37.555 s
Throughput:    1.33 GiB/s
```

## 2026-02-08

Refactor to rely on the writer interface more. Decide on some compositional
boundaries. Might be some useful writers for testing. Think about getting the
compression in. I'd like to refactor the platform specific code (timers) as
well.

Added double buffering on the host input. Needed to also setup the wait for
the host side buffer to clear.

Added compression. Compressed tiles are not compacted at the moment.

I think I'd need to instrument before deciding if compaction is worth it.

The "tile_writer" interface is awkward. It looks like a writev, but doesn't
follow the same symmantics as the "writer interface"; it has to consume all
input.

I don't like the way the different code paths for w/wo compression are handled.
Would be nice to find a more composable thing. Need to clean up the code.

Have to do a hard sync on the compute stream to properly finish the compression
pass - doesn't seem right - need to look into that more.

At least on my laptop (auk) throughput is decent through the kernels. The
wall tije is bad though - 1 GB/s end to end. Table below:


```
  Benchmark Results (50 GiB, mixed data)

  GPU step     avg GB/s best GB/s     avg ms    best ms
  H2D             12.18        -       7.70          -
  scatter        341.81   464.37       0.27       0.20
  compress         3.71     8.91      25.29      10.52
  D2H             13.08    13.31       7.17       7.04
```

## 2026-02-07

I'm trying to get the simplest end-to-end pipeline I can for a streaming
transpose of multidimensional arrays/tensors. I think for streaming, I'll have
to introduce the concept of tiling.

The reason for that is that I need to define when to stream things back to the
host. Tile's make sense, and that's what I want to work toward anyway. Input
data comes in, gets scattered to different tiles, and then when a tile is filled
up, we should transfer it back. It'll probably be the case that many tiles get
filled up at once.

When we push some buffer to the transpose it represents some [beg,end) range of
input indices. These map to an ordered set of coordinates r(beg)..r(end) where
r=(t_{d-1},n_{d-1},...,t_0,n_0). If we just consider the tile indices, we can
write r_t=(t_{d-1},...,t_1,t_0). The sequence r_t(beg)...r_t(end) corresponds
to an ordered set of tiles - with tile indices corresponding to the tile-only
stride projection in the row-major input space.

The input sequence beg..end will map to a corresponding sequence of tile
indices beg_t..end_t. The sequence of tile indices is not strictly increasing.
That's because a carry in one of the n_i dimensions, will carry into t_i and
that in turn might induce a carry (to n_i+1), causing t_i to reset to 0
(decreasing the tile index).

But we should be able to track the min and max tile indices that might be hit
by [beg,end). Let's say u_t is the index of the first element of the input to
hit tile t, and v_t is the last. If beg>v_t, then we know t won't be hit again
and can flush the tile.

The max t for which v_t < beg should be strictly increasing. Similarly the min
t for which u_t < end should be strictly increasing. I'm thinking we'll need to
maintain a ring-buffer of tiles. Tiles finish in the same order they start, so
we can think about a fifo.

The input partitions in to non-overlapping epochs - these are "layers" of tiles
formed from all the dimensions except $$t_{D-1}$$. This gives the size of
fifo. It needs to hold that layer of tiles.

Actually, we need to keep the layer around anyway for compression, which means
we don't really need to think of this as a ring-buffer type of fifo. We just
need to make sure the append() call enforcese the split across layers. Will
need to rezero layers for padding tiles.


## 2026-01-25

Realized I've been making all of this too complicated. The vadd() algorithm
is nice off of the gpu, or when you have a big buffer of indices you write to.
It ends up collapsing the complexity of the loop over dimensions into what
would normally be the loop over elements, so you end up with something O(n)
rather than O(n x d).

But, we're not doing that on the gpu, so the simple unravel-and-accumulate
strides algorithm works. It's O(d) per thread.

Anyway, next step is to actually do the transpose without thinking too much
about memory coherence.

## 2026-01-24

Ok, it turned out to be very easy. I just need to stick to the mixed-radix
addition algorithm.

Added avx2 versions of `vadd` and `vadd2` to make sure I understood how
the vectorization works. The approach in `vadd2` is probably better forev
vectorization and it's simpler to understand, so I used this for both
of the vectorized implementations. They're not very fast - the non-vectorized
`vadd` is very fast. But that's not the point here. 

## 2026-01-24

Trying to see if I can get non-unit step sizes to work. The idea is that the
step should translate into a $\delta r$ in the input array. Each step we're
adding $\delta r$ in the input array space with carries. To compute the output
indices we just keep track of the effect of those carries, just like we did for
the unit-step case.

At the end of the day, I feel like I'm close. I'm stuck on how to handle
carries. I think this probably ends up being another scan.

## 2026-01-23

Working out the output index computation that needs to happen in the kernel.
I'm doing this by prototyping in `tests/index.op.c`.

The easy way to compute the output index is to just take the input index,
unravel it into a coordinate vector, and then project that vector with the
output strides.

However, in the kernel each thread would need O(rank) space and O(rank) time.

The idea is to use a prefix sum approach instead. If we track the change in the
output index for each step along the input index, then we just have to watch
for when the input index reaches a boundary in the array shape. When those
boundaries are hit, we can look up the effect in the output strides and correct
the steps. To get the final output index we just do a cumulative sum over the
deltas. It's possible to do this without storing a coordinate array.

I've tested it for unit step sizes of the input index. I think there's a way
to get that approach to work for larger steps. That's important because we
have to know the output offsets where each warp starts.

## 2026-01-20

Got a simple kernel running in test with a basic cmake-based build. The
kernel just fills a buffer and then it gets streamed back. D2H transfers
happen at around 13 GB/s. This machine has a 4x PCIe Gen4 link to the GPU.
I believe that's a 16 GB/s theoretical max (~83%).

Next is to actually try to write the transpose. I should just try the basic
thing without trying to optimally fill output lines.

What do I want to do first?

The input is going to be a 1GB buffers that contain samples starting at some
offset relative to the beginning of a contiguous array. 

We'll process the data in 32x32 warps. Each warp is responsible for loading
a contiguous 128 bytes (1024 bits) of data: 32 lanes/warp * 4 bytes per lane.
Depending on the sample type, there will be from 32-128 samples per warp.
Let's call that number $B$: the number of samples handled by a warp.

Note that the data we load from warp-to-warp doesn't have to be contiguous.
To keep things simple, we won't worry about that right now: warp $i$ should
load elements from $i*B$ to $(i+1)*B-1$.

Note that at this point we haven't used any information about the shape of
the array.

Then we need to compute where there should go.
