# knn1: nearest-neighbour distance between two point clouds on the PVA

Given two point clouds, `pc_queries` (for example LiDAR frame t) and `pc_ref` (frame t-1),
this finds, for every query point, the nearest ref point. The search runs on the Jetson
Thor's PVA (Programmable Vision Accelerator).

The output per query point:

- `dist2`: the squared distance to its nearest ref point, in cm² (exact integer).
- `idx`: the index of that ref point in `pc_ref`.

The cloud distance is `sqrt(sum of dist2)`.

The result is **exact**: it is the same answer a brute-force comparison of every query point
with every ref point would give on the same centimetre coordinates. If two ref points are
equally close, either index may be returned.

## Quick start

`$KNN1` below is the folder this package is installed in (the one holding this README).

```
make -C $KNN1 run        # cloud distance for the frame pair set in the Makefile
make -C $KNN1 python     # the same through Python
make -C $KNN1 c          # the same through C
make -C $KNN1 selftest   # self-test on random clouds
```

Choose the scans by editing the settings at the top of the `Makefile`, or on the command line:

```
make -C $KNN1 run KITTI_ROOT=/path/to/kitti DATE=2011_10_03 \
     DRIVE=2011_10_03_drive_0027_sync REF_FRAME=0000000076 QUERY_FRAME=0000000077
make -C $KNN1 run PC_REF=/path/a.bin PC_QUERIES=/path/b.bin
```

Results go to `~/knn1_out`, and the compiled C example to `~/knn1_build`. Both are in your own
home folder; change `OUT_DIR` and `BUILD_DIR` to put them elsewhere.

You need to be in the `video` group to use the PVA (check with `id`).

## How it works

A call goes through six steps. Steps 1–3 and 5–6 run on the CPU, step 4 on the PVA.

**1. Convert the coordinates.** The PVA works with 16-bit integers, so every coordinate is
clipped to ±120 m and rounded to the nearest centimetre. A squared distance then always fits
in a 32-bit integer.

**2. Sort `pc_ref` into buckets.** The ref points are split into groups of 128 points that are
close together in space. The split always cuts the widest side of a group at its median
point, again and again, until the groups have 128 points. For each bucket, its bounding box
(the smallest and largest x, y and z) is recorded. A 120,000-point scan gives about 940
buckets.

**3. Group `pc_queries` into teams.** The query points are split the same way, into teams of
16 neighbouring points. 16 is the number of values the PVA's vector unit processes in one
instruction, so the 16 points of a team are always handled together. Neighbours tend to need
the same buckets, which is why teams are built from nearby points.

**4. Search on the PVA, one team at a time.** The PVA has two vector processors (VPUs); each
takes half of the teams.
1. **Rank the buckets.** For every bucket, compute the smallest distance any of the 16 points
   could possibly have to it, using only its box. A point inside the box could be at distance
   0; a point outside cannot be closer than the gap to the box.
2. **Visit buckets nearest-first.** Each point keeps its best distance found so far, starting
   at "infinity". For the nearest remaining bucket, the PVA does one of three things:
   - **stop** if the bucket cannot be closer than the worst of the 16 current bests (no
     later bucket can then improve any of the 16);
   - **skip** it if none of the 16 points could find anything closer in it;
   - otherwise **scan** its 128 points, computing the exact squared distance from all 16 points
     to each of them, and keep each point's minimum. Each VPU keeps the most recently used 480
     buckets in its own memory, so most buckets do not have to be fetched again; neighbouring
     teams need mostly the same buckets.
3. **Write out** the 16 answers.

Because a bucket is skipped only when its box proves it cannot help, the answer is exact. In the
KITTI scans tested, a team scanned on average about 3 of the ~960 buckets.

**5. Put the answers back in order.** The answers are translated back to the original point
order of `pc_queries`, and to indices in the original `pc_ref`.

**6. Cloud distance.** The sum of all squared distances is computed in 64 bits; the cloud
distance is its square root.

## Usage examples

### Command line

```
$KNN1/bin/knn1_bucket_s16_kitti <pc_ref.bin> <pc_queries.bin> [out_dir]
```

The inputs are KITTI velodyne scans: float32 x, y, z, reflectance per point, in metres.

```
$ make -C $KNN1 run
pc_ref     …/0000000089.bin: 122919 points (0 coordinates clamped)
pc_queries …/0000000090.bin: 122987 points (0 coordinates clamped)
buckets 961, teams 7687, buckets fetched 22449 (2.92 per team)
sum d2 = 51148720 cm^2
cloud distance sqrt(sum d2) = 7151.8 cm = 71.518 m
mean nearest distance (RMS) = 20.39 cm
host wall time, single call (not PVA performance): prepare 17.3 ms, submit+wait 13.9 ms, results 0.5 ms
wrote …/knn1_out/dist2.s32 and …/knn1_out/idx.s32
```

`dist2.s32` and `idx.s32` hold one int32 per query point, in the order of the query file:

```python
import numpy as np
dist2 = np.fromfile("knn1_out/dist2.s32", dtype=np.int32)   # cm^2
idx = np.fromfile("knn1_out/idx.s32", dtype=np.int32)
```

### Python

```python
import sys
sys.path.insert(0, "$KNN1/python")
import knn1_distance as k1

pva = k1.Knn1()                                   # create once, reuse for every pair

pc_ref = k1.load_kitti_bin("frame_089.bin")       # float32 (n, 3) in metres
pc_queries = k1.load_kitti_bin("frame_090.bin")
dist2, idx, cloud_dist_cm = pva.nearest(pc_queries, pc_ref)

print(cloud_dist_cm / 100, "m")                   # 71.518 m
print(idx[0], dist2[0] ** 0.5, "cm")              # nearest ref point of query point 0
```

Any float32 array of shape (n, 3) or (n, 4) in metres works as input, not only KITTI files.
For a sequence of frames, reuse the same `Knn1()`:

```python
frames = [k1.load_kitti_bin(p) for p in sorted_paths]
for prev, cur in zip(frames, frames[1:]):
    _, _, d = pva.nearest(cur, prev)
    print(d / 100, "m")
```

Full example: `examples/example.py` (run with `make python`).

### C / C++

Include `include/knn1_distance.h` and link `lib/libknn1_distance.so`:

```c
#include "knn1_distance.h"

knn1_ctx *ctx = knn1_create();                       /* once */

int16_t *q = malloc(3 * nq * sizeof(int16_t));       /* planar: all x, then all y, then all z */
int16_t *r = malloc(3 * nr * sizeof(int16_t));
knn1_quantize_cm(query_xyz, nq, 4, q);               /* float metres -> int16 cm; 4 = KITTI stride */
knn1_quantize_cm(ref_xyz, nr, 4, r);

int64_t sum_d2;
int rc = knn1_nearest(ctx, q, nq, r, nr, dist2, idx, &sum_d2);
if (rc != 0) printf("%s\n", knn1_error_string(rc));
printf("cloud distance %.3f m\n", sqrt((double)sum_d2) / 100.0);

knn1_destroy(ctx);
```

Build:

```
gcc -O2 my_program.c -I $KNN1/include -L $KNN1/lib -lknn1_distance -lm \
    -Wl,-rpath,$KNN1/lib -o my_program
```

Full example: `examples/example.c` (build and run with `make c`).

## Limits

- **Size:** 1 to 131,072 points per cloud (a KITTI scan is about 120,000).
- **Range:** coordinates beyond ±120 m are clipped to ±120 m (the program reports how many
  were).
- **Precision:** coordinates are rounded to whole centimetres before the search. Distances
  are exact for the rounded points.
- **Threads:** one call at a time per `knn1_ctx` / `Knn1()`; use one per thread.

## Speed

These numbers are preliminary: clocks are not pinned and the machine is shared. One KITTI
frame pair (~123,000 × 123,000 points, kitti_a), median of the timed calls:

| Step | Time |
| --- | --- |
| 1–3, preparation on the CPU (one thread) | ~15 ms |
| 4, search on the PVA (both VPUs) | ~14 ms |
| 5–6, answers back | ~0.3 ms |
| Whole call | ~29 ms |

The first version took about 125 ms (105 ms on the PVA). Preparation can use more CPU threads
(`init(2, threads)` in C++), which shortens it to ~3 ms with 8 threads but occupies more cores
while it runs. A CPU KD-tree did the same search in about 32 ms in earlier tests on the same
machine; this operator now takes about the same time per call, of which ~15 ms is CPU work on one
core and the search itself runs on the PVA, leaving the GPU free.

## Files

```
Makefile   run / python / c / selftest shortcuts and their settings
bin/       knn1_bucket_s16_kitti (command line), knn1_bucket_s16_sample (self-test)
include/   knn1_distance.h (C interface)
lib/       libknn1_distance.so
python/    knn1_distance.py (Python interface)
examples/  example.c, example.py
VERSION    build and verification details
```
