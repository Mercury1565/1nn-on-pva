# Bucket preparation (`bucket_prep.cpp`)

This is what happens on the CPU before the PVA starts: Part A of the method, Steps 1–5.
The code is in [../src/bucket_prep.cpp](../src/bucket_prep.cpp); Step 5 is in
[../src/knn1_bucket_s16.cpp](../src/knn1_bucket_s16.cpp) (`prepare()`).
The same rules are written in Python in `src/knn1_bucket_model.py` (`prepare()`), which is
the answer key.

Numbers are for the default build, 128 ref points per bucket (`K1_BUCKET`, CMake option; 256 and
512 also build).

## Words used here

| Word | Meaning |
| --- | --- |
| `pc_queries` | Point cloud P1 (frame t). Every point needs its nearest neighbour. |
| `pc_ref` | Point cloud P2 (frame t-1). The candidate points. |
| query point / ref point | One point of `pc_queries` / `pc_ref`. |
| planar | Memory layout of a cloud: all x values, then all y, then all z. |
| bucket | `K1_BUCKET` (128) ref points that are close together, plus their bounding box. |
| team | 16 query points that are close together. The PVA works on one team at a time. |
| slot | A place in the bucket table: `slot = bucket * K1_BUCKET + position`. |
| lane | One of the 16 number slots of a VPU vector register (see below). |
| leaf | The group size used by the splitting: `K1_BUCKET` for buckets, 16 for teams. |
| chunk | 32 teams stored in one DRAM row and moved in one DMA transfer. |

## Inputs and outputs

**Inputs.** `pc_queries` and `pc_ref`: int16 coordinates in centimetres, planar `[3][n]`,
with 1 ≤ n ≤ 131072 for each cloud. Coordinates are clamped to ±120 m before quantization
(±12000 cm). That way a squared distance is at most 3 × 24000² = 1,728,000,000, which fits
in int32.

**Outputs, written to DRAM for the PVA** (layout comment in
[../vpu/knn1_bucket_s16_params.h](../vpu/knn1_bucket_s16_params.h)):

| Table | Shape (int16) | Content |
| --- | --- | --- |
| Bucket table | 1024 rows × 384 | Row b = x[128] \| y[128] \| z[128] of bucket b |
| Team table | 256 rows × 1536 | Row c = 32 teams, each x[16] \| y[16] \| z[16] |
| Box table | 6 × 1024, in the 4 rows after the team table | Min x, min y, min z, max x, max y, max z of every bucket |

(The box table shares the team table's DRAM buffer because the PVA's DMA has a limited number of
transfer triggers per program; see the VMEM doc.)

**Outputs kept on the CPU** (in `Bookkeeping`), used by `results()` to translate the PVA's
answers back into the caller's numbering:

- `refOfSlot[slot]`: which point of `pc_ref` sits in that slot of the bucket table.
- `queryOfLane[team * 16 + lane]`: which point of `pc_queries` sits in that team and lane.

## The grouping machine: `medianSplit()` / `split()`

Both clouds are grouped with the same function. Only the group size differs.

1. **Copy the cloud** into working arrays: x, y, z and the original point number `id`, one array
   each (`SplitPoints`). The points themselves are moved around, so every later step reads
   memory in order.
2. **Find the widest axis.** For the points in the current range, find which of x, y or z
   spans the largest distance. On a tie, the first axis wins, as in numpy.
3. **Choose the left part's size.** It is about half the points, rounded up to a whole
   number of groups: `nLeft = ceil((size/2) / leaf) * leaf`.
   Example: 1000 ref points with groups of 128 gives 512 on the left (four full buckets) and
   488 on the right. Because the left part is always whole groups, only the very last
   group of the whole cloud can be partly filled. Steps 2 and 4 rely on this.
4. **Find the median value by counting.** Count how many points have each coordinate value
   along that axis (counters only for the values between the range's min and max: one pass if
   the span is under 4096 cm or under the number of points, otherwise two passes, high byte then
   low byte). Walking through the counters gives the value `m` of the `nLeft`-th smallest point
   and how many points are smaller.
5. **Move the points** into the second buffer in one pass: smaller than `m` to the left part,
   larger to the right part, equal to `m` to the left until it is full. The order inside each
   part is kept.
6. **Repeat on both parts** until a range holds one group or less. The two buffers swap roles at
   every level; a finished group is copied into the result buffer if it ended up in the other one.

Afterwards, every run of `leaf` consecutive points is one group of points that are close
together in space.

**Why counting instead of `std::nth_element`.** The first version used `nth_element` on an index
array. On point data it mispredicts about every second comparison, and it read the coordinates
through the index array. The counting version has no branch that depends on the data. It produces
the same groups (same left/right sets at every split; only the order inside a group can differ,
which nothing depends on) and is faster: on loki, `prepare()` went from 21.4 ms to 15.0 ms on one
thread (kitti_a). Tried and not faster on loki: packing a point into one 64-bit word, four sets of
counters, a branch-free store index (they helped on the x86 build host, not on loki's cores).

**Threads.** `init(vpus, cpu_threads)` sets how many CPU threads `prepare()` may use (default 1).
With 2 or more, the query split runs at the same time as the ref split and bucket filling, and
within each split the two halves of a range are split on separate threads, each with half of the
threads. The result is the same for every thread count. Measured on loki (kitti_a, p50):

| `cpu_threads` | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| `prepare()` | 15.0 ms | 8.5 ms | 4.8 ms | 3.3 ms |

More threads finish sooner but occupy more CPU cores while `prepare()` runs; the total CPU work
stays about the same.

**Why the median split and not Morton (Z-order).** Both were measured on the 200 KITTI frame
pairs (`src/bucket_compare.py`, `results/bucket_compare.csv`). With the median split,
buckets are tighter and the search needs about 4× fewer distance evaluations.

## `prepare()`, step by step

### Sizes

```
nBuckets      = ceil(nRef / 128)
nBucketGroups = ceil(nBuckets / 16)
paddedBuckets = nBucketGroups * 16
nTeams        = ceil(nQuery / 16)
```

Example, kitti_a (122,919 ref points, 122,987 query points): 961 buckets → 61 groups → 976
padded buckets; 7,687 teams in 241 chunks.

### Step 1: order `pc_ref` into buckets

`medianSplit(pc_ref, nRef, 128, refs, threads)`. Afterwards sorted points 0..127 are bucket 0,
128..255 bucket 1, and so on.

### Steps 2 and 3: fill the bucket table and the boxes

For every bucket, copy its points' x, y and z into its row and their original numbers into
`refOfSlot`, and compute its box (min and max of x, y, z over its real points):

| Case | Source | Why it keeps the result exact |
| --- | --- | --- |
| Normal slot | The sorted point | — |
| Past the last real point (the partly filled last bucket) | Copy of the last real point | A copy is never closer than the original, and the bucket's box does not grow. |
| Bucket ≥ `nBuckets` (padding up to a multiple of 16) | Copy of the last real bucket | The PVA is given `nBuckets` and never visits these buckets. |

### Step 4: order `pc_queries` into teams

1. `medianSplit(pc_queries, nQuery, 16, queries, threads)`: each team is 16 query points
   that are close together, so they tend to need the same buckets.
2. A partly filled last team is filled with copies of its last query point. Their results
   are computed but never read back.
3. Each team is written as x[16] | y[16] | z[16], with 32 teams per DRAM row (a chunk).
   `queryOfLane` records the source point of every team and lane.

### Step 5: tell the PVA the sizes

In `Knn1BucketS16::prepare()`, the host writes small variables into the local memory of each
VPU program (VMEM scalars, `prog[v]["nBuckets"] = …`): `nBuckets`, `nBucketGroups`, the program's
team range `teamBegin` / `teamEnd`, and `vpuIndex`. With two VPUs, each gets one contiguous half of
the chunks: neighbouring teams need the same buckets, so each VPU's bucket cache only has to hold
its half of the scene. (Alternating chunks between the VPUs balanced the work within 1% but
doubled the bucket copies from DRAM and was not faster.)

## Memory at the end of preparation (what the PVA starts from)

All tables are flat int16 arrays in DRAM, stored row after row. The CPU and the PVA share this
memory: the CPU writes through each buffer's host pointer, and the PVA's DMA reads through its
device pointer. Nothing is copied. Each buffer is reserved once in `init()` for the maximum size
(131,072 points), and a frame fills only the leading part.

**Bucket table**: one row per bucket. 384 int16 per row = 768 B; 1024 rows reserved (768 KB).
`index = bucket*384 + axis*128 + position`

```
           |←── 128 x ──→|←── 128 y ──→|←── 128 z ──→|
row 0      x x x … x      y y y … y      z z z … z       bucket 0
 ⋮
row 960                                                  last real bucket (partly copies)
row 961–975                                              padding: copies of bucket 960
row 976–1023                                             not written for this frame
```

**Team table**: one row per chunk of 32 teams; each team is x[16] | y[16] | z[16] (48 values).
1536 int16 per row = 3 KB; 256 rows reserved, then 4 rows for the box table.
`index = (team/32)*1536 + (team%32)*48 + axis*16 + lane`

```
         |←──────────── one row = 32 teams = 1536 values ────────────→|
row 0    [x×16 y×16 z×16] [x×16 y×16 z×16] … [x×16 y×16 z×16]    teams 0–31
row 240  teams 7680–7686, then 25 unused team spaces
row 241–255  not written
row 256–259  box table: boxes[row * 1024 + bucket], rows min x, min y, min z, max x, max y, max z
```

**Also ready when preparation ends:**

| What | Where | Used by |
| --- | --- | --- |
| `nBuckets`, `nBucketGroups`, `teamBegin`, `teamEnd`, `vpuIndex` | VMEM variables of each VPU program | VPU program, at start-up |
| `refOfSlot`, `queryOfLane` | CPU memory (`Bookkeeping`); the PVA never sees them | `results()` |

**How the PVA reads the tables.** The box table is copied into VMEM once per run. The team table
is loaded one row (32 teams) per transfer. A bucket row is loaded only when a team needs a bucket
that is not already in the VPU's bucket cache. Every row is a whole number of 64-byte pieces
(bucket rows 768 B = 12 × 64, team rows 3,072 B, result rows 4,096 B). 64 bytes is the PVA's DMA
alignment unit (cuPVA API docs in the reference corpus).

## Decisions that come from the hardware

The VPU facts come from the Thor VPU Programmer's Guide in the reference corpus:
`~/pva-work/references/sdk/guides/thor/03-vpu-core.md`, §3.3 Register Files, guide p. 19.

### What the VPU looks like

The VPU has 32 vector registers, each 384 bits wide. The data type decides how a register
is divided into **lanes**, the number slots that one instruction processes together:

| Mode | Lanes × bits | Used for |
| --- | --- | --- |
| Halfword | 16 × 24 | Coordinates and box corners (int16) |
| Word | 8 × 48 | Squared distances (two registers hold 16 results, the `dvintx` type) |
| Byte | 32 × 12 | Not used |

An int16 value loaded into a 24-bit lane is **sign-extended**, which gives headroom so
subtractions cannot overflow. Squares do not fit in 24 bits, so the squaring step (`vsumsq`)
returns 48-bit word lanes.

### Choices in the preparation that follow from this

| Choice in `bucket_prep.cpp` | Hardware reason |
| --- | --- |
| **Coordinates are int16** (cm, clamped to ±120 m) | They use halfword lanes, 16 per register. Differences fit in the 17-bit inputs `vsumsq` accepts, and squared distances fit in int32 in memory and in 48-bit lanes in registers. |
| **Planar layout** (x block, y block, z block) for clouds, buckets and teams | A vector load reads consecutive values. With planar storage, one load gives 16 x values ready to use. The bucket's x, y, z rows go to three different VMEM superbanks so the scan reads them in the same cycle. |
| **Team = 16 query points** (`K1_TEAM`) | One query point per halfword lane. The scan of a bucket compares one ref point with all 16 query points in one vector operation. |
| **Bucket group = 16 buckets** (`K1_GROUP`) | One vector holds 16 buckets' box values; ranking handles 16 buckets (or 16 groups) at once. |
| **Box table stored per coordinate** (6 rows × 1024 columns) | One 16-value load gets, say, the min x of 16 neighbouring buckets. |
| **Bucket count padded to a multiple of 16** | Ranking always processes whole groups of 16. The padding buckets are copies, so their boxes are valid, and the PVA skips them using `nBuckets`. |
| **Maximum 131,072 points per cloud** | 1024 buckets of 128 points. A bucket number fits in 10 bits; the ranking key stores it in its low 10 bits (`K1_KEY_ID_BITS`). |
| **Position in a bucket fits in 7 bits** | The scan returns `(d2 << 7) + position` per lane: 31 + 7 = 38 bits in a 48-bit word lane, so one vector minimum keeps the closest distance and its position together. 128 is also a multiple of 16. |
| **One bucket = one DRAM row** | The VPU fetches a bucket chosen at run time with one gather (GSDF) transfer: three tiles (x, y, z) into the three superbanks. |
| **32 teams per row** (chunk) | Fewer, larger DMA transfers for the teams and results. A design choice, not a hardware limit. |

### Choices that are *not* from the hardware

- **128 points per bucket** was chosen by measurement on loki with the final kernel: fewer ref
  points are scanned in total, and the extra ranking and fetching work per bucket stays small
  (`results/pva_optimization_20261114/`).
- **Median split rather than Morton** was chosen from measurements (see above).

### Hardware details the preparation does not need to know

`vsumsq` returns its 16 results with the even lanes in one register and the odd lanes in
the other (Thor guide §6.2.3.6, p. 51). The VPU program writes its answers in that "slot" order
and `results()` maps them back, so the tables written here are in plain bucket and lane order.

## Cost

On loki (MAXN, clocks not pinned), `prepare()` for kitti_a takes 15.0 ms (p50) on one thread,
3.3 ms on eight, of about 29 ms end to end with one thread. Not done: since frame t's
`pc_queries` is frame t+1's `pc_ref`, its bucketing could be done once and reused; that needs the
teams to be a subdivision of the buckets, which changes the method (model first).
