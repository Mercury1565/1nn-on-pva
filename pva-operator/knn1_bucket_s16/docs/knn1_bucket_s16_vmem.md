# VMEM layout of the VPU program (`knn1_bucket_s16_dev.c`)

What the VPU program keeps in the PVA's local memory (VMEM), where it is, what it holds, and
what it is used for. The declarations are at the top of
[../vpu/knn1_bucket_s16_dev.c](../vpu/knn1_bucket_s16_dev.c); the sizes are set in
[../vpu/knn1_bucket_s16_params.h](../vpu/knn1_bucket_s16_params.h). For what the CPU puts in DRAM
before the run, see [bucket_prep.md](bucket_prep.md).

Numbers below are for the default build, **128 ref points per bucket** (`K1_BUCKET=128`), and the
example frame kitti_a (122,919 ref points = 961 buckets in 61 groups of 16; 122,987 query points =
7,687 teams). Addresses are from the linked program (`build/.../knn1_bucket_s16_dev__gen3.elf`).

Each of the two VPUs runs its own copy of the program with its own VMEM, for its own half of
the teams.

## VMEM on Thor

From the Thor PVA VPU Programmer's Guide, §5 Vector Memory
(`~/pva-work/references/sdk/guides/thor/05-vector-memory-vmem.md`, PDF pp. 26–30):

- **Four superbanks** A, B, C, D, 128 KB each (A at 0x00000, B at 0x40000, C at 0x80000, D at
  0xC0000).
- **Same-cycle access:** several reads from the same superbank in one cycle are served one after
  the other. Reads from different superbanks run in parallel.
- **Stack:** in superbank A by default (§5.3).

## Data transfers

Each transfer has a *handler* (`…Hdl`) in superbank A: a small control block the SDK uses to run
it. All three are GatherScatter dataflows (GSDF): the VPU picks the DRAM row (and for buckets the
VMEM place) at run time. The PVA's DMA has a limited number of transfer triggers per program, so
the box table and the statistics ride along on the team and result transfers instead of having
their own.

| Handler | Direction | What and when |
| --- | --- | --- |
| `teamHdl` | DRAM → `teamBuf` | Start-up: the rows of the box table. Then one chunk of 32 teams at a time, into the half of `teamBuf` that is not being worked on (the next chunk arrives while this one is processed). |
| `bucketHdl` | DRAM → cache slot in B, C, D | One bucket per trigger: its x, y and z rows land at the same position in planes B, C and D (3 tiles at a fixed VMEM stride). Only on a cache miss or a prefetch: 496 + 517 copies for the two VPUs on kitti_a, for 22,528 scanned buckets. |
| `resultHdl` | `resultBuf` → DRAM | One chunk of answers at a time, from alternating halves of `resultBuf`. At the end, the statistics. |

## Superbanks B, C, D: the bucket cache (124 KB each)

Each of B, C, D holds exactly one array, `planeB`, `planeC`, `planeD` (`K1_PLANE_ELEMS` int16 =
126,976 bytes each):

```
planeB:  [ x of slot 0 | x of slot 1 | ... | x of slot 479 ] [ box min x (1024) ] [ box max y (1024) ]
planeC:  [ y of slot 0 | y of slot 1 | ... | y of slot 479 ] [ box max x (1024) ] [ box min z (1024) ]
planeD:  [ z of slot 0 | z of slot 1 | ... | z of slot 479 ] [ box min y (1024) ] [ box max z (1024) ]
            128 int16 per slot, K1_CACHE_SLOTS = 480 slots
```

- **The cache** (`cacheX/Y/Z` = the start of each plane) holds 480 buckets, half of a full frame.
  Neighbouring teams need mostly the same buckets, so a bucket copied in once is reused by many
  teams; each VPU only needs the buckets of its half of the scene.
  - `slotOfBucket[b]` (A): the slot holding bucket b, or −1. `bucketOfSlot[s]` (A): the reverse.
  - Slots are reused oldest first (`clockHand`). The slot being filled by the DMA (`pendingSlot`)
    is waited for before it is read, and the bucket being scanned is never replaced by a prefetch.
- **Why three planes:** the scan loads x, y and z of one ref point in the same cycle. From one
  superbank those loads would be served one after the other; from B, C and D they run in
  parallel. Measured: 11.2 → 9.1 cycles per ref point when the planes were split.
- **Why one array per superbank:** so the three planes are exactly 256 KB apart (checked at
  start-up), and one DMA trigger can fill a slot in all three with a constant stride.
- **The box table copy** (six rows of `K1_MAX_BUCKETS` int16, min/max x/y/z of every bucket) sits
  after the cache slots, two rows per superbank, arranged so that the six loads of Step 6's
  expansion come from three different superbanks.
  - Written once at start-up (from the box rows of the team surface, through `teamBuf`).
  - Used by Step 6 (bucket keys of an expanded group) and Step 8 (the skip test).

## Superbank A: everything else (about 30 KB)

| Name | Type × count | Bytes | What it is |
| --- | --- | --- | --- |
| `teamBuf` | int16 × 2 × 1536 | 6,144 | Two chunks of 32 teams, each team x[16] \| y[16] \| z[16]. One is processed while the DMA fills the other. Also used once at start-up to bring in the box table. |
| `resultBuf` | int32 × 2 × 1024 | 8,192 | Two chunks of answers, per team d2[16] \| ref position[16] in slot order (see "Lane order"). At the end, the statistics. |
| `teamHdl`, `resultHdl`, `bucketHdl` | handlers | 7,072 | DMA control blocks. |
| `nBuckets`, `nBucketGroups`, `teamBegin`, `teamEnd`, `vpuIndex` | int32 | 20 | Run-time sizes, written by the host before every run: number of real buckets, of groups, this VPU's team range, and which statistics row to write. |
| `rankKeys` | int32 × `K1_MAX_BUCKETS` | 4,096 | Step 6 bucket keys of the current team, `(bound, low bits cleared) | bucket`, slot order within each group of 16. Only the groups expanded for this team are valid. A visited bucket's key is set to "infinity". |
| `groupKey` | int32 × `K1_MAX_GROUPS` | 256 | Per group of 16 buckets: its box bound key until the group is expanded, then the smallest key of its buckets. The smallest `groupKey` is the next thing to look at. |
| `expandedBy` | int16 × `K1_MAX_GROUPS` | 128 | The team that last expanded each group (−1: none), so nothing has to be reset between teams. |
| `groupLoX` … `groupHiZ` | int16 × `K1_MAX_GROUPS` × 6 | 768 | The box of every group (min/max over its 16 bucket boxes), made at start-up. Used to rank the groups (Step 6). |
| `groupLaneIdsLo/Hi` | int32 × 8 × 2 | 64 | 0, 2, …, 14 and 1, 3, …, 15: which of 16 buckets ends up in which half of a `vsumsq` result. |
| `slotOfBucket` | int16 × `K1_MAX_BUCKETS` | 2,048 | Cache map: bucket → slot or −1. |
| `bucketOfSlot` | int16 × `K1_CACHE_SLOTS` | 960 | Cache map: slot → bucket. |
| `laneIds`, `tmpA` | | 96 | Start-up lane-order self-check only. |
| `diagCycles` | uint32 × 16 | 64 | Diagnostic builds only (`-DK1_DIAG_TSC`): cycles per phase. |
| stack | | | Placed in A by the SDK. |

The team's 16 query points, its running best keys and best buckets live in vector registers
for the whole team, not in VMEM.

## Statistics (row `K1_MAX_CHUNKS + vpuIndex` of the result surface)

| Slot | Content |
| --- | --- |
| 0 | Buckets scanned (from the cache or DRAM) |
| 1 | Bucket picks, counting the pick that triggered the stop |
| 2 | Teams processed |
| 3 | Lane-order self-check and plane layout check passed (1) or not (0) |
| 4–14 | Diagnostic builds only: VPU cycles per phase (`K1_DIAG_*` in the params header) |
| 15 | Buckets copied from DRAM into the cache |

`results()` refuses to return anything if slot 3 is not 1 for every VPU. `stats()` adds the
counters of the VPUs.

## Lane order

`vsumsq` turns 16 input lanes into a result with the even lanes (0, 2, …, 14) in its low half and
the odd lanes (1, 3, …, 15) in its high half (Thor guide §6.2.3.6, PDF p. 51). Keys and answers are
stored in this "slot" order: slot s holds lane `K1_LANE_OF_SLOT(s)`. The host maps the answers back
in `results()`. The program checks this documented order at start-up and refuses to produce results
if the hardware disagrees.

## Other bucket sizes

`K1_BUCKET` = 256 or 512 changes the sizes above: 256 → 240 cache slots, 512 groups max, 2 KB of
`rankKeys`; 512 → 120 slots. 128 was fastest on the KITTI pairs (see
`results/pva_optimization_20261114/`).
