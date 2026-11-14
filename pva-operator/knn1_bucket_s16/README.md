# Knn1BucketS16 (`knn1_bucket_s16`)

Exact 1-nearest-neighbour search between two int16 point clouds on the Jetson Thor PVA (GEN3).
Built for the Mnemosyne PVA feasibility study, where it computes a cloud-to-cloud distance.

The search uses buckets of 128 reference points (CMake option `K1_BUCKET`: 128, 256 or 512), so
most reference points are never compared. The work is split between both VPUs of the PVA. A
Python model, [`src/knn1_bucket_model.py`](../../src/knn1_bucket_model.py), follows the same
steps one to one; read it first if the C code is unclear.

- Spec: `configure/operator-spec.json`
- Tests: `configure/test-matrix.json`

## What it computes

For every query point `q_i` of cloud P1:

```
dist2[i] = min over all ref points r_j of pc_ref (P2) of (qx-rx)^2 + (qy-ry)^2 + (qz-rz)^2     exact, int32, cm^2
idx[i]   = one j that reaches that minimum (any j on a tie)
```

- **Cloud distance:** `sqrt(Knn1BucketS16::sumDist2(dist2, n))`, where the sum is int64.
- **Chamfer distance:** two calls, with P1 and P2 swapped.
- **Exactness:** a bucket is skipped only when its bounding box proves that none of its points can be closer.

## API ([`include/knn1_bucket_s16.hpp`](include/knn1_bucket_s16.hpp), namespace `pva_knn`)

| Call | What it does |
| --- | --- |
| `init(vpus = 2, cpu_threads = 1)` | Once. Allocates the device buffers for the maximum size and creates one PVA program per VPU (`vpus` 1 or 2). `cpu_threads` (1–16): CPU threads `prepare()` may use; same results for every value. |
| `prepare(q, nq, r, nr)` | **Part A, on the CPU.** Builds buckets, boxes and teams, and writes them to the device buffers. `q` and `r` are host arrays, planar int16 `[3][n]`, with 1 ≤ n ≤ 131072. |
| `submit(stream, fence)` | **Part B, on the PVA.** Asynchronous. With 2 VPUs both programs run side by side. |
| `results(dist2, idx)` | After the fence. Writes `dist2[nq]` and `idx[nq]` in the caller's original query order. |
| `stats(&s)` | After the fence. Reports `n_query`, `n_ref`, `n_buckets`, `n_teams`, `buckets_fetched` (scanned) and `dma_fetches` (copied from DRAM into the PVA's bucket cache). |
| `vpus()`, `lastStatus()` | Number of VPU programs, and the error status of the last submission. |
| `sumDist2(d, n)` | int64 sum of squared distances. |

- Every `int32_t` call returns 0 on success and nonzero on error. None of them throws.
- On an error, the caller's buffers are not touched.
- Only one submission per instance may be outstanding at a time.
- You can call `prepare()` again, with any sizes, between submissions.

**Precondition:** every coordinate is in [-12000, 12000] cm (±120 m, quantized by
`src/knn_data.py`). This keeps every squared distance at or below 1 728 000 000, which is exact in int32.
The operator does not check it.

## How it works

The steps match `src/knn1_bucket_model.py`.

### Part A, CPU ([`src/bucket_prep.cpp`](src/bucket_prep.cpp), explained in [`docs/bucket_prep.md`](docs/bucket_prep.md))

1. **Order P2 into buckets.** Split it recursively at the median of its widest axis, so that every run of 128 points is one bucket. The median is found by counting, without data-dependent branches.
2. **Store P2 bucket by bucket.** Each bucket is one row, `x[128] | y[128] | z[128]`. A partial last bucket is filled with copies of its own last point, which keeps the answer exact.
3. **Build the box table.** Store the min and max x, y, z of every bucket. The table is padded to a multiple of 16 buckets.
4. **Order P1 into teams.** Split it the same way into teams of 16 query points. A partial last team is filled with copies of a query point.

### Part B, PVA ([`vpu/knn1_bucket_s16_dev.c`](vpu/knn1_bucket_s16_dev.c), memory in [`docs/knn1_bucket_s16_vmem.md`](docs/knn1_bucket_s16_vmem.md))

Each VPU handles one contiguous half of the teams, one team at a time; teams arrive in chunks of 32.

5. **Load the team.** Put its 16 query points into the 16 vector lanes.
6. **Rank the buckets** by the smallest squared distance between the team's bounding box and each bucket's box, as a key `(bound, low bits cleared) | bucket`. Done lazily: first the groups of 16 buckets are ranked by the group's box, and a group's 16 bucket keys are worked out only when the group comes up as the smallest. The visiting order is exactly the model's.
7. **Reset the per-lane best** to infinity.
8. **Visit the buckets nearest first.**
   - **Stop** if the bound is not better than the worst lane's best.
   - **Skip** the bucket if no lane can beat its own best (per-lane distance to the box).
   - **Otherwise get and scan it.** Most buckets are already in the VPU's bucket cache (480 buckets in VMEM); a missing one is copied in by GSDF, and the next candidate is copied in while the current one is scanned. The scan takes 8 vector operations per ref point for all 16 query points (3 loads from three superbanks, 3 subtractions, 2 `vsumsq` + 1 add, then `vminLT`/`vmux` for the per-lane minimum and its position), about 5.4 cycles per ref point.
9. **Write the result.** Each team writes 16 distances and 16 ref positions (`bucket * 128 + position`), in `vsumsq` slot order.

### Part C, CPU ([`src/knn1_bucket_s16.cpp`](src/knn1_bucket_s16.cpp))

`results()` maps slots back to lanes, teams back to query indices and ref positions back to ref indices.

## Dataflow and VMEM

Details in [`docs/knn1_bucket_s16_vmem.md`](docs/knn1_bucket_s16_vmem.md).

| Data | DRAM layout | Moved by | VMEM |
| --- | --- | --- | --- |
| Teams, then box table | int16 [256 chunks + 4 box rows][1536] | GSDF, one row per transfer; next chunk arrives while one is processed | A, 2 × 3 KiB |
| Buckets | int16 [1024][384] | GSDF, 3 tiles (x, y, z) per transfer into one cache slot | B, C, D: 480 slots of 128 points, plus the box table copy |
| Results, then statistics | int32 [256 chunks + 2][1024] | GSDF (write), one row per transfer, double buffered | A, 2 × 4 KiB |

The VPU reads its run-time sizes (`nBuckets`, `nBucketGroups`, `teamBegin`, `teamEnd`, `vpuIndex`) from VMEM scalars that `prepare()` sets.

**Lane order.** `vsumsq` returns lanes in the order 0, 2, …, 14, 1, 3, …, 15 (Thor guide 6.2.3.6, PDF p.51). The program
verifies this order at start-up. If it doesn't hold, the program fails, `results()` returns an error, and `stats()` reports it.

**Scalar regions**, all outside the scan loop: picking the next key (a vector minimum over the group keys, then scalar
decisions), the cache lookup and DMA set-up, the merge of a scanned bucket into the running best, and the per-team result write.

## Speed (preliminary)

kitti_a, loki, MAXN with clocks not pinned, shared machine; formal numbers in
`build-test/run2/` and `results/pva_optimization_20261114/`.

| Part | First version | Now |
| --- | --- | --- |
| PVA task (2 VPUs) | 105 ms (1 VPU) | 13.9 ms |
| `prepare()` (1 thread) | 19–21 ms | 15.0 ms (3.3 ms with 8 threads) |
| End to end | ~125 ms | ~29 ms |

## Build

Needs PVA SDK 2.10.1 with NVVPU. On the build host, put the system Python first so that pyelftools is found:

```bash
export PATH=/usr/bin:$PATH
cmake -S . -B build-emu -DPVA_BUILD_MODE=EMU -DPVA_DEFAULT_GEN=GEN3 -DPVA_ONLY_BUILD_GENS=GEN3
cmake --build build-emu -j
PVA_GEN=GEN3 build-emu/knn1_bucket_s16_test ../../testdata/knn1_bucket    # EMU: correctness only

cmake -S . -B build-l4t -DPVA_BUILD_MODE=L4T -DPVA_DEFAULT_GEN=GEN3 -DPVA_ONLY_BUILD_GENS=GEN3
cmake --build build-l4t -j
```

**Targets:**

| Target | What it is |
| --- | --- |
| `knn1_bucket_s16` | Host library, including the embedded VPU program |
| `knn1_bucket_s16_test` | Full test matrix |
| `knn1_bucket_s16_perf` | Timing harness |
| `knn1_bucket_s16_sample` | Minimal lifecycle |
| `knn1_bucket_s16_ref` | Brute-force CPU reference |

`cmake --install` installs the three binaries and the test vectors.

## Run on the target

```bash
knn1_bucket_s16_test   <testdata>/knn1_bucket                    # RESULT lines + SUMMARY
knn1_bucket_s16_perf   <testdata>/knn1_bucket perf.csv --case kitti_a --warmup 100 --samples 1000 [--vpus 2] [--threads 1]
knn1_bucket_s16_sample
```

`knn1_bucket_s16_perf` reports p50, p99, min and max for each part of a call:
- `prepare` (CPU)
- `pva_wall` (submit until the fence returns)
- `pva_total` (profiler task time, including DMA)
- `vpu_exec_sum` (VPU execution time summed over both VPU programs)
- `overhead`
- `results` (CPU)
- `e2e` (end to end)

It also reports buckets scanned and buckets copied from DRAM.

Timings from the emulator or the build host are not PVA performance.
