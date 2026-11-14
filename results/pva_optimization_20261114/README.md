# Optimizing knn1_bucket_s16 (2026-11-13/14)

PRELIMINARY: loki, MAXN, clocks not pinned (`clock_mode=MAXN-dvfs`), shared machine.

Formal numbers come from two `pva-build-test` runs, each with 100 warm-ups and 1000 samples:
- **run1:** the first version.
- **run2:** the optimized version.

Evidence for both is in `pva-operator/knn1_bucket_s16/build-test/run1/` and `.../run2/`. Both runs are
exact on every sample (correctness gate, 0 wrong results), and run2 passes 31/31 hardware test cases.

## Result

Times in ms, p50 / p99. PVA = profiler task time, including all DMA.

| Case | Part | run1 (first version) | run2 (optimized) | Speed-up |
|---|---|---|---|---|
| kitti_a | `prepare()`, 1 CPU thread | 19.17 / 19.32 | 15.06 / 15.17 | 1.27× |
| | PVA task | 105.01 / 105.13 | **13.83 / 13.83** | **7.6×** |
| | `results()` | 0.34 / 0.35 | 0.32 / 0.34 | |
| | **end to end** | 124.54 / 124.82 | **29.24 / 29.35** | **4.3×** |
| kitti_b | `prepare()`, 1 CPU thread | 19.33 / 19.56 | 14.95 / 15.15 | 1.29× |
| | PVA task | 110.09 / 110.17 | **15.26 / 15.27** | **7.2×** |
| | `results()` | 0.35 / 0.36 | 0.33 / 0.35 | |
| | **end to end** | 129.80 / 130.04 | **30.58 / 30.78** | **4.2×** |

Settings: run1 used 512-point buckets and 1 VPU; run2 uses 128-point buckets, 2 VPUs and 1 CPU thread.

Supplementary run2 measurement with `init(2, 8)`, i.e. `prepare()` on 8 CPU threads:

| Case | `prepare()` p50 / p99 | end to end p50 / p99 |
|---|---|---|
| kitti_a | 3.31 / 5.34 | 17.49 / 19.52 |
| kitti_b | 3.24 / 5.17 | 18.82 / 20.75 |

In these two runs only 260 of the 1000 profiler task-time entries were valid (the rest ≤ 0, > 10 s,
or an impossible 4.1 µs). The host-measured PVA wall time is steady (13.85 / 15.28 ms), so use
that for these runs. More threads finish `prepare()` sooner, but they occupy more CPU cores while it
runs; the total CPU work stays about the same.

## What changed, step by step

These are quick diagnostic runs (20–30 samples, kitti_a), not formal measurements. Each line adds one
change to the one before.

| Change | PVA time | Scan cycles per ref point |
|---|---|---|
| First version (512-point buckets, 1 VPU) | 105–106 ms | 10.0 |
| Bucket cache in VMEM, prefetch of the next bucket, Step 6 ranked by team box, 256-point buckets | 56 ms | 12 |
| Scan loads through address generators | 52.7 ms | 11.2 |
| x, y, z of the cache in three superbanks (B, C, D) so the three loads run in parallel; group minima for picking | 41.3 ms | 9.1 |
| Scan with VMinLT + VMux (minimum and "closer" flag in one step) instead of key shifts | 37.9 ms | 8.1 |
| Scan loop unrolled 4× (compiler interleaves points, hides multiply latency) | 28.6 ms | 5.4 |
| Lazy ranking: rank groups of 16 buckets first, expand a group only when it comes up | 28.5 ms | 5.4 |
| 128-point buckets | 27.6 ms | 5.5 |
| **Both VPUs**, each one contiguous half of the teams | **13.9 ms** | |

The CPU side, `prepare()`, also changed: the median split now counts instead of using
`std::nth_element` and has no data-dependent branches (21.4 → 15.0 ms on loki), and it can optionally
use several threads.

Where the cycles go now (1 VPU, kitti_a, bucket 128, cycle-counter build):
- scan: 16.0 M cycles (45%);
- pick, including group expansion: 9.2 M;
- fetch (cache lookup, DMA set-up, prefetch decision): 4.5 M;
- rank: 1.6 M;
- skip test: 0.7 M.

The scan is close to its limit: 8 vector operations per ref point on 2 vector slots is 4 cycles,
and it runs at 5.4. The remaining overhead is mostly scalar control code per visited bucket.

## Tried and not kept

| Idea | Result |
|---|---|
| dz² with VSqSum from a stored −z (8 instead of 9 operations) | 29.3 ms against 28.6 ms: the compiler schedules it worse |
| Hand-pipelined scan loop | 45.0 ms: extra register moves take the vector slots |
| Manual even/odd scan | 32.8 ms |
| Scan unrolled 8× | 35.3 ms |
| Ranking all buckets of every team up front (instead of lazily) | 35.7 ms (bucket 256) |
| Expanding only groups the team box touches, up front | 30.8 ms (bucket 256) |
| Alternating chunks between the two VPUs | Balances the VPUs within 1%, but doubles the DRAM bucket copies: kitti_a 13.87 → 14.24 ms, kitti_b 15.33 → 15.23 ms |
| `prepare()`: 64-bit packed points, four sets of counters, branch-free store index | Faster on the x86 build host, no change on loki |

## Answers vs the model

On pair 0 (kitti_a) at bucket 128, the Python model gives the same sum d2 (51,148,720) with 0 mismatches.
It counts 22,622 scanned buckets against the kernel's 22,449. Points with equal coordinates can fall into
different buckets in C++ and Python; the spec allows this.

## Machine state during run2

- `nvpmodel -q`: MAXN before and after (`build-test/run2/logs/clocks-*.txt`).
- `jetson_clocks --show` needs root and was not run (the error message is saved).
- `tegrastats` (`build-test/run2/perf/perf.tegrastats.txt`): GPU power flat at its idle value (4.7 W).
- One CPU core was ~95% busy before the runs from two kernel threads that are not ours
  (`kworker … ext4-rsv-conversion`, `kworker … devfreq_wq`), as in every earlier run. By the measurement
  rules these runs are therefore not clean. A rerun would see the same threads, and the PVA times vary by
  less than 0.1%.

## Not done

- Reuse frame t's query bucketing as frame t+1's ref bucketing. This needs teams to be a subdivision of
  buckets, which changes the method, so the model comes first (UNRES-REUSE in the spec).
- The remaining per-bucket scalar overhead in the VPU program (pick and fetch ≈ 40% of the cycles).
- A pinned-clock session (to be agreed with the user).
