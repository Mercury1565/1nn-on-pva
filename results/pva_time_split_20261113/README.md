# Where the PVA time goes (2026-11-13)

PRELIMINARY: MAXN, clocks not pinned (`clock_mode=MAXN-dvfs`), shared machine (loki).
This is a diagnostic run, not a formal `pva-build-test` measurement: 98 samples per run (not 1000+), and
it ran on a source with the cycle counters added. Nothing here replaces a formal build-test run.

## How it was measured

Three builds of the same kernel, run on loki in two interleaved rounds, 20 warm-ups and 100 samples each:

| Build | Cycle counters | What they count |
|---|---|---|
| `normal` | none | only the cuPVA execution time (`exec_ns`) |
| `wfe` | `-DK1_DIAG_TSC` | VPU work only: the counter stops while the VPU sleeps waiting for DMA |
| `poll` | `-DK1_DIAG_TSC`, `PVA_PROFILING_MODE=ON` | work plus DMA waiting (the SDK polls instead of sleeping) |

Wait per phase = poll cycles - wfe cycles. Program: `perf/knn1_bucket_s16_diag.cpp`; counters:
`vpu/knn1_bucket_s16_dev.c` (`DIAG_BEGIN`/`DIAG_END`). Each run's sum d2 was checked: 0 wrong runs.

The first two profiler entries of every run are invalid (4.58e10 and ~1e19 ns, right after
`ProfilingContext::reset()`); they are left out, so each run has 98 samples.

## PVA execution time (`exec_ns`, ms, both rounds)

| Case | Build | p50 | p99 |
|---|---|---|---|
| kitti_a (2011_09_26 drive 0001, 89->90) | normal | 105.11 | 105.16 |
| | wfe | 106.03 | 106.08 |
| | poll | 106.21 | 106.32 |
| kitti_b (2011_10_03 drive 0027, 76->77) | normal | 110.09 | 110.17 |
| | wfe | 111.02 | 111.09 |
| | poll | 111.22 | 111.30 |

The counters add 0.9% to the run time. Rounds 1 and 2 agree within 0.01 ms.
Cycle counts are identical from run to run in the wfe build (the work is deterministic).
Derived VPU clock: poll total cycles / exec_ns = 1.215 GHz (both cases).

## Per phase (median cycles, millions; ms at 1.215 GHz)

| Phase | kitti_a work | kitti_a wait | kitti_a ms | share | kitti_b ms | share |
|---|---|---|---|---|---|---|
| scan (512 ref points x 16 lanes per fetched bucket) | 66.37 | 0 | 54.6 | 51% | 60.2 | 54% |
| rank (Step 6, all buckets, every team) | 33.36 | 0 | 27.5 | 26% | 25.1 | 23% |
| fetch (GSDF bucket fetch) | 5.51 | 11.02 | 13.6 | 13% | 15.0 | 14% |
| merge (per-bucket merge + worst-best update) | 5.15 | 0.04 | 4.3 | 4% | 4.7 | 4% |
| pick (smallest key) | 3.73 | 0 | 3.1 | 3% | 3.0 | 3% |
| result, skip test, chunk in/out, start-up | 2.0 | 0.38 | 2.0 | 2% | 2.0 | 2% |
| **total** | **117.58** | **11.46** | **106.2** | | **111.2** | |

Totals: DMA waiting is 11.5 M cycles = 9.4 ms = **8.9%** (kitti_b: 9.3%). The other 91% is VPU work.

## What it says

- The hypothesis behind next step 3 (most time is DMA waiting) is wrong. Overlapping fetches with
  scanning (double buffering) can save at most ~9 ms of 105-110 ms.
- Scan costs 10.0 cycles per ref point (66.37 M / (12905 x 512); kitti_b also 10.0). The compiler listing
  (`build/diag-wfe/.../knn1_bucket_s16_dev.o.lst`, around 0x1264) schedules about 7 cycles per ref point:
  3 scalar loads, 3 subtractions, 2 vsumsq, 1 add, 2 shifts, 1 add of j, 1 dvmin. The other ~3 cycles are
  not explained yet (stalls; possibly the x/y/z reads from superbank C, the open question in the VMEM doc).
- Rank costs ~4300 cycles per team (16 query points x 241 buckets), but a team then fetches only
  1.7 buckets on average (12905 / 7687 teams). Most of the ranking work is spent on buckets that are never visited.
- Fetch has 427 cycles of VPU work per bucket on top of the wait (GSDF set-up and trigger).

## Machine state

`nvpmodel -q`: MAXN before and after (`clocks-before.txt`, `clocks-after.txt`).
`tegrastats.txt`: no GPU load. One CPU core at ~95% throughout, from two kernel threads that are not ours
(`kworker ... ext4-rsv-conversion`, `kworker ... devfreq_wq`, running for 17-18 days, also present before
and after the run, `load-*.txt`). They do not run on the PVA, and the PVA times are stable to 0.05 ms, but by
the measurement rules this run is not clean; a rerun would see the same threads.
