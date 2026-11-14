# Build-Test report: knn1_bucket_s16, run2 (2026-11-14T06:07:50Z)

**Verdict: PASSED**. Failed stage: NONE.

Scope: correctness-and-performance. EMU policy: required. Generation GEN3, PVA SDK 2.10.1, NVVPU, non-safety.
Source: optimized version (128-point buckets, both VPUs, `prepare()` on 1 CPU thread by default).
Manifest before: `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/source-manifest-before.sha256`; after: `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/source-manifest.sha256`.

PRELIMINARY performance: loki in MAXN, clocks not pinned (`clock_mode=MAXN-dvfs`), shared machine.

## Stages

| Stage | Status | Exit | Reason |
|---|---|---|---|
| INPUT | PASSED | 0 | Doctor READY, reference lock d1315d2a… matches, handoff VALID (17 req, 31 correctness, 2 perf) |
| EMU_BUILD | PASSED | 0 | PVA_BUILD_MODE=EMU GEN3 build |
| EMU_TEST | PASSED | 0 | 31/31 cases PASS on EMU |
| L4T_BUILD | PASSED | 0 | Release, PVA_BUILD_MODE=L4T, GEN3, NVVPU, PVA_DISASSEMBLE=ON, K1_BUCKET=128 |
| LISTING | PASSED | 0 | scan loop vectorized: VLDH_S agen broadcast loads, VSumSqHHW, VMinLTW, VMuxW, software-pipelined 4 points per iteration, no spills |
| TARGET_PROBE | PASSED | 0 | aarch64, 6.8.12-tegra, L4T R38.4, pva-sdk-2.10-l4t 2.10.1, user in video group |
| DEPLOY | PASSED | 0 | install tree tar to /tmp/<user>-pva-knn1-bt-run2, sha256 verified, cupva 2.10 resolved |
| HARDWARE_TEST | PASSED | 0 | 31/31 cases PASS on loki; sample PASS |
| CLOCK | PASSED | 0 | clock_mode=MAXN-dvfs: MAXN before and after (read-only); not pinned per CLAUDE.md (CONFLICT-CLOCKS) |
| PROFILE | PASSED | 0 | 100 warm-ups + 1000 samples per case, Profiling Stream, correctness gate PASS on every case |
| COMPARE | NOT_RUN |  | no performance target or acceptance rule (acceptance not_set) |
| SOURCE_FRESHNESS | PASSED | 0 | build inputs unchanged; only change: pva-operator/knn1_bucket_s16/capi/README.md (documentation, edited after the run to update its example output; not compiled or installed by the test build) |

## Correctness

Reference: scipy cKDTree exact 1-NN on the quantized inputs (testdata/knn1_bucket). Comparison: dist2 bit-exact, idx in range with recomputed d2 equal (ties allowed).

| Case | EMU | Hardware | Hardware detail |
|---|---|---|---|
| CORR-KITTI-A | PASS | PASS | nq=122987 nr=122919 teams=7687 buckets=961 fetched=22449 fetched_per_team=2.92039 sum_dist2=51148720 |
| CORR-KITTI-B | PASS | PASS | nq=122563 nr=122753 teams=7661 buckets=960 fetched=25158 fetched_per_team=3.28391 sum_dist2=302125029 |
| CORR-KITTI-C | PASS | PASS | nq=123714 nr=123867 teams=7733 buckets=968 fetched=22906 fetched_per_team=2.96211 sum_dist2=267022911 |
| CORR-KITTI-D | PASS | PASS | nq=123298 nr=123431 teams=7707 buckets=965 fetched=21904 fetched_per_team=2.84209 sum_dist2=263371403 |
| CORR-KITTI-A-SWAP | PASS | PASS | nq=122919 nr=122987 teams=7683 buckets=961 fetched=22820 fetched_per_team=2.97019 sum_dist2=51025322 |
| CORR-KITTI-16K | PASS | PASS | nq=16384 nr=16384 teams=1024 buckets=128 fetched=2302 fetched_per_team=2.24805 sum_dist2=32839805 |
| CORR-KITTI-UNEQUAL | PASS | PASS | nq=4096 nr=122919 teams=256 buckets=961 fetched=1332 fetched_per_team=5.20312 sum_dist2=4676428 |
| CORR-KITTI-SELF | PASS | PASS | nq=16384 nr=16384 teams=1024 buckets=128 fetched=1821 fetched_per_team=1.77832 sum_dist2=0 |
| CORR-MAX-131072 | PASS | PASS | nq=131072 nr=131072 teams=8192 buckets=1024 fetched=38448 fetched_per_team=4.69336 sum_dist2=10335670310 |
| CORR-SMALL-1X1 | PASS | PASS | nq=1 nr=1 teams=1 buckets=1 fetched=1 fetched_per_team=1 sum_dist2=9154422 |
| CORR-SMALL-1X512 | PASS | PASS | nq=1 nr=512 teams=1 buckets=4 fetched=2 fetched_per_team=2 sum_dist2=204627 |
| CORR-SMALL-16X1 | PASS | PASS | nq=16 nr=1 teams=1 buckets=1 fetched=1 fetched_per_team=1 sum_dist2=316980738 |
| CORR-SMALL-15X511 | PASS | PASS | nq=15 nr=511 teams=1 buckets=4 fetched=4 fetched_per_team=4 sum_dist2=3354203 |
| CORR-SMALL-16X512 | PASS | PASS | nq=16 nr=512 teams=1 buckets=4 fetched=4 fetched_per_team=4 sum_dist2=3271854 |
| CORR-SMALL-17X513 | PASS | PASS | nq=17 nr=513 teams=2 buckets=5 fetched=5 fetched_per_team=2.5 sum_dist2=3409418 |
| CORR-SMALL-513X17 | PASS | PASS | nq=513 nr=17 teams=33 buckets=1 fetched=33 fetched_per_team=1 sum_dist2=957540993 |
| CORR-SMALL-1000X3000 | PASS | PASS | nq=1000 nr=3000 teams=63 buckets=24 fetched=263 fetched_per_team=4.1746 sum_dist2=65423830 |
| CORR-EXTREMES | PASS | PASS | nq=32 nr=5 teams=2 buckets=1 fetched=2 fetched_per_team=1 sum_dist2=27648000000 |
| CORR-DUPLICATES-GRID | PASS | PASS | nq=4000 nr=6000 teams=250 buckets=47 fetched=553 fetched_per_team=2.212 sum_dist2=0 |
| CORR-ALL-REFS-IDENTICAL | PASS | PASS | nq=1000 nr=2000 teams=63 buckets=16 fetched=63 fetched_per_team=1 sum_dist2=145822220189 |
| CORR-DEGENERATE-LINE | PASS | PASS | nq=1000 nr=3000 teams=63 buckets=24 fetched=89 fetched_per_team=1.4127 sum_dist2=30705 |
| CORR-DEGENERATE-PLANE | PASS | PASS | nq=2000 nr=5000 teams=125 buckets=40 fetched=282 fetched_per_team=2.256 sum_dist2=74707564 |
| CORR-FAR-CLUSTERS | PASS | PASS | nq=1200 nr=6000 teams=75 buckets=47 fetched=427 fetched_per_team=5.69333 sum_dist2=55061939485 |
| CORR-REPEATED-SUBMISSION | PASS | PASS | 100 submissions identical and correct |
| CORR-REPREPARE-SIZES | PASS | PASS | 5-step sequence correct |
| CORR-ASYNC-FENCE | PASS | PASS | no sentinel left, case correct |
| CORR-SUMDIST2-INT64 | PASS | PASS | 3 sums > 2^31 exact |
| CORR-STATS-VS-MODEL | PASS | PASS | kitti_a:fetched=22449,teams=7687 kitti_b:fetched=25158,teams=7661 kitti_c:fetched=22906,teams=7733 kitti_d:fetched=21904,teams=7707  |
| ERR-ORDER | PASS | PASS | all out-of-order calls rejected |
| ERR-ARGS | PASS | PASS | bad arguments rejected, later valid run correct |
| CORR-NO-ALIAS | PASS | PASS | disjoint buffers, inputs unchanged |

EMU: 31/31 PASS. Hardware: 31/31 PASS. Sample: PASS.

## Performance (100 warm-ups, 1000 samples per case, ms)

| Case | Metric | min | p50 | p99 | max | mean | valid samples |
|---|---|---|---|---|---|---|---|
| PERF-KITTI-A | PVA task time (profiler TOTAL_TIME) | 13.825 | 13.827 | 13.833 | 13.837 | 13.827 | 1000 |
| PERF-KITTI-A | prepare() host time, 1 thread | 15.002 | 15.055 | 15.165 | 15.270 | 15.062 | 1000 |
| PERF-KITTI-A | results() host time | 0.310 | 0.322 | 0.343 | 0.366 | 0.323 | 1000 |
| PERF-KITTI-A | end to end (prepare + submit..fence + results) | 29.184 | 29.239 | 29.346 | 29.442 | 29.244 | 1000 |
| PERF-KITTI-B | PVA task time (profiler TOTAL_TIME) | 15.260 | 15.261 | 15.268 | 15.271 | 15.262 | 1000 |
| PERF-KITTI-B | prepare() host time, 1 thread | 14.910 | 14.953 | 15.147 | 15.544 | 14.970 | 1000 |
| PERF-KITTI-B | results() host time | 0.318 | 0.332 | 0.348 | 0.359 | 0.332 | 1000 |
| PERF-KITTI-B | end to end (prepare + submit..fence + results) | 30.532 | 30.580 | 30.785 | 31.185 | 30.596 | 1000 |

Correctness gate on the last timed call of every case: PASS (bad = 0). Work counters: kitti_a 22,449 buckets scanned and 1,011 copied from DRAM; kitti_b 25,158 and 1,030.

Supplementary (not a spec performance case): with `cpu_threads = 8`, `prepare()` p50 is 3.31 ms (kitti_a) and 3.24 ms (kitti_b), and end to end 17.49 / 18.82 ms. In those two runs only 260 of the 1000 profiler TOTAL_TIME entries were valid, so use the host PVA wall time there (13.85 / 15.28 ms).

## Machine state

- `nvpmodel -q`: MAXN before and after. `jetson_clocks --show` needs root and was not run (the error message is saved).
- tegrastats: GPU power flat at its idle value (4.7 W). Before the runs one CPU core was ~95% busy from kernel threads that are not ours (`kworker … ext4-rsv-conversion`, `kworker … devfreq_wq`), as in every earlier run. By the measurement rules this run is not clean. A rerun would see the same threads.
- Remote directory removed after the run (CLAUDE.md rule; the skill would otherwise keep it). No processes left on loki.

## Listing

`/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/build-l4t/knn1_bucket_s16_dev__gen3.lst` (sha256 9af3e950cbfd7f79b0700d14a1b685304f945d5f0e7ca48a415ffeec5f0398c0); notes and the scan-loop excerpt in `logs/listing.txt`.

## Evidence

| Artifact | Path | sha256 |
|---|---|---|
| source manifest before | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/source-manifest-before.sha256` | `737d032297b45c80…` |
| source manifest after | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/source-manifest.sha256` | `39ef090f7406e8df…` |
| commands log | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/commands.log` | `041620074d03ead3…` |
| compiler listing | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/build-l4t/knn1_bucket_s16_dev__gen3.lst` | `9af3e950cbfd7f79…` |
| listing notes | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/listing.txt` | `ed81ebbdfcda377f…` |
| EMU test log | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/emu-test.log` | `7afe7c5de1d0c275…` |
| hardware test log | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/hw-test.log` | `7afe7c5de1d0c275…` |
| sample log | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/hw-sample.log` | `c8ec979fb314c23e…` |
| deploy archive | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/deploy.tar.gz` | `f4026f7feb540ace…` |
| deploy manifest | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/deploy-manifest.sha256` | `b9732174dc3a7b85…` |
| perf log | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/perf.log` | `abf1670aa5f17a89…` |
| perf kitti_a | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/perf/perf_kitti_a.csv` | `319271fb3fde5b44…` |
| perf kitti_b | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/perf/perf_kitti_b.csv` | `6f1a8f97f98e4bab…` |
| perf kitti_a threads=8 (supplementary) | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/perf/perf_kitti_a_threads8.csv` | `3284404c3c456a28…` |
| perf kitti_b threads=8 (supplementary) | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/perf/perf_kitti_b_threads8.csv` | `e3973d75ac3f640d…` |
| tegrastats | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/perf/perf.tegrastats.txt` | `bd7d4cec18dbc88a…` |
| clocks before | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/clocks-before.txt` | `717d3f4c834496ea…` |
| clocks after | `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/build-test/run2/logs/clocks-after.txt` | `9475b9cb2d2c2f5b…` |

## Source changes after the run (2026-11-14)

After the run finished, comments were trimmed in the operator sources, `hermon` was replaced by
`<user>` in raw logs, and a build-host `Makefile` was added. No code changed. Checked by rebuilding
L4T from the edited source: `knn1_bucket_s16_dev__merged.bin` (VPU program) is byte-identical to this
run's, and the `.text`/`.rodata` sections of all host programs and `libknn1_distance.so` are
identical. The results above therefore apply to the current code. `source-manifest.sha256` is the
state at the end of the run, not the current text.
