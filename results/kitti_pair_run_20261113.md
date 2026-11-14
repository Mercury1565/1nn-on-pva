# KITTI pair run, 2026-11-13 (functional check, not a performance measurement)

Program: `pva-operator/knn1_bucket_s16/sample/knn1_bucket_s16_kitti.cpp`. Build: L4T, GEN3,
Release, current source (after the pc_ref/pc_queries rename). Run on loki (MAXN-dvfs, clocks
not pinned, shared machine). This is not a formal pva-build-test run.

The program reads the raw velodyne .bin files and quantizes them like `src/knn_data.py`
(`quantize_cm`). Check: every query point compared on the build host against SciPy cKDTree
on the same quantized points, with d2 recomputed in int64.

| Pair (pc_ref -> pc_queries) | n_ref | n_query | Buckets fetched (per team) | sum d2 (cm^2) | sqrt(sum d2) | RMS nearest distance | d2 mismatches | idx not at min distance | idx differs only by tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2011_09_26 drive_0001 89 -> 90 (= test case kitti_a) | 122919 | 122987 | 12905 (1.68) | 51148720 | 71.518 m | 20.39 cm | 0 | 0 | 3878 |
| 2011_10_03 drive_0027 76 -> 77 | 118742 | 118174 | 17082 (2.31) | 103540842 | 101.755 m | 29.60 cm | 0 | 0 | 3992 |

The first pair's sum d2 matches the stored answer of test case kitti_a (51148720).

Host wall-clock times of one call each (orientation only, NOT PVA performance):

| Pair | prepare | submit + wait | results |
| --- | --- | --- | --- |
| kitti_a pair | 20.1 ms | 104.1 ms | 0.5 ms |
| 2011_10_03 pair | 20.0 ms | 126.3 ms | 0.5 ms |
