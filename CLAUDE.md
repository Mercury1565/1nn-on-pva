# Exact 1-NN point-cloud distance on the Jetson Thor PVA

## Goal

Build, verify and characterize one PVA kernel: exact 1-nearest-neighbour search between two
LiDAR point clouds, `knn1_bucket_s16` in `pva-operator/knn1_bucket_s16/`. For every point of
`pc_queries` (frame t), find the nearest point of `pc_ref` (frame t-1) and its squared distance;
the cloud distance is sqrt(sum of d2). This is the project's only focus. Earlier, broader work
(brute-force K=8 kernel, GPU/CPU baselines) is in `~/pva-knn-archive`; do not bring it back
without asking.

Report what the measurements show. A slow or negative result is a valid result; never shade
numbers toward a positive answer.

## Method (fixed decisions)

- Names: `pc_queries` / `pc_ref` for the clouds, "query point" / "ref point" for single points.
- Inputs: int16 centimetres after clamping to +/-120 m (so d2 <= 3 * 24000^2 fits in int32),
  planar [3][n]. 1 <= n <= 131072 per cloud.
- Part A, CPU (`src/bucket_prep.cpp`, explained in `docs/bucket_prep.md`): median-split `pc_ref`
  into buckets of 128 (CMake `K1_BUCKET`) with bounding boxes; median-split `pc_queries` into teams of 16; padding
  with copies keeps results exact.
- Part B, PVA (`vpu/knn1_bucket_s16_dev.c`, VMEM layout in `docs/knn1_bucket_s16_vmem.md`): per
  team, rank buckets by box lower bound, visit nearest-first with the exact stop and skip rules,
  fetch buckets with GSDF and scan them.
- `src/knn1_bucket_model.py` is the step-by-step model and answer key. Change the method there
  first, then in the kernel.
- The user wants plain explanations without unnecessary jargon, clear code with brief comments,
  and documentation kept next to the code it explains.

## Machines and environment

- **This machine** (build host, x86-64, Ubuntu 22.04, native install, no Docker): PVA SDK 2.10.1
  at `/opt/nvidia/pva-sdk-2.10`, NVVPU compiler, EMU, aarch64 cross compilers. No ASIP.
  The user has sudo here, but ask before running anything with sudo.
  Run PVA cmake with `PATH=/usr/bin:$PATH` (conda's python3 lacks pyelftools).
- **loki** (`ssh loki`, own user account): Jetson AGX Thor, L4T R38.4 (JetPack 7.1), PVA GEN3,
  power mode MAXN, cuPVA runtimes 2.10.1 and 2.9.0 (our builds use 2.10). Key-based SSH works in
  batch mode. The account is in the `video` group, so test runs need no sudo. loki is SHARED with
  other people: never run sudo there, never change its clocks or power mode, and never leave
  processes running after a run. Deploy test builds under `/tmp/<user>-pva-knn1-*` (`make deploy` / `make test-loki` in the operator folder do this) and remove
  them when done.
- **deadshot** (`ssh deadshot`): holds the full KITTI raw dataset. Read only; see Dataset.
- **Disk is tight on loki** (~50 GB free). Never copy the full KITTI dataset anywhere. Clean up
  build directories and temporary files you create.

## PVA workflow

- Skills are in `~/.claude/skills/pva-*`. Work directory: `~/pva-work` (reference corpus in
  `~/pva-work/references/sdk`). `pva-doctor` must report READY before Spec/OpGen/Build-Test.
- Skills in order: `pva-spec` -> `pva-opgen` -> `pva-build-test` -> `pva-debug` on failures.
- The spec handoff (`configure/opgen-inputs.json`) records eight project files by hash and path
  (model, bucket comparison, vector generator, loader, `results/frames.txt`, two result CSVs,
  `testdata/knn1_bucket/manifest.json`). Do not move them. After changing any of them, or the
  spec, regenerate with `configure/build_spec.py`, re-finalize, and note it in
  `reports/configure-report.md`.
- Any source change needs a fresh `pva-build-test` run (`build-test/runN/`) before its results
  count as evidence.
- Always target GEN3 only: configure with `-DPVA_DEFAULT_GEN=GEN3 -DPVA_ONLY_BUILD_GENS=GEN3`,
  and run EMU binaries with `PVA_GEN=GEN3`.
- EMU is for correctness only. Never report EMU or host timings as PVA performance.
- Look up VPU capabilities (data types, vector widths, VMEM superbanks, DMA patterns) in the
  reference corpus instead of assuming. Cite the reference file when a design choice depends on it.

## Dataset: KITTI raw

- The full dataset lives on deadshot at `/z/home/kidus/drive_data/kitti_full`. It belongs to a
  colleague: never write, move or delete anything there. Layout:
  `<date>/<date>_drive_<id>_sync/velodyne_points/data/<frame>.bin`.
- Velodyne scans are float32 `[x, y, z, reflectance]`, ~120k points per scan.
- Pairs are temporal: `pc_queries` = frame t, `pc_ref` = frame t-1 of the same drive.
- Subset: 200 consecutive frame pairs chosen with a fixed seed, listed in `results/frames.txt`.
  Copies: `~/pva-knn-study/data/kitti_subset/` (here) and `~/pva-knn/data/kitti_subset/` (loki).
- Quantization (`src/knn_data.py`, `quantize_cm`): clip to +/-120 m in float32, multiply by 100
  in float64, round half to even, convert to int16. Any other loader must match it exactly.
- Test vectors: `testdata/knn1_bucket/` (23 cases with exact expected answers), written by
  `src/gen_knn1_vectors.py`.

## Correctness

Compare against an exact reference on the same quantized inputs: d2 must match exactly in int32;
indices must match except for ties at equal distance. EMU first, then loki. The full matrix is
31 cases (`tests/knn1_bucket_s16_test.cpp`).

## Measurement rules

- Clocks are NOT pinned: loki is shared. It runs in MAXN; do not change nvpmodel or run
  jetson_clocks, and do not ask the user to. Label results `clock_mode=MAXN-dvfs` and save
  `nvpmodel -q` and `jetson_clocks --show` output with each result set.
- PVA performance: follow `pva-build-test` (separate warm-up, >= 1000 samples). Report p50 and
  p99, never only averages.
- Time the stages separately: `prepare()` (CPU), PVA execution, `results()` (CPU), end to end.
- Run `tegrastats` during every benchmark and save the raw lines. If it shows GPU or CPU load
  that isn't ours, discard that run and rerun it.
- Mark results preliminary (MAXN, unpinned clocks, shared machine) until a pinned-clock session
  is agreed with the user.

## Layout and outputs

See `README.md` for the folder layout. Build-test evidence stays in
`pva-operator/knn1_bucket_s16/build-test/runN/`; summaries go in `results/`.

## Next steps

Done 2026-11-14: PVA time split (`results/pva_time_split_20261113/`), then optimization
(`results/pva_optimization_20261114/`, formal evidence `build-test/run2/`): bucket cache + prefetch,
team-box lazy ranking, three-superbank scan, 128-point buckets, both VPUs, faster `prepare()`
(optional CPU threads). kitti_a end to end 124.5 -> 29.2 ms (PVA 105 -> 13.8 ms).

Open:
1. Reuse of bucketing across frames (method change: model first; UNRES-REUSE).
2. Per-bucket scalar overhead in the VPU program (pick + fetch ~40% of cycles).
3. Pinned-clock session, if the user agrees.

Stop and ask the user when: a step needs sudo, something large must be installed, disk usage
would exceed a few GB, or a result contradicts the plan's assumptions.
