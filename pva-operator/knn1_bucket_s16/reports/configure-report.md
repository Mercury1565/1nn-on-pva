# PVA Spec report

## Result

- Result: `CONFIGURED`
- Reason: All 17 requirements are covered by tests. Both finalizer commands report VALID. No blocking questions remain. All six assumptions were approved by the user on 2026-11-13. The inputs were then regenerated (`configure/build_spec.py`) and re-finalized, and both commands still report VALID.
- Operator: `Knn1BucketS16` (`knn1_bucket_s16`)
- Task mode: `new`
- EMU policy: `required`
- Hardware policy: `required`

## Sources

### User sources

- `/home/sitotaem/pva-knn-study/CLAUDE.md`: quantization, exactness, GEN3, EMU then loki, MAXN-dvfs measurement rules.
- The user's requests in the 2026-11-13 session: a 1-NN point-cloud distance kernel using bucketing (median split chosen over Morton), with the proposed decisions approved for the spec ("let's do this").

### Contract evidence

These project files are hashed by the finalizer into `configure/opgen-inputs.json`:

- `src/knn1_bucket_model.py`: step-by-step model of the kernel. It is exact on all 200 KITTI pairs and all 25 test cases, at bucket sizes 512 and 256.
- `src/bucket_compare.py`, `src/knn_data.py`, `src/gen_knn1_vectors.py`.
- `testdata/knn1_bucket/manifest.json`: 25 cases, 17 MB. Expected results come from scipy cKDTree, which is independent of the model. The kitti_a `sum_dist2` (51,148,720) matches the model.
- `results/knn1_bucket_model.csv`, `results/bucket_compare.csv`, `results/frames.txt`.

### SDK references

- Root: `/home/sitotaem/pva-work/references/sdk`, lock sha256 `d1315d2a947b018ba6398307b3140277b19b83e8f64029a682d4ef8bb960eda4`.
- Every cited Markdown file was checked against the lock.

| Markdown file | What it supports |
| --- | --- |
| `html/pva_sdk_apis/classcupva_1_1GatherScatterDataFlow.md` | GSDF |
| `html/tutorial/roi-align-host.md` | GSDF device and host usage |
| `html/pva_sdk_apis/program_listing_file_cupva_types.h.md` | `GSDF_MAX_NUM_TILES` = 32 |
| `html/tutorial/vmem-scalars.md` | VMEM scalars |
| `html/pva_sdk_apis/classcupva_1_1CmdProgram.md` | `updateDataFlows` |
| `html/tutorial/profiling-tools.md` | Profiling |
| `html/tutorial/vmem-buffers-dataflows.md` | DataFlow types |
| `guides/thor/02-architecture-overview.md` | Architecture |
| `guides/thor/05-vector-memory-vmem.md` | VMEM |
| `guides/thor/06-instruction-set-architecture.md` | ISA, overview only |

Raw source SHA-256 values are in `configure/evidence.json`.

### Assumptions

All six were approved by the user on 2026-11-13:

- ASM-PLANAR
- ASM-HOST-IO
- ASM-API
- ASM-STATS
- ASM-ONE-VPS
- ASM-SINGLE-THREAD-PREP

### Conflicts

- CONFLICT-CLOCKS: the build-test procedure prefers pinned clocks. CLAUDE.md wins: `clock_mode=MAXN-dvfs`.

## Outputs

- Operator specification: `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/configure/operator-spec.json`
- Test matrix: `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/configure/test-matrix.json`
- Evidence manifest: `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/configure/evidence.json`
- OpGen inputs: `/home/sitotaem/pva-knn-study/pva-operator/knn1_bucket_s16/configure/opgen-inputs.json`
- Active workspace: `/home/sitotaem/pva-knn-study`

## Coverage

- Requirement count: 17
- Correctness case count: 31 (23 data cases plus 8 API, error, repeat and statistics cases)
- Performance case count: 2 (PERF-KITTI-A, PERF-KITTI-B; acceptance `not_set`)
- Uncovered requirements: none
- Blocking questions: 0
- Non-blocking unresolved items:
  - UNRES-B256
  - UNRES-SECOND-VPS
  - UNRES-PREP-ON-PVA
  - UNRES-THREADS

## Re-finalization 2026-11-13 (model rename)

- During Build-Test run1 (around 22:47 UTC), the user edited `src/knn1_bucket_model.py`:
  - renamed `ref`→`pc1`, `qry`→`pc2`, `B`→`BUCKETS`;
  - renamed the CSV column `n_ref`→`n_pc1`.
- The user approved keeping the edit. Claude restored two comment headers that the rename had changed ("Part BUCKETS" back to "Part B").
- The model logic is unchanged. Rerun on pair 0: fetched_total 11763, sum_d2 51148720, 0 mismatches. This is identical to `results/knn1_bucket_model.csv`.
- The handoff was re-finalized, and both finalizer commands report VALID.
- Only the model's evidence entry changed:
  - before: sha256 `bb3f2966…a274`, 7089 bytes;
  - after: sha256 `1425c215…9ef1`, 7203 bytes.
- No requirement, test case or other evidence changed.
- Note: the existing `results/knn1_bucket_model.csv` still uses the column name `n_ref`. A rerun of the model writes `n_pc1`.

## Re-finalization 2026-11-13 (project scoped to 1-NN only)

- The user scoped the project to the 1-NN kernel only. Earlier work, including the `knn_bf_s16` kernel, was moved to `~/pva-knn-archive`.
- Assumption ASM-PLANAR's text no longer refers to that kernel:
  - before: "Inputs are planar int16 [3][n] (same convention as knn_bf_s16)."
  - after: "Inputs are planar int16 [3][n] (all x, then all y, then all z)."
- Its meaning is unchanged, and so is its approval.
- `evidence.json` changed only in that line. `operator-spec.json` and `test-matrix.json` are byte-identical.
- The handoff was re-finalized, and both finalizer commands report VALID (17 requirements, 31 correctness cases, 2 performance cases).
- The eight recorded project files (model, comparison, generator, loader, frames list, two CSVs, test-data manifest) did not move.

## Re-finalization 2026-11-14 (optimization)

- The user asked to optimize as much as possible (cheaper ranking, faster scan, both VPUs,
  prefetch, and anything else), then to finish up.
- Model `src/knn1_bucket_model.py` changed (hash-recorded file):
  - Step 6 ranks by the team box (`team_lb`), with key `(team_lb >> id_bits) << id_bits | bucket`;
    `id_bits` follows from the bucket size; default `--bucket` is now 128;
  - docstring note: the kernel works out the Step 6 keys lazily per group of 16 buckets, with the
    same visiting order.
  - Check on pair 0 (kitti_a) at bucket 128: sum_d2 51148720, 0 d2 / idx mismatches,
    fetched_total 22622 (kernel: 22449; equal coordinates can fall into different buckets, which
    the spec allows).
- `src/compare_pva_python.py`: `BUCKET` 512 → 128 (not hash-recorded).
- Requirement text updated in `configure/build_spec.py` (no requirement added or removed):
  REQ-SIZES (1024 × 128), REQ-PREP (optional CPU threads), REQ-BUCKET-SIZE (default 128; 256 and
  512 still build), REQ-API (`init(vpus = 2, cpu_threads = 1)`, `vpus()` instead of `program()`,
  `kMaxCpuThreads`), REQ-STATS (`dma_fetches`), REQ-GEN3 (both VPS), DC-VMEM, DC-CANDIDATE-VPS
  (implemented), `kBucketSize` 128, performance workload (bucket 128, 2 VPUs, 1 CPU thread).
- Unresolved items: UNRES-B256, UNRES-SECOND-VPS, UNRES-THREADS resolved and removed;
  UNRES-REUSE added (reuse of bucketing across frames; non-blocking).
- Assumptions: ASM-ONE-VPS replaced by ASM-TWO-VPS; ASM-SINGLE-THREAD-PREP replaced by
  ASM-PREP-THREADS.
- Handoff re-finalized; both finalizer commands report VALID (17 requirements, 31 correctness
  cases, 2 performance cases).
