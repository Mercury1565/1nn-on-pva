# Exact 1-NN point-cloud distance on the Jetson Thor PVA

For every point of `pc_queries` (frame t), find the nearest point of `pc_ref`
(frame t-1). The cloud distance is sqrt(Σ d²). The search is exact: the ref points are
split into buckets of 512 with bounding boxes, and the query points into teams of 16.

Earlier work outside this scope is archived in `~/pva-knn-archive` (see its `MOVED.md`).

## Layout

```
pva-operator/knn1_bucket_s16/     the PVA operator (PVA workflow layout)
    configure/                    spec, test matrix, evidence, OpGen handoff (pva-spec)
    include/                      public C++ API
    src/                          host code: API, CPU bucket preparation (+ bucket_prep.md)
    vpu/                          VPU program and shared constants (+ knn1_bucket_s16_vmem.md)
    ref/                          independent brute-force CPU reference
    tests/                        correctness tests (31 cases)
    perf/                         hardware performance benchmark
    sample/                       minimal usage example
    docs/                         reference figures (VPU block diagram)
    reports/                      pva-spec / pva-opgen reports
    build-test/                   pva-build-test runs (builds, logs, evidence)
    CMakeLists.txt, README.md
src/                              Python: model, bucketing comparison, test-vector generator
    knn_data.py                   KITTI loader and int16 quantizer
    knn1_bucket_model.py          step-by-step model of the method (the answer key)
    bucket_compare.py             Morton vs median-split bucketing
    gen_knn1_vectors.py           writes testdata/knn1_bucket
    select_frames.py, validate_subset.py   choose and check the KITTI frame subset
results/                          frames.txt, bucket_compare.csv, knn1_bucket_model.csv
testdata/knn1_bucket/             23 test cases with exact expected answers
data/kitti_subset/                the KITTI frames used (local copy)
```

## Where to read

1. [pva-operator/knn1_bucket_s16/README.md](pva-operator/knn1_bucket_s16/README.md):
   what the operator does, its API, how to build and run it.
2. [src/bucket_prep.md](pva-operator/knn1_bucket_s16/src/bucket_prep.md): the CPU
   preparation (Steps 1–5) and the DRAM tables it produces.
3. [vpu/knn1_bucket_s16_vmem.md](pva-operator/knn1_bucket_s16/vpu/knn1_bucket_s16_vmem.md):
   what the VPU keeps in VMEM and why.
