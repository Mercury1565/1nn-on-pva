#!/usr/bin/env python3
"""Writes operator-spec.json, test-matrix.json and evidence.json (strict JSON) for Knn1BucketS16."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
TD = "testdata/knn1_bucket"
man = json.load(open(os.path.join(HERE, "../../..", TD, "manifest.json")))
NAME = "Knn1BucketS16"
REF = "/home/sitotaem/pva-work/references/sdk"
DOC = "/opt/nvidia/pva-sdk-2.10/share/doc"
THOR_PDF = "_downloads/4b0cb81071f7879ee61d3dd450fb77f1/Thor_PVA_VPU_Programmers_Guide.pdf"
THOR_SHA = "6e754ea330087504fc0ddeac6ee8600a3c06e9ca045662221d3b505aa787e02f"
SRC = {
 "roi": ("html/tutorial/roi-align-host.md", "tutorial/roi-align-host.html", "e4e24205f80a37c0d46a409ec05f437682941db0732d814dcbca1cfd27a554f2"),
 "scalars": ("html/tutorial/vmem-scalars.md", "tutorial/vmem-scalars.html", "e19020fd6f371fca500f7cdaa818720e0b014c8e26173d81c3018d85f6a370d9"),
 "gsdf": ("html/pva_sdk_apis/classcupva_1_1GatherScatterDataFlow.md", "pva_sdk_apis/classcupva_1_1GatherScatterDataFlow.html", "ea7644a25f1ef91f636fda851d311caa3adadd0c4150f3d50d4a9cf1de557bfd"),
 "types": ("html/pva_sdk_apis/program_listing_file_cupva_types.h.md", "pva_sdk_apis/program_listing_file_cupva_types.h.html", "0b61630503ee5c28a68f2dfd63a9a20a37e213a85e5225a93237cbb386d0a22a"),
 "prof": ("html/tutorial/profiling-tools.md", "tutorial/profiling-tools.html", "af8c2a35f55d71034a1ebc4fc86704075a7c6edee763adc8370909c59f5cf93c"),
 "dataflows": ("html/tutorial/vmem-buffers-dataflows.md", "tutorial/vmem-buffers-dataflows.html", "b98635d81bdc243e6cef61e645a4659c647963790bc6fc85369fbbc0ecb73fcb"),
 "cmdprog": ("html/pva_sdk_apis/classcupva_1_1CmdProgram.md", "pva_sdk_apis/classcupva_1_1CmdProgram.html", "2119c8d80b7d20ed2941caacf3f18dc684f78aaff057358ebbc82bb27ccd892e"),
}
def cite(key, sec):
    md, src, sha = SRC[key]
    return {"sdk_doc_root": DOC, "local_markdown": f"{REF}/{md}", "packaged_source": src, "source_sha256": sha, "version": "2.10.1", "section": sec}
def thor(md, sec):
    return {"sdk_doc_root": DOC, "local_markdown": f"{REF}/guides/thor/{md}", "packaged_source": THOR_PDF, "source_sha256": THOR_SHA, "version": "2.10.1", "section": sec}

R = lambda i, a, b: {"id": i, "authority": a, "behavior": b}
reqs = [
 R("REQ-PROBLEM", "user_input", "Exact one-directional 1-nearest-neighbour search between two point clouds: for every query point p in P1 (queries), report the smallest squared Euclidean distance to any point of P2 (refs) and the index of one P2 point at that distance. A cloud-to-cloud distance is derived on the host as sqrt(sum over P1 of d2). The symmetric (Chamfer) form is two calls with P1 and P2 swapped."),
 R("REQ-EXACT", "user_input", "Results are exact: dist2[i] equals min over all refs j of d2(q_i, r_j). The search may skip references only when it is proven they cannot be nearer (box lower bound >= the current best); no approximation, sampling or early termination that can change any dist2."),
 R("REQ-SIZES", "user_input", "Point counts are runtime values: 1 <= n_query <= 131072 and 1 <= n_ref <= 131072, independently (n_query != n_ref allowed). 131072 = 1024 buckets x 128 refs (default build) and 8192 teams x 16 queries; it covers a full KITTI HDL-64 frame (~115k-128k points)."),
 R("REQ-INPUT-FORMAT", "user_input", "Inputs are host-memory arrays of int16 centimetre coordinates in planar layout: queries int16[3][n_query] (x plane, y plane, z plane, each contiguous, tight pitch) and refs int16[3][n_ref] likewise. Produced by clamping float metres to [-120 m, +120 m] and rounding half-to-even to cm (src/knn_data.py quantize_cm). The operator copies them; it never modifies them."),
 R("REQ-INPUT-RANGE", "user_input", "Precondition: every coordinate is in [-12000, 12000]. Then every d2 <= 3*24000^2 = 1728000000 < 2^31-1 and is exact in int32. Out-of-range input is unspecified and not checked by the operator."),
 R("REQ-DIST", "user_input", "d2(q, r) = (qx-rx)^2 + (qy-ry)^2 + (qz-rz)^2 computed in integer arithmetic with no rounding, saturation or overflow for in-range inputs. No floating point in distance computation or comparison."),
 R("REQ-OUTPUT-FORMAT", "user_input", "Outputs are host-memory arrays in the ORIGINAL query order: dist2 int32[n_query] (cm^2) and idx int32[n_query] with idx[i] in [0, n_ref) indexing the caller's refs array in its original order, and d2(q_i, r_idx[i]) == dist2[i]."),
 R("REQ-TIES", "user_input", "When several refs share the minimum d2, any of them is a correct idx. Check: dist2 bit-exact against the independent expected file; idx in range and its recomputed d2 equals dist2. Matching the expected idx file is not required."),
 R("REQ-PREP", "user_input", "Bucket preparation (Part A) runs on the host CPU inside the operator's prepare() call: split refs into buckets of kBucketSize by recursive median split of the widest axis, store them bucket by bucket with a per-bucket bounding box, and group queries into teams of 16 the same way. Padding of a partial bucket or team must not change any result (e.g. copies of a real point). prepare() time is part of the operator's cost and is measured and reported separately from PVA time. prepare() may use several CPU threads when the caller asks for it (init cpu_threads, default 1); the results do not depend on the thread count."),
 R("REQ-BUCKET-SIZE", "user_input", "Bucket size and team size 16 queries are compile-time constants (kBucketSize, kTeamSize). kBucketSize is 128 by default (CMake option K1_BUCKET: 128, 256 or 512); 128 was fastest on loki with the optimized kernel (user request 2026-11-14: optimize as much as possible)."),
 R("REQ-API", "skill_workflow", "C++ API in knn1_bucket_s16.hpp, namespace pva_knn, class Knn1BucketS16: static constexpr int32_t kMaxPoints=131072, kBucketSize=K1_BUCKET (default 128), kTeamSize=16, kMaxCpuThreads=16; int32_t init(int32_t vpus = 2, int32_t cpu_threads = 1) (allocate internal cuPVA buffers for the maximum size, create one program per VPU once; vpus in [1, 2], cpu_threads in [1, kMaxCpuThreads]); int32_t prepare(const int16_t* queries, int32_t n_query, const int16_t* refs, int32_t n_ref) (Part A, host pointers); int32_t submit(cupva::Stream&, cupva::Fence&) (asynchronous); int32_t results(int32_t* dist2, int32_t* idx) (host pointers, after the fence; restores original query order); int32_t stats(Knn1Stats*) (see REQ-STATS); int32_t vpus() const; cupva::Error lastStatus() const; static int64_t sumDist2(const int32_t* dist2, int32_t n). All int32_t calls return 0 on success and nonzero on error and never throw."),
 R("REQ-STATS", "user_input", "After a completed submission, stats() reports n_query, n_ref, n_buckets, n_teams and the total number of buckets fetched (scanned) by the PVA over all teams, as counted by the kernel itself, and the number of buckets copied from DRAM into the VPUs' bucket caches (dma_fetches). Used to relate PVA time to work and to compare with the Python model (src/knn1_bucket_model.py); it is diagnostic, not part of correctness, except the sanity bound n_teams <= fetched <= n_teams * n_buckets."),
 R("REQ-ERRORS", "skill_workflow", "Nonzero return, without submitting work and without modifying caller output buffers: prepare() before a successful init(); null pointer or n_query/n_ref outside [1, 131072] in prepare(); submit() before a successful prepare(); results()/stats() before a successful submit(); null pointer in results()/stats(); any cuPVA error (exceptions caught)."),
 R("REQ-ALIASING", "user_input", "Caller buffers are host memory and are only read (inputs) or written (outputs) by prepare()/results(); dist2 and idx must not overlap each other or the inputs. Internal device buffers are owned by the operator."),
 R("REQ-REPEAT", "user_input", "The prepared data can be submitted repeatedly (>= 1000 times) with bit-identical dist2 each time. prepare() can be called again with different sizes between submissions (e.g. large, small, large); each submission's results reflect only the most recent prepare() with no stale data from earlier, larger inputs. One outstanding submission per instance."),
 R("REQ-GEN3", "user_input", "Target PVA GEN3 only (Jetson AGX Thor, L4T R38.4, cuPVA 2.10.1): -DPVA_DEFAULT_GEN=GEN3 -DPVA_ONLY_BUILD_GENS=GEN3, NVVPU compiler, EMU with PVA_GEN=GEN3. Both VPS of the PVA by default (teams split in two contiguous halves); vpus = 1 uses one."),
 R("REQ-PERF", "user_input", "Characterise on loki, per KITTI workload: prepare() host time, PVA task time including all DMA (profiler TOTAL_TIME) and VPU execution time, results() host time, and end-to-end time; p50 and p99 over >= 1000 samples after a separate warm-up; buckets fetched per call; clock_mode=MAXN-dvfs (clocks not pinned; nvpmodel -q and jetson_clocks --show recorded; tegrastats running). EMU and build-host timings are never PVA performance. No acceptance threshold."),
]
ids = [r["id"] for r in reqs]

DC = lambda i, kind, text, ev: {"id": i, "kind": kind, "constraint": text, "evidence": ev}
dcs = [
 DC("DC-VMEM", "platform", "Each Thor VPS has 512 KB VMEM (4 superbanks x 128 KB). P2 at the maximum size (131072 x 3 x int16 = 768 KiB) does NOT fit, so refs must be brought in piecewise (buckets of 128 x 3 x int16 = 768 B; a cache of 480 buckets fits in superbanks B-D). The box table (1024 x 6 x int16 = 12 KiB) fits.", thor("05-vector-memory-vmem.md", "Ch. 5 VMEM, PDF p. 29")),
 DC("DC-GSDF", "platform", "GatherScatterDataFlow moves tiles whose DRAM locations are decided at run time on the VPU (cupvaGSDFOpen / cupvaGSDFUpdateTiles(x, y, n) / cupvaGSDFTrig / cupvaGSDFSync / cupvaGSDFClose). Direction (read to VMEM or write from VMEM) is fixed on the host; the host tile() size sets bandwidth allocation; the source surface must hold at least one tile. At most GSDF_MAX_NUM_TILES = 32 tiles per trigger.", cite("gsdf", "GatherScatterDataFlow::src(), tile(), numTilesPerTrigger(); also tutorial/roi-align-host.md device and host code; cupva_types.h line 94 GSDF_MAX_NUM_TILES")),
 DC("DC-SCALARS", "platform", "The host passes runtime values (e.g. tile counts) to the VPU by writing VMEM scalars through the CmdProgram (prog[\"name\"] = value / .set()). A CmdProgram can be reconfigured and submitted again; CmdProgram::updateDataFlows() updates dataflows of an existing program.", cite("scalars", "vmem-scalars tutorial (reuse and resubmission of CmdProgram); rdf-halo/rdf-tiles tutorials prog[\"tileCount\"]; CmdProgram::updateDataFlows")),
 DC("DC-LANES", "platform", "384-bit vector registers: 16 x 24-bit or 8 x 48-bit lanes (dvintx gives 16 x 48-bit). Coordinate differences fit 24-bit lanes; squares and their sum (<= 1.728e9) need wider lanes (48-bit accumulation), never 24-bit.", thor("06-instruction-set-architecture.md", "6.2.3.1 Extended Precision (PDF p. 47), 6.2.3.5 (PDF p. 50)")),
 DC("DC-COMPARE", "platform", "1-cycle vector compare and min/max instructions; sum-of-squares MAC instructions; up to 2 vector operations per clock.", thor("06-instruction-set-architecture.md", "6.2.3, PDF p. 46-47")),
 DC("DC-TSC", "measurement", "The VPU TSC stops while the VPU waits for DMA, so time including DMA must come from cupva_utils::ProfilingContext (TOTAL_TIME) or host submit-to-fence timing.", cite("prof", "Timestamp Counter (TSC) note; Profiling Stream")),
 DC("DC-DMA", "platform", "Two VPS each with a DMA engine and shared L2 SRAM; data movement through cuPVA DataFlows (RasterDataFlow, SequenceDataFlow, GatherScatterDataFlow).", thor("02-architecture-overview.md", "Ch. 2.1-2.2, PDF p. 12-13")),
 DC("DC-CANDIDATE-LAYOUT", "candidate", "Candidate: refs stored bucket-major as a 2-D surface, one row per bucket = [x0..x511 | y0..y511 | z0..z511] (1536 int16), fetched as one GSDF tile per bucket at (0, b); box table and team list as small arrays; host pads to the maximum dimensions so dataflows compile once, with the actual counts passed as VMEM scalars.", None),
 DC("DC-CANDIDATE-RANK", "candidate", "Candidate for Step 6: instead of sorting all buckets per team, repeatedly pick the nearest not-yet-visited bucket by the team's lower bound (min over the 16 lanes), stopping when that bound >= the largest lane best; teams visit ~1.5-3 buckets on KITTI (results/knn1_bucket_model.csv). Must produce exact results (REQ-EXACT).", None),
 DC("DC-CANDIDATE-OVERLAP", "candidate", "Candidate: double-buffer bucket fetches (fetch the next candidate while scanning the current), and batch query teams and outputs with RasterDataFlow or SequenceDataFlow. A first version without overlap is acceptable.", None),
 DC("DC-CANDIDATE-VPS", "candidate", "Candidate (implemented 2026-11-14): split teams across both VPS.", None),
]

spec = {
 "schema_version": 2, "schema_authority": "skill_workflow", "status": "CONFIGURED",
 "operator": {"name": NAME,
   "purpose": "Exact 1-nearest-neighbour squared distances from every point of one int16 point cloud (up to 131072 points) to another, using median-split buckets (128 points by default) so most reference points are never fetched or compared, on the Thor PVA, for the Mnemosyne PVA feasibility study (cloud-to-cloud distance).",
   "task_mode": "new",
   "public_api": {"language": ["C++"], "headers": ["knn1_bucket_s16.hpp"],
     "symbols": ["pva_knn::Knn1BucketS16", "pva_knn::Knn1Stats", "pva_knn::Knn1BucketS16::kMaxPoints", "pva_knn::Knn1BucketS16::kBucketSize", "pva_knn::Knn1BucketS16::kTeamSize", "pva_knn::Knn1BucketS16::init", "pva_knn::Knn1BucketS16::prepare", "pva_knn::Knn1BucketS16::submit", "pva_knn::Knn1BucketS16::results", "pva_knn::Knn1BucketS16::stats", "pva_knn::Knn1BucketS16::program", "pva_knn::Knn1BucketS16::lastStatus", "pva_knn::Knn1BucketS16::sumDist2"]}},
 "platform": {"pva_sdk_version": "2.10.1", "pva_generations": ["GEN3"], "operating_systems": ["L4T"],
   "emulator": {"policy": "required"},
   "safety": {"required": False, "standard": None, "constraints": []},
   "interoperability": []},
 "requirements": reqs,
 "resources": {
   "inputs": [
     {"name": "queries", "type": "int16", "shape": [3, "n_query"], "layout": "planar x,y,z; tight pitch", "memory": "host memory (copied by prepare)", "range": [-12000, 12000], "units": "cm"},
     {"name": "refs", "type": "int16", "shape": [3, "n_ref"], "layout": "planar x,y,z; tight pitch", "memory": "host memory (copied by prepare)", "range": [-12000, 12000], "units": "cm"}],
   "outputs": [
     {"name": "dist2", "type": "int32", "shape": ["n_query"], "layout": "original query order", "memory": "host memory (written by results)", "range": [0, 1728000000], "units": "cm^2"},
     {"name": "idx", "type": "int32", "shape": ["n_query"], "layout": "original query order", "memory": "host memory (written by results)", "range": [0, "n_ref-1"]},
     {"name": "stats", "type": "Knn1Stats", "shape": [1], "layout": "struct {int32 n_query, n_ref, n_buckets, n_teams; int64 buckets_fetched}", "memory": "host memory (written by stats)", "range": None}]},
 "parameters": [
   {"name": "n_query", "value": "1..131072", "kind": "runtime"},
   {"name": "n_ref", "value": "1..131072", "kind": "runtime"},
   {"name": "kMaxPoints", "value": 131072, "kind": "compile_time_constant"},
   {"name": "kBucketSize", "value": 128, "kind": "compile_time_constant"},
   {"name": "kTeamSize", "value": 16, "kind": "compile_time_constant"}],
 "behavior": {
   "border": {"modes": [], "constant_value": None, "halo": None},
   "arithmetic": {"rounding": "none: exact integer arithmetic", "saturation": "none for in-range inputs (max d2 1728000000)", "overflow": "impossible for in-range inputs; sumDist2 accumulates in int64", "nan": "not applicable (integer inputs)", "infinity": "not applicable (integer inputs)"},
   "aliasing": {"in_place": False, "permitted_pairs": [], "prohibited_pairs": [["dist2", "idx"], ["dist2", "queries"], ["dist2", "refs"], ["idx", "queries"], ["idx", "refs"]]},
   "execution": {"asynchronous": True, "stream_rules": "submit() enqueues on the given stream; one outstanding submission per instance; prepare() must not be called while a submission is outstanding.", "fence_rules": "results() and stats() are valid only after the fence passed to submit() expires.", "repeated_submission": "Supported; bit-identical dist2; re-prepare with any valid sizes between submissions."},
   "errors": [
     {"condition": "prepare before init; null pointer; n_query or n_ref outside [1, 131072]", "result": "nonzero; nothing submitted; caller buffers untouched"},
     {"condition": "submit before a successful prepare", "result": "nonzero"},
     {"condition": "results or stats before a successful submit, or null pointer", "result": "nonzero; caller buffers untouched"},
     {"condition": "cuPVA error", "result": "nonzero; exception not propagated"}]},
 "design_constraints": dcs,
 "correctness_contracts": [
   {"requirement_ids": ["REQ-PROBLEM", "REQ-EXACT", "REQ-SIZES", "REQ-INPUT-FORMAT", "REQ-INPUT-RANGE", "REQ-DIST", "REQ-OUTPUT-FORMAT", "REQ-TIES", "REQ-PREP"],
    "reference": "Independent exact 1-NN from scipy cKDTree on the quantized int16 coordinates, d2 recomputed in int64 (src/gen_knn1_vectors.py -> testdata/knn1_bucket). Cross-checked offline: the step-by-step bucket model src/knn1_bucket_model.py reproduces every case exactly at bucket sizes 512 and 256 (and is bucket-size generic).",
    "rule": "dist2 equal to expected bit-exactly for every query (tolerance 0); every idx in [0, n_ref) and host-recomputed d2(q_i, r_idx[i]) == dist2[i]; sumDist2(dist2) equals the manifest sum_dist2."},
   {"requirement_ids": ["REQ-API", "REQ-ERRORS", "REQ-ALIASING", "REQ-REPEAT", "REQ-STATS"],
    "reference": "API behaviour from this spec.",
    "rule": "Return codes as specified; caller buffers untouched on error; repeated and re-prepared submissions correct; stats within sanity bounds."}],
 "performance_contract": {"acceptance": "not_set", "target_source": None,
   "sol_model": {"required": False, "assumptions": [], "limiting_resource": None, "predicted_time_or_cycles": None, "acceptance_rule": None}},
 "unresolved_requirements": [
   {"id": "UNRES-PREP-ON-PVA", "blocking": False, "question": "Whether bucket preparation should later move from the CPU to the PVA, if prepare() time turns out to dominate."},
   {"id": "UNRES-REUSE", "blocking": False, "question": "Whether to reuse frame t's query bucketing as frame t+1's ref bucketing (needs teams to subdivide buckets; method change, model first)."}]
}

C = lambda cid, rids, **kw: {"id": cid, "requirement_ids": rids, **kw}
base = ["REQ-PROBLEM", "REQ-EXACT", "REQ-SIZES", "REQ-INPUT-FORMAT", "REQ-INPUT-RANGE", "REQ-DIST", "REQ-OUTPUT-FORMAT", "REQ-TIES", "REQ-PREP", "REQ-BUCKET-SIZE", "REQ-API", "REQ-GEN3"]
rule = "dist2 bit-exact vs expected; idx in [0, n_ref) and recomputed d2 == dist2; sumDist2 == manifest sum_dist2; stats sanity bound holds"
cases = []
for m in man["cases"]:
    cid = m["case"]
    cases.append(C("CORR-" + cid.upper().replace("_", "-"), base + ["REQ-STATS"],
      input={"queries": f"{TD}/{cid}/queries.s16", "refs": f"{TD}/{cid}/refs.s16", "n_query": m["n_query"], "n_ref": m["n_ref"],
             "sha256": {k: v for k, v in m["sha256"].items() if k in ("queries.s16", "refs.s16")}},
      expected={"dist2": f"{TD}/{cid}/dist2.s32", "idx_informational": f"{TD}/{cid}/idx.s32", "sum_dist2": m["sum_dist2"],
                "sha256": {k: v for k, v in m["sha256"].items() if k in ("dist2.s32", "idx.s32")}, "source": "scipy cKDTree exact 1-NN, d2 in int64"},
      description=m["description"], tie_profile={"queries_with_tied_nearest": m["queries_with_tied_nearest"], "min_dist2": m["min_dist2"], "max_dist2": m["max_dist2"]},
      tolerance=0, check=rule, run_on=["EMU", "hardware"]))
cases += [
 C("CORR-REPEATED-SUBMISSION", ["REQ-REPEAT", "REQ-API", "REQ-EXACT"],
   input={"case": "kitti_a", "submissions": 100}, expected={"rule": "every submission passes the case check; dist2 bit-identical across all 100"}, tolerance=0, run_on=["EMU", "hardware"]),
 C("CORR-REPREPARE-SIZES", ["REQ-REPEAT", "REQ-SIZES", "REQ-PREP"],
   input={"sequence": ["max_131072", "small_17x513", "kitti_b", "small_1x1", "max_131072"], "procedure": "one instance: prepare, submit, wait, results, check, for each case in order; output buffers prefilled with sentinel 0x7FFFFFFF each time"},
   expected={"rule": "each step passes its own case check; no element beyond n_query is written; no stale data"}, tolerance=0, run_on=["EMU", "hardware"]),
 C("CORR-ASYNC-FENCE", ["REQ-REPEAT", "REQ-API"],
   input={"case": "kitti_16k", "procedure": "submit, wait on fence, then results into sentinel-filled buffers"},
   expected={"rule": "no sentinel remains in [0, n_query); case check passes"}, tolerance=0, run_on=["EMU", "hardware"]),
 C("CORR-SUMDIST2-INT64", ["REQ-PROBLEM", "REQ-API"],
   input={"cases": ["extremes", "all_refs_identical", "max_131072"]},
   expected={"rule": "sumDist2 equals manifest sum_dist2 (all three exceed 2^31) exactly"}, tolerance=0, run_on=["EMU", "hardware"]),
 C("CORR-STATS-VS-MODEL", ["REQ-STATS"],
   input={"cases": ["kitti_a", "kitti_b", "kitti_c", "kitti_d"]},
   expected={"rule": "informational: report stats buckets_fetched next to the Python model's count for the same case (src/knn1_bucket_model.py, same bucket size); must satisfy n_teams <= fetched <= n_teams*n_buckets; exact equality not required (bucket contents may differ on equal coordinates)"},
   tolerance=None, run_on=["EMU", "hardware"]),
 C("ERR-ORDER", ["REQ-ERRORS", "REQ-API"],
   input={"procedure": "fresh instance: prepare before init; submit before prepare; results and stats before submit"},
   expected={"rule": "each returns nonzero; no exception; sentinel-filled caller buffers unchanged"}, tolerance=0, run_on=["EMU", "hardware"]),
 C("ERR-ARGS", ["REQ-ERRORS", "REQ-SIZES"],
   input={"procedure": "after init: prepare with each null pointer, n_query or n_ref = 0, -1, 131073; results/stats with null pointers after a valid submit"},
   expected={"rule": "each returns nonzero; no exception; a later valid prepare/submit still passes the small_16x512 check"}, tolerance=0, run_on=["EMU", "hardware"]),
 C("CORR-NO-ALIAS", ["REQ-ALIASING"],
   input={"procedure": "all cases use separate host vectors for queries, refs, dist2, idx; harness asserts disjoint ranges; inputs compared to a saved copy after results()"},
   expected={"rule": "assertion holds; inputs unchanged; all data cases pass"}, tolerance=0, run_on=["EMU", "hardware"]),
]
def perf_case(cid, case):
    m = next(x for x in man["cases"] if x["case"] == case)
    return C(cid, ["REQ-PERF", "REQ-GEN3", "REQ-PREP", "REQ-STATS"],
      workload={"inputs": f"{TD}/{case}", "n_query": m["n_query"], "n_ref": m["n_ref"], "bucket_size": 128, "vpus": 2, "cpu_threads": 1},
      metrics=["prepare_us (host CPU, loki)", "pva_total_us = ProfilingContext TOTAL_TIME incl. DMA", "pva_exec_us = EXECUTION_TIME", "results_us (host CPU)", "end_to_end_us = prepare + submit-to-fence wall + results", "buckets_fetched (stats)"],
      statistics=["p50", "p99", "min", "max"], warmup_submissions=100, samples=1000,
      target_conditions={"device": "loki, Jetson AGX Thor, L4T R38.4, cuPVA 2.10.1, both VPS", "clocks": "clock_mode=MAXN-dvfs: MAXN, clocks not pinned (shared machine, no nvpmodel/jetson_clocks changes); nvpmodel -q and jetson_clocks --show saved with results", "monitoring": "tegrastats raw lines saved; runs with foreign GPU/CPU load are discarded and rerun", "gpu_load": "idle (0%)"},
      correctness_gate=f"the last timed submission passes the CORR-{case.upper().replace('_', '-')} check",
      acceptance="not_set", run_on=["hardware"])
perf = [perf_case("PERF-KITTI-A", "kitti_a"), perf_case("PERF-KITTI-B", "kitti_b")]
matrix = {"schema_version": 1, "schema_authority": "skill_workflow", "operator": NAME, "specification": "operator-spec.json",
 "execution_policy": {"emulator": "required", "hardware": "required"},
 "correctness_cases": cases, "performance_cases": perf,
 "coverage": {"requirement_ids": ids, "uncovered_requirement_ids": [], "unresolved_cases": []}}

def pub(key, title, sections, rids, dcids):
    md, src, sha = SRC[key]
    return {"title": title, "sdk_doc_root": DOC, "local_markdown": f"{REF}/{md}", "packaged_source": src, "source_sha256": sha, "version": "2.10.1", "sections": sections, "requirement_ids": rids, "design_constraints": dcids}
def pubthor(md, title, sections, rids, dcids):
    return {"title": title, "sdk_doc_root": DOC, "local_markdown": f"{REF}/guides/thor/{md}", "packaged_source": THOR_PDF, "source_sha256": THOR_SHA, "version": "2.10.1", "sections": sections, "requirement_ids": rids, "design_constraints": dcids}
ev = {"schema_version": 2, "schema_authority": "skill_workflow",
 "additional_evidence_used": [
   {"category": "reference_implementation", "relative_path": "src/knn1_bucket_model.py", "requirement_ids": ["REQ-PROBLEM", "REQ-EXACT", "REQ-PREP", "REQ-BUCKET-SIZE", "REQ-STATS"], "used_for": "Step-by-step model of the kernel (Part A bucketing, Step 6 ranking, stop/skip rule, scan). Exact on all 200 KITTI pairs and all test cases; source of expected fetch counts."},
   {"category": "reference_implementation", "relative_path": "src/bucket_compare.py", "requirement_ids": ["REQ-PREP", "REQ-BUCKET-SIZE"], "used_for": "Median-split ordering used by the model; Morton vs median comparison that selected median splits."},
   {"category": "reference_implementation", "relative_path": "src/knn_data.py", "requirement_ids": ["REQ-INPUT-FORMAT", "REQ-INPUT-RANGE"], "used_for": "Quantization to int16 cm with clamp +/-120 m and KITTI loading."},
   {"category": "test_vector_generator", "relative_path": "src/gen_knn1_vectors.py", "requirement_ids": ["REQ-SIZES", "REQ-OUTPUT-FORMAT", "REQ-TIES", "REQ-EXACT"], "used_for": "Generates the 25 correctness cases with independent cKDTree expected results."},
   {"category": "test_vectors", "relative_path": f"{TD}/manifest.json", "requirement_ids": ["REQ-EXACT", "REQ-TIES", "REQ-SIZES"], "used_for": "Case list with sizes, SHA-256, sum_dist2 and tie counts."},
   {"category": "workload", "relative_path": "results/knn1_bucket_model.csv", "requirement_ids": ["REQ-STATS", "REQ-PERF"], "used_for": "Model work counts (buckets fetched, box checks, distance evaluations) on 200 KITTI pairs at bucket sizes 256 and 512."},
   {"category": "workload", "relative_path": "results/bucket_compare.csv", "requirement_ids": ["REQ-PREP"], "used_for": "Morton vs median-split comparison on 200 KITTI pairs."},
   {"category": "workload", "relative_path": "results/frames.txt", "requirement_ids": ["REQ-PERF"], "used_for": "Fixed KITTI frame-pair list the KITTI cases are drawn from."}],
 "public_sources": [
   pub("gsdf", "cuPVA API: cupva::GatherScatterDataFlow", "src(), dst(), tile(), numTilesPerTrigger()", ["REQ-PREP"], ["DC-GSDF"]),
   pub("roi", "PVA SDK tutorial: ROI Gather via GatherScatter Dataflow (host and device code)", "Device code (cupvaGSDFOpen/UpdateTiles/Trig/Sync/Close); Host code (GatherScatterDataFlow setup)", [], ["DC-GSDF"]),
   pub("types", "cuPVA header listing: cupva_types.h", "line 94: #define GSDF_MAX_NUM_TILES (32U)", [], ["DC-GSDF"]),
   pub("scalars", "PVA SDK tutorial: VMEM scalars and batch submission", "CmdProgram reuse and resubmission", ["REQ-SIZES", "REQ-REPEAT"], ["DC-SCALARS"]),
   pub("cmdprog", "cuPVA API: cupva::CmdProgram", "updateDataFlows()", ["REQ-REPEAT"], ["DC-SCALARS"]),
   pub("prof", "PVA SDK tutorial: Profiling Tools", "Timestamp Counter (TSC); Profiling Stream", ["REQ-PERF"], ["DC-TSC"]),
   pub("dataflows", "PVA SDK tutorial: VMEM buffers and DataFlows", "DataFlow types", [], ["DC-DMA"]),
   pubthor("02-architecture-overview.md", "Thor PVA VPU Programmer's Guide, Ch. 2 Architecture Overview", "PDF p. 12-14", ["REQ-GEN3"], ["DC-DMA"]),
   pubthor("05-vector-memory-vmem.md", "Thor PVA VPU Programmer's Guide, Ch. 5 VMEM", "PDF p. 29", ["REQ-SIZES"], ["DC-VMEM"]),
   pubthor("06-instruction-set-architecture.md", "Thor PVA VPU Programmer's Guide, Ch. 6 ISA (overview only)", "6.2.3 (PDF p. 46-47), 6.2.3.1 (p. 47), 6.2.3.5 (p. 50)", ["REQ-DIST"], ["DC-LANES", "DC-COMPARE"])],
 "user_sources": [
   {"source": "/home/sitotaem/pva-knn-study/CLAUDE.md: int16 cm quantization with +/-120 m clamp, int32 distances, exact reference on quantized inputs, EMU then loki, GEN3 only, MAXN-dvfs measurement rules, p50/p99, >= 1000 samples, tegrastats.", "requirement_ids": ["REQ-INPUT-FORMAT", "REQ-INPUT-RANGE", "REQ-DIST", "REQ-EXACT", "REQ-TIES", "REQ-GEN3", "REQ-PERF"]},
   {"source": "User, session 2026-11-14: 'Do as much optimization as possible including these 4 [cheaper ranking with the team box, faster scan loop, both VPUs, double buffering/prefetch] ... let's see how much we can reduce the overall time'; then 'go in that order and finish up'.", "requirement_ids": ["REQ-BUCKET-SIZE", "REQ-GEN3", "REQ-PREP", "REQ-API", "REQ-STATS", "REQ-PERF"]},
   {"source": "User, session 2026-11-13: build a 1-NN point-cloud distance PVA kernel (for each P1 point the nearest P2 point and its distance, combined with a Euclidean norm); bucketing approach; 'try it both ways' (Morton vs median, result: median); approved proceeding with the proposed decisions (outputs d2+idx, host sqrt(sum d2), runtime sizes up to 131072, CPU bucket preparation shipped with the operator and timed separately, bucket 512 first, one VPS).", "requirement_ids": ["REQ-PROBLEM", "REQ-EXACT", "REQ-SIZES", "REQ-OUTPUT-FORMAT", "REQ-PREP", "REQ-BUCKET-SIZE", "REQ-STATS", "REQ-ALIASING", "REQ-REPEAT", "REQ-PERF"]}],
 "assumptions": [
   {"id": "ASM-PLANAR", "text": "Inputs are planar int16 [3][n] (all x, then all y, then all z).", "requirement_ids": ["REQ-INPUT-FORMAT"], "disposition": "proposed by Claude; approved by the user 2026-11-13"},
   {"id": "ASM-HOST-IO", "text": "Inputs and outputs are host-memory arrays (not cuPVA allocations) because prepare() reorders them on the CPU anyway and results() restores the original order on the CPU; internal device buffers are owned by the operator.", "requirement_ids": ["REQ-API", "REQ-INPUT-FORMAT", "REQ-OUTPUT-FORMAT", "REQ-ALIASING"], "disposition": "proposed by Claude; approved by the user 2026-11-13"},
   {"id": "ASM-API", "text": "API shape init/prepare/submit/results/stats with nonzero return codes and no exceptions; sumDist2 helper in int64; the sqrt and Chamfer combination are left to the caller.", "requirement_ids": ["REQ-API", "REQ-ERRORS", "REQ-PROBLEM"], "disposition": "proposed by Claude; approved by the user 2026-11-13"},
   {"id": "ASM-STATS", "text": "The kernel counts buckets fetched and reports them through stats() (diagnostic output, small extra DMA).", "requirement_ids": ["REQ-STATS"], "disposition": "proposed by Claude; approved by the user 2026-11-13"},
   {"id": "ASM-TWO-VPS", "text": "Both VPS by default, each with its own VPU program for a contiguous half of the teams (replaces ASM-ONE-VPS of 2026-11-13).", "requirement_ids": ["REQ-GEN3", "REQ-PERF"], "disposition": "user request 2026-11-14 (optimize, including both VPUs)"},
   {"id": "ASM-PREP-THREADS", "text": "prepare() runs on one CPU thread by default (keeps the CPU load low); more threads on request via init(). Performance cases use 1 thread; other thread counts are reported separately as preliminary.", "requirement_ids": ["REQ-PREP", "REQ-PERF"], "disposition": "proposed by Claude 2026-11-14 under the user's optimization request"}],
 "conflicts": [
   {"id": "CONFLICT-CLOCKS", "description": "The pva-build-test performance procedure prefers pinned clocks; CLAUDE.md forbids changing clocks on the shared loki.", "resolution": "CLAUDE.md (explicit user requirement) wins: clock_mode=MAXN-dvfs, clocks recorded, not pinned.", "requirement_ids": ["REQ-PERF"]}]}

for fn, obj in (("operator-spec.json", spec), ("test-matrix.json", matrix), ("evidence.json", ev)):
    with open(os.path.join(HERE, fn), "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False); f.write("\n")
print(len(ids), "requirements;", len(cases), "correctness;", len(perf), "performance")
