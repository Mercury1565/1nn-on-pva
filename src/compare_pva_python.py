#!/usr/bin/python3
"""Compare the PVA kernel with the Python model on one pair of KITTI scans.

1. Run the PVA program (knn1_bucket_s16_kitti) on loki for pc_ref = frame t-1 and
   pc_queries = frame t, and copy its per-point answers back.
2. Run the Python model (knn1_bucket_model.run_frame) on the same two scans here.
3. Compare them point by point and print the cloud distance from both.

Edit the settings below, then run:  /usr/bin/python3 src/compare_pva_python.py
(System Python: it has numpy and scipy; conda's python3 here does not.)
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_data as kd  # noqa: E402
import knn1_bucket_model as model  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------
# Which KITTI scans
DATE = "2011_09_26"
DRIVE = "2011_09_26_drive_0001_sync"
REF_FRAME = "0000000089"
QUERY_FRAME = "0000000090"

# The same KITTI subset on both machines (layout <date>/<drive>/velodyne_points/data/<frame>.bin).
LOCAL_KITTI_ROOT = os.path.expanduser("~/pva-knn-study/data/kitti_subset")
LOKI_KITTI_ROOT = "~/pva-knn/data/kitti_subset"  # expanded by the shell on loki

# PVA program, built on this machine for loki:
PVA_PROGRAM = os.path.expanduser("~/pva-knn-study/pva-operator/knn1_bucket_s16/build/l4t/knn1_bucket_s16_kitti")

# loki: SSH host and a scratch directory, removed after the run.
LOKI = "loki"
LOKI_RUN_DIR = "/tmp/pva-knn1-compare"

# Bucket size; must match the kernel (K1_BUCKET in vpu/knn1_bucket_s16_params.h).
BUCKET = 128

# Where the PVA answers are stored on this machine.
OUT_DIR = os.path.expanduser("~/pva-knn-study/results/compare_pva_python")
# ------------------------------------------------------------------------------------------------

SSH = ["ssh", "-o", "BatchMode=yes", LOKI]
SCP = ["scp", "-q", "-o", "BatchMode=yes"]


def scan_path(root, frame):
    return kd.bin_path(root, DATE, DRIVE, frame)


def sha256_local(path):
    return subprocess.run(["sha256sum", path], check=True, capture_output=True, text=True).stdout.split()[0]


def run_on_pva():
    """Run the PVA program on loki. Returns (dist2, idx, buckets_fetched, sum_d2) as printed by it."""
    if not os.path.isfile(PVA_PROGRAM):
        sys.exit(f"PVA program not found: {PVA_PROGRAM} (see the build commands in the settings)")

    ref_remote, qry_remote = scan_path(LOKI_KITTI_ROOT, REF_FRAME), scan_path(LOKI_KITTI_ROOT, QUERY_FRAME)
    ref_local, qry_local = scan_path(LOCAL_KITTI_ROOT, REF_FRAME), scan_path(LOCAL_KITTI_ROOT, QUERY_FRAME)

    # Both machines must read the very same files.
    remote_sums = subprocess.run(SSH + [f"sha256sum {ref_remote} {qry_remote}"], check=True, capture_output=True,
                                 text=True).stdout.split()[0::2]
    
    if remote_sums != [sha256_local(ref_local), sha256_local(qry_local)]:
        sys.exit("the scans on loki differ from the local copies")

    os.makedirs(OUT_DIR, exist_ok=True)
    program = f"{LOKI_RUN_DIR}/knn1_bucket_s16_kitti"
    
    try:
        subprocess.run(SSH + [f"rm -rf {LOKI_RUN_DIR} && mkdir -p {LOKI_RUN_DIR}"], check=True)
        subprocess.run(SCP + [PVA_PROGRAM, f"{LOKI}:{program}"], check=True)
        
        run = subprocess.run(SSH + [f"{program} {ref_remote} {qry_remote} {LOKI_RUN_DIR}"], capture_output=True,
                             text=True)
        print(run.stdout.rstrip())
        
        if run.returncode != 0:
            sys.exit(f"PVA program failed (exit {run.returncode}): {run.stderr.strip()}")
        subprocess.run(SCP + [f"{LOKI}:{LOKI_RUN_DIR}/dist2.s32", f"{LOKI}:{LOKI_RUN_DIR}/idx.s32", OUT_DIR],
                       check=True)
    finally:
        subprocess.run(SSH + [f"rm -rf {LOKI_RUN_DIR}"])  # never leave files behind on loki

    fetched = int(re.search(r"buckets fetched (\d+)", run.stdout).group(1))
    sum_d2 = int(re.search(r"sum d2 = (\d+)", run.stdout).group(1))
    dist2 = np.fromfile(os.path.join(OUT_DIR, "dist2.s32"), dtype=np.int32).astype(np.int64)
    idx = np.fromfile(os.path.join(OUT_DIR, "idx.s32"), dtype=np.int32).astype(np.int64)
    return dist2, idx, fetched, sum_d2


def main():
    print(f"pc_ref     = {DATE} {DRIVE} frame {REF_FRAME}")
    print(f"pc_queries = {DATE} {DRIVE} frame {QUERY_FRAME}\n")

    print("--- PVA (loki) ---")
    pva_d2, pva_idx, pva_fetched, pva_printed_sum = run_on_pva()

    print("\n--- Python model (this machine) ---")
    ref_xyz, qry_xyz = kd.load_pair(LOCAL_KITTI_ROOT, (DATE, DRIVE, REF_FRAME, QUERY_FRAME))
    pc_ref, pc_queries = kd.quantize_cm(ref_xyz)[0], kd.quantize_cm(qry_xyz)[0]
    t0 = time.time()
    py_d2, py_idx, _, py_fetched, _ = model.run_frame(pc_ref, pc_queries, BUCKET)
    print(f"{len(pc_ref)} ref points, {len(pc_queries)} query points, done in {time.time() - t0:.1f} s")

    # --- comparison ---------------------------------------------------------------------------
    n = len(pc_queries)
    if len(pva_d2) != n:
        sys.exit(f"PVA returned {len(pva_d2)} answers for {n} query points")

    # The index may legitimately differ when two ref points are equally close (a tie), so check
    # that the PVA's index really is at the minimum distance instead of requiring the same index.
    q64, r64 = pc_queries.astype(np.int64), pc_ref.astype(np.int64)
    idx_in_range = (pva_idx >= 0) & (pva_idx < len(pc_ref))
    d2_at_pva_idx = ((q64 - r64[np.clip(pva_idx, 0, len(pc_ref) - 1)]) ** 2).sum(axis=1)

    d2_mismatch = int(np.count_nonzero(pva_d2 != py_d2))
    idx_bad = int(np.count_nonzero(~idx_in_range | (d2_at_pva_idx != py_d2)))
    idx_tie = int(np.count_nonzero(pva_idx != py_idx)) - idx_bad

    pva_sum, py_sum = int(pva_d2.sum()), int(py_d2.sum())
    print("\n--- Comparison ---")
    print(f"query points compared               {n}")
    print(f"d2 mismatches                        {d2_mismatch}")
    print(f"PVA index not at the min distance    {idx_bad}")
    print(f"index differs only by a tie          {idx_tie}   (allowed)")
    print(f"sum d2  PVA / Python                 {pva_sum} / {py_sum} cm^2")
    print(f"cloud distance PVA / Python          {np.sqrt(pva_sum) / 100:.3f} / {np.sqrt(py_sum) / 100:.3f} m")
    print(f"buckets fetched PVA / Python         {pva_fetched} / {int(py_fetched.sum())}   (informational)")

    ok = d2_mismatch == 0 and idx_bad == 0 and pva_sum == py_sum == pva_printed_sum
    print(f"\nRESULT: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
