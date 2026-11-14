"""Step-by-step model of the planned PVA kernel knn1_bucket_s16 (exact 1-NN with median-split buckets).

It follows the kernel's steps one to one, so it serves as (1) the answer key for the PVA kernel and
(2) a counter of the work the kernel will really do. It is checked against scipy's KD-tree.

Part A (CPU, before the PVA)
  1. split pc1s (P2) at the median of the widest axis into buckets of BUCKETS points
  2. store pc1s bucket by bucket; pad the last bucket with copies of one of its own points
  3. box table: per bucket min/max x, y, z
  4. split queries (P1) the same way into teams of 16 (pad the last team with copies of a query)
Part B (PVA, per team of 16 queries)
  5. load the team (16 lanes)
  6. per bucket: team value = smallest possible distance^2 from the team's bounding box to the bucket's
     box (a lower bound for all 16 lanes); rank key = (team value with its low ID_BITS bits cleared) | bucket
  7. best[lane] = "infinity"
  8. walk buckets in key order: stop when key bound >= max(best) (no later bucket can help);
     otherwise skip the bucket if no lane can improve (lane value >= best for all lanes);
     otherwise fetch it and scan its BUCKETS points, keeping the per-lane minimum
  9. write best d^2 and its pc1 index per lane
     (The kernel works out the step 6 keys lazily, 16 buckets at a time, only for groups of buckets that
      come up; a group's box bound is <= each of its buckets' bounds, so the visiting order, and every
      count below, is exactly the one of this model. See rankGroups() in the kernel.)
Part C
 11. total = sqrt(sum d^2) in int64

Coordinates: the study's int16 cm quantization (src/knn_data.py). All distance arithmetic is integer.
Index ties (equal d^2) are resolved to the first point met in scan order; tests must accept any index
with the same d^2.

usage: /usr/bin/python3 src/knn1_bucket_model.py [--pairs N] [--bucket 128] [--jobs 10]
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_data as kd  # noqa: E402
from bucket_compare import median_order  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "kitti_subset")
TEAM = 16
MAX_POINTS = 131072
INF = np.int64(2**31 - 1)


# ---------- Part A: preparation on the CPU ------------------------------------------------------
def prepare(pc1, pc2, BUCKETS):
    """pc1, pc2: int16 (n, 3). Returns the arrays the PVA would receive."""
    ro = median_order(pc1, BUCKETS)                                   # step 1
    nb = (len(ro) + BUCKETS - 1) // BUCKETS
    pad = nb * BUCKETS - len(ro)
    ro = np.concatenate([ro, np.repeat(ro[-1], pad)])           # step 2 padding: copies, answer unchanged
    pc1_b = pc1[ro].reshape(nb, BUCKETS, 3)                           # bucket-major
    lo, hi = pc1_b.min(1), pc1_b.max(1)                          # step 3
    qo = median_order(pc2, TEAM)                                 # step 4
    nt = (len(qo) + TEAM - 1) // TEAM
    qo_p = np.concatenate([qo, np.repeat(qo[-1], nt * TEAM - len(qo))])
    teams = pc2[qo_p].reshape(nt, TEAM, 3)
    return dict(pc1_b=pc1_b, pc1_idx=ro.reshape(nb, BUCKETS), lo=lo, hi=hi, teams=teams,
                pc2_idx=qo_p.reshape(nt, TEAM), n_query=len(pc2))


# ---------- Part B: one team on the PVA -----------------------------------------------------------
def id_bits(BUCKETS):
    """Bits of a bucket id in the kernel's rank key: log2 of the most buckets a cloud can have."""
    return int(np.log2(MAX_POINTS // BUCKETS))


def run_team(t, pc1_b, lo, hi):
    q = t.astype(np.int64)                                       # step 5: (16, 3)
    lo, hi = lo.astype(np.int64), hi.astype(np.int64)
    gap = np.maximum(np.maximum(lo[None] - q[:, None], q[:, None] - hi[None]), 0)
    lb2 = (gap * gap).sum(-1)                                    # per lane, used by the skip rule (step 8)
    tgap = np.maximum(np.maximum(lo - q.max(0), q.min(0) - hi), 0)
    team_lb = (tgap * tgap).sum(-1)                              # step 6: team box to bucket box, (nb,)
    bits = id_bits(pc1_b.shape[1])
    key_bound = (team_lb >> bits) << bits                        # rank key without the id: still a lower bound
    order = np.argsort(key_bound | np.arange(len(lo)))           # = the kernel's order of keys
    best = np.full(TEAM, INF)                                    # step 7
    best_j = np.full((TEAM, 2), -1, dtype=np.int64)              # (bucket, slot)
    visited = fetched = 0
    
    for b in order:                                              # step 8
        visited += 1
        
        if key_bound[b] >= best.max():
            break
        
        if np.all(lb2[:, b] >= best):
            continue
        
        fetched += 1
        d = pc1_b[b].astype(np.int64)                            # (BUCKETS, 3)
        d2 = ((q[:, None, :] - d[None]) ** 2).sum(-1)            # (16, BUCKETS)
        j = d2.argmin(1)
        m = d2[np.arange(TEAM), j]
        better = m < best
        best = np.where(better, m, best)
        best_j[better] = np.stack([np.full(better.sum(), b), j[better]], 1)
    
    return best, best_j, visited, fetched


def run_frame(pc1, pc2, BUCKETS):
    p = prepare(pc1, pc2, BUCKETS)
    nt = len(p["teams"])
    d2 = np.empty((nt, TEAM), dtype=np.int64)
    idx = np.empty((nt, TEAM), dtype=np.int64)
    visited = np.empty(nt, dtype=np.int64)
    fetched = np.empty(nt, dtype=np.int64)
    
    for t in range(nt):
        best, bj, visited[t], fetched[t] = run_team(p["teams"][t], p["pc1_b"], p["lo"], p["hi"])
        d2[t] = best                                             # step 9
        idx[t] = p["pc1_idx"][bj[:, 0], bj[:, 1]]
    
    # back to original query order (padding copies dropped)
    out_d2 = np.empty(p["n_query"], dtype=np.int64)
    out_idx = np.empty(p["n_query"], dtype=np.int64)
    out_d2[p["pc2_idx"].ravel()] = d2.ravel()
    out_idx[p["pc2_idx"].ravel()] = idx.ravel()
    
    return out_d2, out_idx, visited, fetched, p


def run_pair(args):
    pair_idx, row, BUCKETS = args
    rf, qf = kd.load_pair(ROOT, row)
    pc1, pc2 = kd.quantize_cm(rf)[0], kd.quantize_cm(qf)[0]
    d2, idx, visited, fetched, p = run_frame(pc1, pc2, BUCKETS)
    
    # independent check: scipy KD-tree on the same quantized points
    dd, _ = cKDTree(pc1.astype(np.float64)).query(pc2.astype(np.float64), k=1)
    exp = np.rint(dd * dd).astype(np.int64)
    got_from_idx = ((pc2.astype(np.int64) - pc1[idx].astype(np.int64)) ** 2).sum(1)
    d2_bad = int(np.count_nonzero(d2 != exp))
    idx_bad = int(np.count_nonzero(got_from_idx != d2))
    nb, nt = len(p["lo"]), len(p["teams"])
    
    return dict(pair=pair_idx, bucket=BUCKETS, n_pc1=len(pc1), n_query=len(pc2), n_buckets=nb, n_teams=nt,
                box_checks=nt * nb,
                visited_mean=float(visited.mean()), visited_p99=float(np.percentile(visited, 99)),
                fetched_total=int(fetched.sum()), fetched_mean=float(fetched.mean()),
                fetched_p99=float(np.percentile(fetched, 99)), fetched_max=int(fetched.max()),
                dist_evals=int(fetched.sum()) * BUCKETS * TEAM, brute_evals=len(pc1) * len(pc2),
                sum_d2=int(d2.sum()), cloud_dist_cm=float(np.sqrt(float(d2.sum()))),
                d2_mismatch=d2_bad, idx_mismatch=idx_bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--bucket", type=int, nargs="+", default=[128])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "knn1_bucket_model.csv"))
    a = ap.parse_args()
    
    frames = kd.read_frames()[: a.pairs or None]
    jobs = [(i, r, BUCKETS) for BUCKETS in a.bucket for i, r in enumerate(frames)]
    
    with Pool(a.jobs) as pool:
        rows = list(pool.imap(run_pair, jobs))
    
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.normpath(a.out)}")


if __name__ == "__main__":
    main()
