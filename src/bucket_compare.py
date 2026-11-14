"""Compare two ways of grouping a KITTI frame into fixed-size buckets for exact 1-NN search.

Schemes (applied to refs = frame t-1 with bucket size B, and to queries = frame t with group size 16):
  morton : sort by 3-D Morton (Z-order) code, cut the sorted list every B points.
  median : recursive split at the median of the widest axis, cut points so every leaf has B points.

Metric (hardware independent): for each group of 16 queries, the number of buckets the group MUST
open to get an exact answer = buckets whose box is closer than the true nearest neighbour of at
least one of the 16 queries. This is a lower bound for any exact bucketed search using these boxes
(a real kernel opens a few more while it is still looking). Work = buckets_opened * B * 16 distance
evaluations, because all 16 lanes scan every opened bucket.

Coordinates are the study's int16 cm quantization (src/knn_data.py). Build times are numpy on the
x86 build host and only indicate relative cost; they are not PVA or loki timings.

usage: /usr/bin/python3 src/bucket_compare.py [--pairs N] [--out results/bucket_compare.csv]
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_data as kd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kitti_subset")
GROUP = 16
BUCKET_SIZES = (256, 512, 1024)


# ---------- orderings -------------------------------------------------------------------------
def _spread15(v):
    """Insert two zero bits between each of the low 15 bits of v (uint64)."""
    v = v & 0x7FFF
    v = (v | (v << 16)) & 0x0000FF0000FF
    v = (v | (v << 8)) & 0x00F00F00F00F
    v = (v | (v << 4)) & 0x0C30C30C30C3
    v = (v | (v << 2)) & 0x249249249249
    return v


def morton_order(p):
    u = (p.astype(np.int64) + kd.QMAX).astype(np.uint64)  # [0, 24000] fits 15 bits
    code = _spread15(u[:, 0]) | (_spread15(u[:, 1]) << np.uint64(1)) | (_spread15(u[:, 2]) << np.uint64(2))
    return np.argsort(code, kind="stable")


def median_order(p, leaf):
    """Index order such that consecutive chunks of `leaf` points are kd-tree leaves."""
    out = []
    stack = [np.arange(len(p))]
    while stack:
        idx = stack.pop()
        if len(idx) <= leaf:
            out.append(idx)
            continue
        sub = p[idx]
        axis = int(np.argmax(sub.max(0) - sub.min(0)))
        nleft = ((len(idx) // 2 + leaf - 1) // leaf) * leaf  # left side holds whole leaves
        part = np.argpartition(sub[:, axis], nleft - 1)
        stack.append(idx[part[nleft:]])
        stack.append(idx[part[:nleft]])  # popped first -> left before right
    return np.concatenate(out)


# ---------- metric ----------------------------------------------------------------------------
def boxes(sorted_pts, size):
    n = len(sorted_pts)
    nb = (n + size - 1) // size
    lo, hi = np.empty((nb, 3)), np.empty((nb, 3))
    for b in range(nb):
        chunk = sorted_pts[b * size:(b + 1) * size]
        lo[b], hi[b] = chunk.min(0), chunk.max(0)
    return lo, hi


def buckets_needed(q_sorted, dnn2_sorted, lo, hi):
    """Per group of 16 queries: number of buckets with box distance^2 < nn distance^2 for any lane."""
    ng = (len(q_sorted) + GROUP - 1) // GROUP
    counts = np.empty(ng, dtype=np.int64)
    step = 256  # groups per chunk to bound memory
    for g0 in range(0, ng, step):
        g1 = min(ng, g0 + step)
        q = q_sorted[g0 * GROUP:g1 * GROUP]
        d = dnn2_sorted[g0 * GROUP:g1 * GROUP]
        gap = np.maximum(np.maximum(lo[None] - q[:, None], q[:, None] - hi[None]), 0.0)  # (nq, nb, 3)
        lb2 = (gap * gap).sum(-1)
        need = lb2 < d[:, None]
        gid = np.arange(len(q)) // GROUP
        per_group = np.zeros((g1 - g0, lo.shape[0]), dtype=bool)
        np.logical_or.at(per_group, gid, need)
        counts[g0:g1] = per_group.sum(1)
    return counts


def run_pair(args):
    pair_idx, row = args
    ref_f, qry_f = kd.load_pair(ROOT, row)
    ref = kd.quantize_cm(ref_f)[0].astype(np.float64)
    qry = kd.quantize_cm(qry_f)[0].astype(np.float64)
    dnn, _ = cKDTree(ref).query(qry, k=1)
    dnn2 = np.rint(dnn * dnn)  # exact integers on quantized inputs (distances^2 < 2^31)
    rows = []
    for scheme in ("morton", "median"):
        t0 = time.perf_counter()
        qo = morton_order(qry) if scheme == "morton" else median_order(qry, GROUP)
        t_q = time.perf_counter() - t0
        for B in BUCKET_SIZES:
            t0 = time.perf_counter()
            ro = morton_order(ref) if scheme == "morton" else median_order(ref, B)
            lo, hi = boxes(ref[ro], B)
            t_r = time.perf_counter() - t0
            cnt = buckets_needed(qry[qo], dnn2[qo], lo, hi)
            evals = int(cnt.sum()) * B * GROUP
            rows.append(dict(pair=pair_idx, scheme=scheme, bucket=B, n_ref=len(ref), n_query=len(qry),
                             n_buckets=len(lo), n_groups=len(cnt),
                             buckets_p50=float(np.percentile(cnt, 50)),
                             buckets_p99=float(np.percentile(cnt, 99)),
                             buckets_max=int(cnt.max()), buckets_mean=float(cnt.mean()),
                             evals=evals, brute_evals=len(ref) * len(qry),
                             build_ref_s=t_r, build_query_s=t_q))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=0, help="0 = all pairs in frames.txt")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "..", "results", "bucket_compare.csv"))
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    frames = kd.read_frames()
    if a.pairs:
        frames = frames[:a.pairs]
    with Pool(a.jobs) as pool:
        rows = [r for rs in pool.imap(run_pair, list(enumerate(frames))) for r in rs]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.normpath(a.out)}")


if __name__ == "__main__":
    main()
