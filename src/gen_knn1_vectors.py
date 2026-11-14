"""Test vectors for knn1_bucket_s16 (exact 1-NN, int16 cm, runtime point counts).

Per case directory testdata/knn1_bucket/<case>/:
  queries.s16  int16 planar [3][nq]  (x plane, y plane, z plane)
  refs.s16     int16 planar [3][nr]
  dist2.s32    int32 [nq]  exact nearest squared distance (cm^2)       <- the check
  idx.s32      int32 [nq]  one valid nearest ref index (informational; ties allowed)
manifest.json: sizes, SHA-256 of every file, sum of dist2, tie statistics, description.

Expected values come from scipy's cKDTree (exact search) on the quantized integer coordinates,
independently of the bucket model (src/knn1_bucket_model.py) and of the PVA kernel. Squared distances
are recomputed in int64 from the returned index, so they are exact integers.

usage: /usr/bin/python3 src/gen_knn1_vectors.py [--out testdata/knn1_bucket]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import knn_data as kd  # noqa: E402

ROOT = os.path.join(HERE, "..", "data", "kitti_subset")
SEED = 20261112
MAXN = 131072


def rng(tag):
    return np.random.default_rng([SEED, int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")])


def uniform(tag, n, lim=kd.QMAX):
    return rng(tag).integers(-lim, lim + 1, size=(n, 3)).astype(np.int16)


def kitti_pairs():
    """First frame pair of each of the first four dates in frames.txt (deterministic)."""
    seen, out = set(), []
    for i, row in enumerate(kd.read_frames()):
        if row[0] not in seen:
            seen.add(row[0])
            out.append((i, row))
    return out[:4]


def cases():
    kp = kitti_pairs()
    k = {}
    for i, row in kp:
        r, q = kd.load_pair(ROOT, row)
        k[i] = (kd.quantize_cm(q)[0], kd.quantize_cm(r)[0], row)
    (i0, _), (i1, _), (i2, _), (i3, _) = kp
    c = []
    for tag, i in zip("abcd", (i0, i1, i2, i3)):
        q, r, row = k[i]
        c.append((f"kitti_{tag}", q, r, f"KITTI full frames, pair {i} ({row[1]} {row[2]}->{row[3]}): queries frame t, refs frame t-1"))
    q, r, row = k[i0]
    c.append(("kitti_a_swap", r, q, "kitti_a with roles swapped (reverse direction of a Chamfer distance)"))
    q16 = q[np.sort(rng("q16").permutation(len(q))[:16384])]
    r16 = r[np.sort(rng("r16").permutation(len(r))[:16384])]
    c.append(("kitti_16k", q16, r16, "kitti_a randomly subsampled to 16384 queries and 16384 refs"))
    c.append(("kitti_unequal", q16[:4096], r, "4096 queries (subsample of kitti_a) against the full ref frame"))
    c.append(("kitti_self", q16, q16.copy(), "queries and refs have identical coordinates (separate buffers): every dist2 is 0"))
    c.append(("max_131072", uniform("maxq", MAXN), uniform("maxr", MAXN), "maximum size: 131072 x 131072 uniform random in [-12000, 12000]"))
    for nq, nr in ((1, 1), (1, 512), (16, 1), (15, 511), (16, 512), (17, 513), (513, 17), (1000, 3000)):
        c.append((f"small_{nq}x{nr}", uniform(f"sq{nq}_{nr}", nq, 3000), uniform(f"sr{nq}_{nr}", nr, 3000),
                  f"{nq} queries x {nr} refs, uniform in [-3000, 3000] (partial bucket / partial team / tails)"))
    far = np.full((5, 3), kd.QMAX, dtype=np.int16)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.int16) * kd.QMAX
    c.append(("extremes", np.repeat(corners, 4, 0), far, "queries on the 8 corners of the +/-120 m cube, refs at +corner: max dist2 = 1728000000"))
    g = rng("dup")
    grid_q = (g.integers(-2, 3, size=(4000, 3)) * 100).astype(np.int16)
    grid_r = (g.integers(-2, 3, size=(6000, 3)) * 100).astype(np.int16)
    c.append(("duplicates_grid", grid_q, grid_r, "points on a 5x5x5 grid with many duplicates: heavy exact ties"))
    c.append(("all_refs_identical", uniform("ariq", 1000), np.tile(np.array([[123, -456, 78]], np.int16), (2000, 1)),
              "2000 identical refs: every ref is a valid answer for every query"))
    line = np.zeros((3000, 3), np.int16); line[:, 0] = rng("line").integers(-12000, 12001, 3000)
    lq = np.zeros((1000, 3), np.int16); lq[:, 0] = rng("lineq").integers(-12000, 12001, 1000)
    c.append(("degenerate_line", lq, line, "all points on the x axis (zero-width y and z)"))
    plane_r = uniform("planer", 5000); plane_r[:, 2] = -173
    plane_q = uniform("planeq", 2000); plane_q[:, 2] = -173
    c.append(("degenerate_plane", plane_q, plane_r, "all points on the plane z = -173 cm"))
    cl = rng("clusters")
    rcl = np.concatenate([cl.integers(-300, 301, (3000, 3)) - 11000, cl.integers(-300, 301, (3000, 3)) + 11000]).astype(np.int16)
    qcl = np.concatenate([cl.integers(-300, 301, (500, 3)) - 11000, cl.integers(-300, 301, (500, 3)) + 11000,
                          cl.integers(-12000, 12001, (200, 3))]).astype(np.int16)
    c.append(("far_clusters", qcl, rcl, "two dense ref clusters at opposite corners; some queries in the empty space between"))
    return c


def expected(q, r):
    t = cKDTree(r.astype(np.float64))
    _, idx = t.query(q.astype(np.float64), k=1)
    d2 = ((q.astype(np.int64) - r[idx].astype(np.int64)) ** 2).sum(1)
    ties = 0
    if len(r) >= 2:
        _, i2 = t.query(q.astype(np.float64), k=2)
        d2b = ((q.astype(np.int64) - r[i2[:, 1]].astype(np.int64)) ** 2).sum(1)
        ties = int(np.count_nonzero(d2b == d2))
    return d2, idx.astype(np.int64), ties


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "testdata", "knn1_bucket"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = {"generator": "src/gen_knn1_vectors.py", "seed": SEED, "max_points": MAXN,
           "expected_source": "scipy cKDTree exact 1-NN on quantized int16 coordinates, d2 recomputed in int64",
           "cases": []}
    for name, q, r, desc in cases():
        assert 1 <= len(q) <= MAXN and 1 <= len(r) <= MAXN
        assert np.abs(q).max() <= kd.QMAX and np.abs(r).max() <= kd.QMAX
        d2, idx, ties = expected(q, r)
        assert d2.max() <= 3 * (2 * kd.QMAX) ** 2
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        np.ascontiguousarray(q.T).astype("<i2").tofile(os.path.join(d, "queries.s16"))
        np.ascontiguousarray(r.T).astype("<i2").tofile(os.path.join(d, "refs.s16"))
        d2.astype("<i4").tofile(os.path.join(d, "dist2.s32"))
        idx.astype("<i4").tofile(os.path.join(d, "idx.s32"))
        files = ("queries.s16", "refs.s16", "dist2.s32", "idx.s32")
        man["cases"].append({"case": name, "n_query": int(len(q)), "n_ref": int(len(r)), "description": desc,
                             "sum_dist2": int(d2.sum()), "min_dist2": int(d2.min()), "max_dist2": int(d2.max()),
                             "queries_with_tied_nearest": ties,
                             "sha256": {f: sha(os.path.join(d, f)) for f in files}})
        print(f"{name:22} nq={len(q):6} nr={len(r):6} sum_d2={int(d2.sum())} ties={ties}")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
