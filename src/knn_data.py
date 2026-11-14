"""Shared data loading, subsampling and quantization for the PVA KNN study.

Used unchanged by every backend (GPU, CPU, PVA test-vector generation) so all of them see
exactly the same points.

Conventions
-----------
* Frame pair (date, drive, ref_frame, query_frame) from results/frames.txt:
  references = frame t-1, queries = frame t.
* Subsampling to N points is a deterministic integer-hash ordering (splitmix64), not
  numpy.random, so it is bit-identical across numpy versions and machines (x86 container,
  aarch64 loki). Selected points keep their original file order.
  N >= frame size  ->  the whole frame is used ("full").
* PVA quantization: clamp x, y, z to [-120 m, +120 m], then round-half-to-even to int16
  centimetres (range [-12000, 12000]). Max squared distance 3 * 24000^2 = 1.728e9 < 2^31 - 1,
  so squared distances fit exactly in int32.
* Exact reference KNN: squared Euclidean distance in int64 on the quantized inputs,
  ordered by (distance, reference index) ascending -> lowest index wins ties.
"""
import os
import numpy as np

CLAMP_M = 120.0
SCALE = 100.0  # metres -> centimetres
QMAX = int(round(CLAMP_M * SCALE))  # 12000

DEFAULT_FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "frames.txt")

ROLE_REF, ROLE_QUERY = 0, 1


def read_frames(path=DEFAULT_FRAMES):
    """Return list of (date, drive, ref_frame, query_frame) strings."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(tuple(line.split()))
    return rows


def bin_path(root, date, drive, frame):
    return os.path.join(root, date, drive, "velodyne_points", "data", f"{frame}.bin")


def load_bin(path):
    """Load a KITTI Velodyne scan -> float32 (n, 4) [x, y, z, reflectance]."""
    a = np.fromfile(path, dtype=np.float32)
    if a.size % 4:
        raise ValueError(f"{path}: size {a.size} floats is not a multiple of 4")
    return a.reshape(-1, 4)


def load_pair(root, row):
    """Return (ref_xyz, query_xyz) float32 (n, 3) for one frames.txt row."""
    date, drive, rf, qf = row
    ref = load_bin(bin_path(root, date, drive, rf))[:, :3]
    qry = load_bin(bin_path(root, date, drive, qf))[:, :3]
    return np.ascontiguousarray(ref), np.ascontiguousarray(qry)


def _splitmix64(x):
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def subsample_idx(n_total, n, seed, pair_idx, role):
    """Deterministic subset of n indices out of n_total (sorted), or all if n >= n_total."""
    if n is None or n >= n_total:
        return np.arange(n_total)
    salt = np.uint64((seed * 1000003 + pair_idx * 7919 + role * 104729 + n) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        keys = _splitmix64(np.arange(n_total, dtype=np.uint64) ^ (salt * np.uint64(0x9E3779B97F4A7C15)))
    sel = np.argpartition(keys, n - 1)[:n]
    return np.sort(sel)


def subsample(xyz, n, seed, pair_idx, role):
    return xyz[subsample_idx(len(xyz), n, seed, pair_idx, role)]


def quantize_cm(xyz):
    """float32 metres -> int16 centimetres after clamping to +/-120 m. Returns (q, n_clamped)."""
    clamped = np.clip(xyz, -CLAMP_M, CLAMP_M)
    n_clamped = int(np.count_nonzero(clamped != xyz))
    q = np.rint(clamped.astype(np.float64) * SCALE).astype(np.int16)
    return q, n_clamped


def exact_knn_int(q_xyz, r_xyz, k, chunk=256):
    """Exact brute-force KNN on int16 inputs.

    Returns (idx int32 (nq, k), dist2 int32 (nq, k)), ordered by (dist2, ref index).
    """
    q = q_xyz.astype(np.int64)
    r = r_xyz.astype(np.int64)
    nq = len(q)
    out_i = np.empty((nq, k), np.int32)
    out_d = np.empty((nq, k), np.int32)
    ridx = np.arange(len(r), dtype=np.int64)
    for s in range(0, nq, chunk):
        qq = q[s:s + chunk]
        d = ((qq[:, None, :] - r[None, :, :]) ** 2).sum(-1)  # int64 exact
        assert d.max() < 2**31
        key = d * len(r) + ridx[None, :]  # unique keys -> (dist, index) order
        part = np.argpartition(key, k - 1, axis=1)[:, :k]
        pk = np.take_along_axis(key, part, 1)
        order = np.argsort(pk, axis=1)
        sel = np.take_along_axis(part, order, 1)
        out_i[s:s + chunk] = sel
        out_d[s:s + chunk] = np.take_along_axis(d, sel, 1)
    return out_i, out_d
