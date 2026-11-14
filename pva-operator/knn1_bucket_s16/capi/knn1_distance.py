"""Python access to the PVA 1-NN kernel through libknn1_distance.so (ctypes + numpy).

    import knn1_distance as k1
    pva = k1.Knn1()                                   # once
    pc_ref     = k1.load_kitti_bin("frame_t-1.bin")   # float32 (n, 3) metres
    pc_queries = k1.load_kitti_bin("frame_t.bin")
    dist2, idx, cloud_dist_cm = pva.nearest(pc_queries, pc_ref)

dist2[i] is the squared distance (cm^2, int32) from query point i to its nearest point of
pc_ref, idx[i] the index of that point, cloud_dist_cm = sqrt(sum(dist2)). Inputs are float
metres (n, 3) or (n, 4); they are clipped to +/-120 m and rounded to whole centimetres.
"""
import ctypes
import os

import numpy as np

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "libknn1_distance.so")
_lib = ctypes.CDLL(_LIB_PATH)

_i16p = np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS")
_i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")

_lib.knn1_create.restype = ctypes.c_void_p
_lib.knn1_destroy.argtypes = [ctypes.c_void_p]
_lib.knn1_nearest.restype = ctypes.c_int
_lib.knn1_nearest.argtypes = [ctypes.c_void_p, _i16p, ctypes.c_int32, _i16p, ctypes.c_int32, _i32p, _i32p,
                              ctypes.POINTER(ctypes.c_int64)]
_lib.knn1_quantize_cm.restype = ctypes.c_int32
_lib.knn1_quantize_cm.argtypes = [_f32p, ctypes.c_int32, ctypes.c_int32, _i16p]
_lib.knn1_last_buckets_fetched.restype = ctypes.c_int64
_lib.knn1_last_buckets_fetched.argtypes = [ctypes.c_void_p]
_lib.knn1_error_string.restype = ctypes.c_char_p
_lib.knn1_error_string.argtypes = [ctypes.c_int]


def load_kitti_bin(path):
    """KITTI velodyne scan -> float32 (n, 3) x, y, z in metres."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def quantize_cm(xyz):
    """float metres (n, 3) or (n, 4) -> planar int16 centimetres (3 * n values)."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n, stride = xyz.shape
    planar = np.empty(3 * n, dtype=np.int16)
    if _lib.knn1_quantize_cm(xyz.ravel(), n, stride, planar) < 0:
        raise ValueError("point array must be (n, 3) or (n, 4) with n >= 1")
    return planar


class Knn1:
    """One PVA context. Not thread-safe: use one instance per thread."""

    def __init__(self):
        self._ctx = _lib.knn1_create()
        if not self._ctx:
            raise RuntimeError("knn1_create failed (is this a Thor with the PVA, and are you in the 'video' group?)")

    def __del__(self):
        if getattr(self, "_ctx", None):
            _lib.knn1_destroy(self._ctx)
            self._ctx = None

    def nearest(self, pc_queries, pc_ref):
        """Float metres in; returns (dist2 int32 cm^2, idx int32, cloud distance in cm)."""
        q, r = quantize_cm(pc_queries), quantize_cm(pc_ref)
        nq, nr = len(q) // 3, len(r) // 3
        dist2 = np.empty(nq, dtype=np.int32)
        idx = np.empty(nq, dtype=np.int32)
        s = ctypes.c_int64()
        rc = _lib.knn1_nearest(self._ctx, q, nq, r, nr, dist2, idx, ctypes.byref(s))
        if rc != 0:
            raise RuntimeError(f"knn1_nearest failed ({rc}): {_lib.knn1_error_string(rc).decode()}")
        return dist2, idx, float(np.sqrt(s.value))

    def last_buckets_fetched(self):
        return _lib.knn1_last_buckets_fetched(self._ctx)
