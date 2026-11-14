"""Cloud distance between two KITTI scans through the Python wrapper.

    python3 examples/example.py <pc_ref.bin> <pc_queries.bin>
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import knn1_distance as k1  # noqa: E402

pc_ref = k1.load_kitti_bin(sys.argv[1])        # float32 (n, 3), metres
pc_queries = k1.load_kitti_bin(sys.argv[2])

pva = k1.Knn1()                                # create once, reuse for every frame pair
t0 = time.time()
dist2, idx, cloud_dist_cm = pva.nearest(pc_queries, pc_ref)
ms = (time.time() - t0) * 1000

d = np.sqrt(dist2.astype(np.float64))          # per-point nearest distance, cm
print(f"pc_ref {len(pc_ref)} points, pc_queries {len(pc_queries)} points")
print(f"cloud distance sqrt(sum d2) = {cloud_dist_cm / 100:.3f} m")
print(f"nearest distance per point: median {np.median(d):.1f} cm, 99th percentile {np.percentile(d, 99):.1f} cm, "
      f"max {d.max():.1f} cm")
print(f"query point 0 -> ref point {idx[0]} at {d[0]:.1f} cm")
print(f"buckets fetched {pva.last_buckets_fetched()}, call took {ms:.0f} ms (wall clock)")
