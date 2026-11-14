"""Frame-to-frame distance on the PVA from point clouds already in memory.

    python3 examples/example_memory.py <frame_t-1.bin> <frame_t.bin>

The files are only read to fill memory, the way a pipeline would hand frames over.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import knn1_distance as k1  # noqa: E402


def pva_frame_distance(pva, frame_t, frame_t1):
    """Nearest frame t-1 point for every frame t point.

    frame_t, frame_t1: float32 arrays (n, 4) x, y, z, reflectance in metres, or (n, 3).
    From a torch tensor on the GPU: tensor.cpu().numpy().
    Returns per-point squared distance (cm^2), matched index into frame_t1, cloud distance (cm).
    """
    return pva.nearest(frame_t, frame_t1)  # pc_queries = frame t, pc_ref = frame t-1


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)

    # 1. Frames in memory (here from KITTI files).
    frame_t1 = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 4)
    frame_t = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, 4)
    print(f"frame t-1: {len(frame_t1)} points, frame t: {len(frame_t)} points")

    # 2. One PVA context, created once and reused for every frame.
    pva = k1.Knn1()

    # 3. Distance between the two frames.
    t0 = time.perf_counter()
    dist2, idx, cloud_cm = pva_frame_distance(pva, frame_t, frame_t1)
    ms = (time.perf_counter() - t0) * 1000

    # 4. Use the answers.
    d_cm = np.sqrt(dist2.astype(np.float64))
    print(f"cloud distance sqrt(sum d2) = {cloud_cm / 100:.3f} m")
    print(f"nearest distance per point: median {np.median(d_cm):.1f} cm, "
          f"99th percentile {np.percentile(d_cm, 99):.1f} cm, max {d_cm.max():.1f} cm")

    # Motion vector of each frame t point = its position minus its match in frame t-1.
    motion = frame_t[:, :3] - frame_t1[idx, :3]
    print(f"point 0 moved by {motion[0]} m (matched frame t-1 point {idx[0]})")
    print(f"call took {ms:.0f} ms wall clock (includes conversion to cm; not a PVA measurement)")


if __name__ == "__main__":
    main()
