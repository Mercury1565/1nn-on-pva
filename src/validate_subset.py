import sys, os, numpy as np, datetime as dt
from knn_data import *

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kitti_subset")
rows = read_frames()
npts, clamp, tot, bad, gaps = [], 0, 0, 0, []

def ts(date, drive, i):
    with open(os.path.join(root, date, drive, "velodyne_points", "timestamps.txt")) as f:
        line = f.readlines()[i].strip()
    return dt.datetime.strptime(line[:26], "%Y-%m-%d %H:%M:%S.%f")

for row in rows:
    for fr in row[2:]:
        p = bin_path(root, row[0], row[1], fr); tot += os.path.getsize(p)
        a = load_bin(p)
        if not np.isfinite(a).all(): bad += 1
        npts.append(len(a)); clamp += quantize_cm(a[:, :3])[1]
    
    if os.path.exists(os.path.join(root, row[0], row[1], "velodyne_points", "timestamps.txt")):  # full dataset only
        gaps.append((ts(row[0], row[1], int(row[3])) - ts(row[0], row[1], int(row[2]))).total_seconds())

npts = np.array(npts); gaps = np.array(gaps)
print(f"files={len(npts)} bytes={tot/1e6:.1f} MB nonfinite_files={bad} clamped_coords={clamp}")
print(f"points/frame min={npts.min()} p50={int(np.median(npts))} max={npts.max()}")
if len(gaps): print(f"pair dt s: min={gaps.min():.4f} p50={np.median(gaps):.4f} max={gaps.max():.4f}  >0.15s: {(gaps>0.15).sum()}")
