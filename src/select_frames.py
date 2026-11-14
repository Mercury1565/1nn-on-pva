#!/usr/bin/env python3
"""Pick a fixed benchmark subset of consecutive KITTI raw Velodyne frame pairs.

Pair = (frame t-1 = references, frame t = queries) from the same drive.
Stratified: equal number of pairs per date; within a date, drives are visited in a
seeded random round-robin so pairs spread over as many drives as possible.
Pairs never share a frame. Output: results/frames.txt (one pair per line).
"""
import argparse, os, random, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/z/home/kidus/drive_data/kitti_full")  # full KITTI raw on deadshot (read only)
ap.add_argument("--pairs", type=int, default=200)
ap.add_argument("--seed", type=int, default=20261112)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "frames.txt"))
a = ap.parse_args()

rng = random.Random(a.seed)
dates = sorted(d for d in os.listdir(a.root) if re.fullmatch(r"\d{4}_\d{2}_\d{2}", d))
drives = {}  # date -> {drive: sorted frame ids}
for d in dates:
    for dr in sorted(os.listdir(os.path.join(a.root, d))):
        p = os.path.join(a.root, d, dr, "velodyne_points", "data")
        
        if not dr.endswith("_sync") or not os.path.isdir(p):
            continue
       
        ids = sorted(int(f[:-4]) for f in os.listdir(p) if f.endswith(".bin"))
        s = set(ids)
        valid_t = [t for t in ids if t - 1 in s]          # t-1 must exist on disk
        
        if valid_t:
            drives.setdefault(d, {})[dr] = valid_t

per_date = [a.pairs // len(drives)] * len(drives)
for i in range(a.pairs - sum(per_date)):
    per_date[i] += 1

rows = []
for (d, drs), n in zip(sorted(drives.items()), per_date):
    names = sorted(drs); rng.shuffle(names)
    used = {dr: set() for dr in names}
    got, i, stall = 0, 0, 0
   
    while got < n:
        dr = names[i % len(names)]; i += 1
        cand = [t for t in drs[dr] if not ({t, t - 1} & used[dr])]
        
        if not cand:
            stall += 1
            if stall > len(names): sys.exit(f"not enough frames in {d}")
            continue
        
        stall = 0
        t = rng.choice(cand)
        used[dr] |= {t - 1, t}
        rows.append((d, dr, t - 1, t)); got += 1

rows.sort()
with open(a.out, "w") as f:
    f.write(f"# KITTI raw velodyne frame pairs. root={a.root} seed={a.seed} pairs={len(rows)}\n")
    f.write("# refs = frame t-1, queries = frame t. columns: date drive ref_frame query_frame\n")
    for d, dr, r, q in rows:
        f.write(f"{d} {dr} {r:010d} {q:010d}\n")
        
print(f"wrote {len(rows)} pairs from {len({(r[0], r[1]) for r in rows})} drives to {a.out}")
for d in sorted(drives):
    print(d, sum(1 for r in rows if r[0] == d), "pairs,", len({r[1] for r in rows if r[0] == d}), "drives")
