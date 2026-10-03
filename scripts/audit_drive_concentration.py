# scripts/audit_drive_concentration.py
import sys
import numpy as np
from collections import Counter
sys.path.insert(0, ".")
from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_splits

loader = KITTILoader("data/kitti")
for name, fids in load_splits("splits").items():
    per_drive = Counter()
    for fid in fids:
        fr = loader.load_frame(fid)
        per_drive[fr.drive] += sum(o.obj_class == "Car" and o.passes_hard_filter() for o in fr.objects)
    with_car = {d: n for d, n in per_drive.items() if n > 0}
    total = sum(with_car.values())
    top1 = max(with_car.values()) / total if total else float("nan")
    shares = np.array(list(with_car.values())) / total if total else np.array([1.0])
    n_eff = 1.0 / np.sum(shares ** 2) if total else float("nan")
    top5 = sorted(with_car.items(), key=lambda kv: -kv[1])[:5]
    print(f"{name}: drives={len(per_drive)} with_CarHard={len(with_car)} n={total} top1_share={top1:.2f}")
    print(f"  n_eff={n_eff:.1f} top5={top5}")
