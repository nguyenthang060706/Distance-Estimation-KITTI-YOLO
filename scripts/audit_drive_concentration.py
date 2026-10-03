# scripts/audit_drive_concentration.py
import sys
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
    print(f"{name}: drives={len(per_drive)} with_CarHard={len(with_car)} n={total} top1_share={top1:.2f}")
