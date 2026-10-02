"""KS test and chi-square test for split quality validation."""
import sys, numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, '.')
from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_splits

loader = KITTILoader('data/kitti')
splits = load_splits('splits')

# Collect depths per split (Hard-filtered)
split_depths = {}
split_class_counts = {}
for name in ['A', 'V', 'B', 'C', 'T']:
    depths = []
    class_counts = defaultdict(int)
    for fid in splits[name]:
        frame = loader.load_frame(fid)
        for obj in frame.objects:
            if obj.passes_hard_filter():
                depths.append(obj.depth)
                class_counts[obj.obj_class] += 1
    split_depths[name] = np.array(depths)
    split_class_counts[name] = dict(class_counts)

# KS test: compare each split's depth distribution against A
print('=== KS Test: Depth distribution vs A (Hard-filtered) ===')
print(f'{"Split":>6s} {"n":>6s} {"KS stat":>8s} {"p-value":>10s} {"Result":>10s}')
print('-' * 45)
for name in ['V', 'B', 'C', 'T']:
    ks_stat, p_val = stats.ks_2samp(split_depths['A'], split_depths[name])
    result = 'OK' if p_val > 0.05 else 'DIFF!'
    print(f'{name:>6s} {len(split_depths[name]):>6d} {ks_stat:>8.4f} {p_val:>10.4f} {result:>10s}')

# Pairwise B/C/T (should be exchangeable)
print()
print('=== KS Test: Pairwise B/C/T (should be similar) ===')
for a, b in [('B','C'), ('B','T'), ('C','T')]:
    ks_stat, p_val = stats.ks_2samp(split_depths[a], split_depths[b])
    result = 'OK' if p_val > 0.05 else 'DIFF!'
    print(f'  {a} vs {b}: KS={ks_stat:.4f}, p={p_val:.4f}  {result}')

# Class distribution chi-square test
print()
print('=== Chi-square Test: Class distribution ===')
all_classes = ['Car', 'Van', 'Truck']
total_counts = defaultdict(int)
for name in split_class_counts:
    for cls, cnt in split_class_counts[name].items():
        total_counts[cls] += cnt
total = sum(total_counts.values())
expected_props = {cls: total_counts[cls]/total for cls in all_classes}
print(f'Overall proportions: Car={expected_props["Car"]:.3f}, Van={expected_props["Van"]:.3f}, Truck={expected_props["Truck"]:.3f}')
print()
for name in ['A', 'V', 'B', 'C', 'T']:
    n = sum(split_class_counts[name].get(c,0) for c in all_classes)
    observed = [split_class_counts[name].get(c,0) for c in all_classes]
    expected = [expected_props[c] * n for c in all_classes]
    chi2, p_val = stats.chisquare(observed, expected)
    result = 'OK' if p_val > 0.05 else 'DIFF!'
    props = ' '.join(f'{c}={split_class_counts[name].get(c,0)/n:.3f}' for c in all_classes)
    print(f'  {name}: chi2={chi2:7.1f}, p={p_val:.4f}  {result}  [{props}]')

# Depth summary stats per split
print()
print('=== Depth Summary Stats ===')
print(f'{"Split":>6s} {"n":>6s} {"mean":>7s} {"median":>7s} {"std":>7s} {"min":>6s} {"max":>7s}')
print('-' * 55)
for name in ['A', 'V', 'B', 'C', 'T']:
    d = split_depths[name]
    print(f'{name:>6s} {len(d):>6d} {d.mean():>7.1f} {np.median(d):>7.1f} {d.std():>7.1f} {d.min():>6.1f} {d.max():>7.1f}')
