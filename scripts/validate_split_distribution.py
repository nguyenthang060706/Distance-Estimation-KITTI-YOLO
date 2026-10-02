"""KS test Car-only for split quality (D1 decision gate)."""
import sys, numpy as np
from scipy import stats
sys.path.insert(0, '.')
from src.utils.kitti_loader import KITTILoader
from src.utils.split_builder import load_splits

loader = KITTILoader('data/kitti')
splits = load_splits('splits')

# Collect depths: Car-only, Hard-filtered
split_depths = {}
for name in ['B', 'C', 'T']:
    depths = []
    for fid in splits[name]:
        frame = loader.load_frame(fid)
        for obj in frame.objects:
            if obj.obj_class == 'Car' and obj.passes_hard_filter():
                depths.append(obj.depth)
    split_depths[name] = np.array(depths)
    print(f'{name}: {len(depths)} Car (Hard)')

print()
print('=== KS Test: Car-only, Hard-filtered ===')
print(f'{"Pair":>8s} {"KS stat":>8s} {"n1":>6s} {"n2":>6s}')
print('-' * 35)
threshold = 0.07
all_ok = True
for a, b in [('C','T'), ('B','T'), ('B','C')]:
    ks_stat, _ = stats.ks_2samp(split_depths[a], split_depths[b])
    ok = 'OK' if ks_stat <= threshold else 'HIGH'
    if ks_stat > threshold:
        all_ok = False
    print(f'{a+" vs "+b:>8s} {ks_stat:>8.4f} {len(split_depths[a]):>6d} {len(split_depths[b]):>6d}  {ok}')

print()
# Depth summary Car-only
print('=== Depth Summary (Car-only, Hard) ===')
print(f'{"Split":>6s} {"n":>6s} {"mean":>7s} {"median":>7s} {"std":>7s}')
for name in ['B', 'C', 'T']:
    d = split_depths[name]
    print(f'{name:>6s} {len(d):>6d} {d.mean():>7.1f} {np.median(d):>7.1f} {d.std():>7.1f}')

# Count per distance bin, Car-only
print()
print('=== Car-only per bin ===')
bins = [0, 10, 20, 30, 50, float('inf')]
labels = ['0-10m', '10-20m', '20-30m', '30-50m', '>50m']
print(f'{"Split":>6s}', ''.join(f'{l:>8s}' for l in labels))
for name in ['B', 'C', 'T']:
    d = split_depths[name]
    row = f'{name:>6s}'
    for i in range(len(bins)-1):
        cnt = int(((d >= bins[i]) & (d < bins[i+1])).sum())
        flag = '*' if cnt < 100 else ' '
        row += f'{cnt:>7d}{flag}'
    print(row)

print()
if all_ok:
    print(f'ALL pairs KS stat <= {threshold}: FREEZE split.')
else:
    print(f'Some pairs KS stat > {threshold}: consider seed search.')
