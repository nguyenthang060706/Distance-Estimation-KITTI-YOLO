# Đánh giá Độ phủ có điều kiện (Conditional Coverage) — T08

> Trung bình độ phủ có điều kiện qua 20 lần chia lại eval out-of-sample.
> Các nhóm: Ẑ dự đoán, Z thật (chẩn đoán), Truncated, Occluded, Chạm biên, Góc θ (D19), Difficulty, Fallback (D51).

## Detector `yolo11s_640`

### Dải cự ly theo Ẑ dự đoán (Prospective Bins)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `0-10m` | 114.5 | 65.57% | 68.04% | 74.88% |
| `10-20m` | 349.1 | 90.32% | 89.93% | 87.49% |
| `20-30m` | 454.9 | 88.88% | 88.52% | 85.98% |
| `>=30m` | 479.4 | 83.34% | 82.56% | 82.65% |

### Dải cự ly theo Z thật (Retrospective Bins — Chẩn đoán)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `0-10m` | 113.8 | 65.49% | 67.98% | 73.68% |
| `10-20m` | 356.4 | 90.68% | 89.68% | 87.74% |
| `20-30m` | 459.1 | 87.90% | 87.84% | 85.57% |
| `30-50m` | 467.1 | 83.80% | 83.34% | 83.26% |
| `>50m` | 1.5 | 63.33% | 80.00% | 63.33% |

### Mức độ cắt biên (Truncated)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `non_truncated` | 1315.0 | 86.80% | 85.85% | 84.92% |
| `truncated` | 82.8 | 66.98% | 74.81% | 79.32% |

### Mức độ che khuất (Occluded)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `occ_0` | 868.2 | 85.97% | 82.71% | 82.28% |
| `occ_1` | 408.8 | 84.48% | 88.76% | 87.82% |
| `occ_2` | 121.0 | 83.59% | 88.31% | 87.56% |

### Chạm biên ảnh (Touch Edge)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `no_edge_touch` | 1296.3 | 87.96% | 86.92% | 85.92% |
| `touches_edge` | 101.6 | 59.17% | 66.72% | 71.13% |

### Góc hướng quan sát θ (Decision D19)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `side_0_30deg` | 128.2 | 72.84% | 81.44% | 80.09% |
| `diagonal_30_60deg` | 138.9 | 79.44% | 82.57% | 81.64% |
| `front_rear_60_90deg` | 1130.8 | 86.69% | 84.51% | 84.03% |

### Mức độ khó KITTI (Difficulty)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `Easy` | 533.9 | 86.39% | 83.15% | 81.62% |
| `Moderate` | 715.6 | 85.31% | 85.81% | 85.54% |
| `Hard` | 148.4 | 79.32% | 85.21% | 85.69% |

### Nhóm Fallback Pattern 000 (Decision D51)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `normal_cues` | 1387.0 | 85.95% | 85.40% | 84.67% |
| `fallback_pattern_000` | 10.8 | 42.25% | 58.13% | 67.76% |

## Detector `yolov8s_640`

### Dải cự ly theo Ẑ dự đoán (Prospective Bins)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `0-10m` | 119.0 | 66.57% | 68.93% | 73.57% |
| `10-20m` | 348.9 | 92.58% | 91.95% | 89.24% |
| `20-30m` | 447.5 | 91.32% | 90.83% | 87.19% |
| `>=30m` | 430.1 | 85.00% | 81.77% | 81.61% |

### Dải cự ly theo Z thật (Retrospective Bins — Chẩn đoán)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `0-10m` | 115.5 | 66.61% | 68.80% | 72.64% |
| `10-20m` | 357.4 | 93.06% | 91.79% | 89.30% |
| `20-30m` | 456.1 | 90.15% | 89.49% | 86.14% |
| `30-50m` | 416.2 | 85.11% | 82.52% | 82.02% |
| `>50m` | 0.4 | 100.00% | 100.00% | 100.00% |

### Mức độ cắt biên (Truncated)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `non_truncated` | 1263.7 | 88.56% | 86.81% | 84.90% |
| `truncated` | 81.8 | 70.16% | 76.37% | 79.11% |

### Mức độ che khuất (Occluded)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `occ_0` | 852.8 | 87.11% | 83.79% | 81.99% |
| `occ_1` | 385.1 | 87.11% | 89.61% | 88.40% |
| `occ_2` | 107.6 | 88.47% | 90.00% | 88.00% |

### Chạm biên ảnh (Touch Edge)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `no_edge_touch` | 1246.6 | 89.89% | 88.09% | 86.18% |
| `touches_edge` | 98.9 | 60.36% | 66.00% | 68.80% |

### Góc hướng quan sát θ (Decision D19)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `side_0_30deg` | 125.2 | 83.11% | 85.08% | 83.14% |
| `diagonal_30_60deg` | 135.7 | 85.38% | 85.67% | 84.99% |
| `front_rear_60_90deg` | 1084.6 | 87.82% | 85.33% | 83.70% |

### Mức độ khó KITTI (Difficulty)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `Easy` | 538.1 | 87.83% | 84.81% | 81.89% |
| `Moderate` | 673.0 | 86.98% | 86.32% | 85.50% |
| `Hard` | 134.3 | 84.15% | 86.90% | 86.35% |

### Nhóm Fallback Pattern 000 (Decision D51)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `normal_cues` | 1334.5 | 87.79% | 86.42% | 84.64% |
| `fallback_pattern_000` | 11.1 | 39.91% | 57.58% | 68.59% |

## Detector `yolov5su_640`

### Dải cự ly theo Ẑ dự đoán (Prospective Bins)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `0-10m` | 117.0 | 64.50% | 66.04% | 75.23% |
| `10-20m` | 343.5 | 91.63% | 88.94% | 86.97% |
| `20-30m` | 450.9 | 90.66% | 89.47% | 83.82% |
| `>=30m` | 464.9 | 81.96% | 82.53% | 83.94% |

### Dải cự ly theo Z thật (Retrospective Bins — Chẩn đoán)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `0-10m` | 114.3 | 64.11% | 65.81% | 74.14% |
| `10-20m` | 354.8 | 92.25% | 88.92% | 86.82% |
| `20-30m` | 456.8 | 88.11% | 87.21% | 82.36% |
| `30-50m` | 449.6 | 83.44% | 84.32% | 85.22% |
| `>50m` | 0.8 | 87.50% | 93.75% | 93.75% |

### Mức độ cắt biên (Truncated)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `non_truncated` | 1293.3 | 87.19% | 85.91% | 84.34% |
| `truncated` | 83.0 | 65.69% | 71.03% | 75.70% |

### Mức độ che khuất (Occluded)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `occ_0` | 860.9 | 85.94% | 82.11% | 80.80% |
| `occ_1` | 398.0 | 85.46% | 88.71% | 88.11% |
| `occ_2` | 117.5 | 84.19% | 91.38% | 89.86% |

### Chạm biên ảnh (Touch Edge)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `no_edge_touch` | 1279.0 | 88.47% | 87.13% | 85.30% |
| `touches_edge` | 97.3 | 55.78% | 61.37% | 68.28% |

### Góc hướng quan sát θ (Decision D19)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `side_0_30deg` | 126.8 | 78.37% | 85.09% | 82.67% |
| `diagonal_30_60deg` | 136.1 | 83.20% | 84.35% | 84.18% |
| `front_rear_60_90deg` | 1113.3 | 86.41% | 84.10% | 82.89% |

### Mức độ khó KITTI (Difficulty)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `Easy` | 535.6 | 87.32% | 82.07% | 79.27% |
| `Moderate` | 695.8 | 84.98% | 86.05% | 86.39% |
| `Hard` | 144.9 | 79.70% | 87.35% | 86.84% |

### Nhóm Fallback Pattern 000 (Decision D51)

| Phân nhóm | Mean n (eval) | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|
| `normal_cues` | 1363.8 | 86.22% | 85.23% | 83.91% |
| `fallback_pattern_000` | 12.6 | 50.34% | 60.46% | 68.26% |
