# Đánh giá Độ phủ có điều kiện (Conditional Coverage) — T08

> Báo cáo độ phủ theo phân nhóm qua 20 lần chia lại eval out-of-sample ($B \cup C$).
> Tuân thủ D18, D26, D50, D51, D54:
> - $n_{\text{unique}}$: Số lượng mẫu duy nhất trong toàn bộ quần thể $B \cup C$ (không cộng dồn lặp qua seed).
> - $k_{\text{clusters}}$: Số lượng cụm hành trình duy nhất chứa mẫu thuộc phân nhóm đó.
> - Cờ `*`: Gắn nhãn khi $n_{\text{unique}} < 100$ cảnh báo cỡ mẫu nhỏ / độ biến động cao.
> - Dải theo Z thật chỉ dùng cho mục đích chẩn đoán (retrospective diagnostic, v4 §6), không dùng để chọn cấu hình.

---

## 1. Detector `yolo11s_640` (Tổng $N_{\text{unique}} = 5,012$, $k=22$ cụm)

### Dải cự ly theo Ẑ dự đoán (Prospective Bins)

| Phân nhóm Ẑ | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `0-10m` | 415 | 20 | 114.5 | 65.57% | 68.04% | **74.88%** |
| `10-20m` | 1519 | 21 | 349.1 | 90.32% | 89.93% | 87.49% |
| `20-30m` | 1682 | 17 | 454.9 | 88.88% | 88.52% | 85.98% |
| `>=30m` | 1396 | 19 | 479.4 | 83.34% | 82.56% | 82.65% |

### Dải cự ly theo Z thật (Retrospective Bins — Chẩn đoán)

| Dải $Z_{\text{gt}}$ | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `0-10m` | 415 | 20 | 113.8 | 65.49% | 67.98% | **73.68%** |
| `10-20m` | 1519 | 21 | 356.4 | 90.68% | 89.68% | 87.74% |
| `20-30m` | 1682 | 17 | 459.1 | 87.90% | 87.84% | 85.57% |
| `30-50m` | 1390 | 19 | 467.1 | 83.80% | 83.34% | 83.26% |
| `>50m` | 6* | 3 | 1.5 | 63.33% | 80.00% | 63.33% |

### Mức độ cắt biên (Truncated) & Chạm viền ảnh (Touch Edge)

| Phân nhóm | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `non_truncated` | 4742 | 22 | 1315.0 | 86.80% | 85.85% | 84.92% |
| `truncated` | 270 | 18 | 82.8 | 66.98% | 74.81% | **79.32%** |
| `no_edge_touch` | 4709 | 22 | 1296.3 | 87.96% | 86.92% | 85.92% |
| `touches_edge` | 303 | 18 | 101.6 | 59.17% | 66.72% | **71.13%** |

### Mức độ che khuất (Occluded)

| Phân nhóm | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `occ_0` | 2819 | 22 | 868.2 | 85.97% | 82.71% | 82.28% |
| `occ_1` | 1490 | 17 | 408.8 | 84.48% | 88.76% | 87.82% |
| `occ_2` | 703 | 14 | 121.0 | 83.59% | 88.31% | 87.56% |

### Góc hướng quan sát θ (Decision D19)

| Phân nhóm góc | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `side (<30 deg)` | 403 | 11 | 128.2 | 72.84% | 81.44% | 80.09% |
| `diagonal (30-60 deg)` | 772 | 15 | 138.9 | 79.44% | 82.57% | 81.64% |
| `front_rear (>=60 deg)` | 3837 | 19 | 1130.8 | 86.69% | 84.51% | 84.03% |

### Mức độ khó KITTI (Difficulty)

| Phân nhóm | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `Easy` | 1873 | 21 | 533.9 | 86.39% | 83.15% | 81.62% |
| `Moderate` | 2345 | 21 | 715.6 | 85.31% | 85.81% | 85.54% |
| `Hard` | 794 | 16 | 148.4 | 79.32% | 85.21% | 85.69% |

### Nhóm Fallback Pattern 000 (Decisions D13, D51)

| Phân nhóm | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| `normal_cues` | 4967 | 22 | 1387.0 | 85.95% | 85.40% | 84.67% |
| `fallback_pattern_000` | 45* | 8 | 10.8 | 42.25% | 58.13% | 67.76% |

---

## 2. Detector `yolov8s_640` (Tổng $N_{\text{unique}} = 4,872$, $k=22$ cụm)

| Phân nhóm | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| **Ẑ: 0-10m** | 419 | 20 | 119.0 | 66.57% | 68.93% | **73.57%** |
| **Ẑ: 10-20m** | 1518 | 21 | 348.9 | 92.58% | 91.95% | 89.24% |
| **Ẑ: 20-30m** | 1663 | 18 | 447.5 | 91.32% | 90.83% | 87.19% |
| **Ẑ: >=30m** | 1272 | 18 | 430.1 | 85.00% | 81.77% | 81.61% |
| **$Z_{\text{gt}}$: >50m** | 2* | 2 | 0.4 | 100.00% | 100.00% | 100.00% |
| **Truncated** | 268 | 18 | 81.8 | 70.16% | 76.37% | 79.11% |
| **Touch edge** | 295 | 18 | 90.9 | 61.12% | 69.17% | 71.30% |
| **Fallback (000)** | 46* | 8 | 9.9 | 43.14% | 55.67% | 66.19% |

---

## 3. Detector `yolov5su_640` (Tổng $N_{\text{unique}} = 4,926$, $k=22$ cụm)

| Phân nhóm | $n_{\text{unique}}$ | $k_{\text{clusters}}$ | Mean $n_{\text{eval}}$ | Split Conformal Cov | Standard CQR Cov | Mondrian CQR Cov |
|---|---|---|---|---|---|---|
| **Ẑ: 0-10m** | 415 | 20 | 118.8 | 68.51% | 66.86% | **72.63%** |
| **Ẑ: 10-20m** | 1506 | 21 | 348.4 | 91.73% | 90.17% | 88.06% |
| **Ẑ: 20-30m** | 1651 | 18 | 446.7 | 89.44% | 88.75% | 86.86% |
| **Ẑ: >=30m** | 1354 | 18 | 447.8 | 83.33% | 81.74% | 81.25% |
| **$Z_{\text{gt}}$: >50m** | 4* | 2 | 1.1 | 58.33% | 66.67% | 75.00% |
| **Truncated** | 269 | 18 | 83.2 | 67.92% | 74.22% | 78.69% |
| **Touch edge** | 287 | 17 | 89.8 | 61.35% | 68.21% | 70.82% |
| **Fallback (000)** | 50* | 10 | 11.2 | 41.87% | 56.40% | 67.31% |
