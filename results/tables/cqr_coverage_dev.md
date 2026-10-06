# Kết quả Conformal Quantile Regression (CQR) trên Dev Split C (T07)

> Đánh giá Leave-One-Drive-Out (LODO) trên 10 cụm drive có Car Hard của Split C.
> Thống kê khoảng tin cậy cự ly danh nghĩa 90% (alpha = 0.1).
> Tuân thủ Decisions D18, D26, D46, D47, D48, D49, D50, D51.

## 1. Bảng tổng hợp các detector (LODO Calibration trên Split C)

| Detector | N (C) | Raw Cov (Q̂=0) | Pooled Cov | Macro Cov (10 cụm) | Macro Cov (n >= 30) | Mean Width (Z_hi/Z_lo) | Mean Winkler | Crossings | Q_hat (full C) | Gate [85%, 95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| `yolo11s_640` | 1489 | 0.6978 | **0.8717** | 0.8976 | 0.9014 | 1.330 | 0.5785 | 0 (0.0%) | +0.05107 | ✅ PASS |
| `yolov8s_640` | 1445 | 0.6512 | **0.8713** | 0.9158 | 0.9001 | 1.353 | 0.6503 | 0 (0.0%) | +0.06390 | ✅ PASS |
| `yolov5su_640` | 1430 | 0.6930 | **0.8797** | 0.9013 | 0.9004 | 1.344 | 0.6642 | 0 (0.0%) | +0.05246 | ✅ PASS |

## 2. Phân tích chẩn đoán độ dịch chuyển & Fallback (Decisions D50, D51)

> Cảnh báo: Nhóm fallback_flag=True (pattern 000) có |r| trung bình ~2.56 (do Model e under-predict Z_e ≈ 0.5m so với Z_gt ≈ 6–8.5m),
> nằm ngoài khoảng CQR ([r_lo, r_hi] ≈ [-0.2, +0.4]), dẫn tới độ phủ 0.000 ở cả 3 detector.

| Detector | Mean r (B OOF) | Mean r (C) | Residual Shift (C - B) | Q_hat LODO (Mean) | Fallback n | Fallback Cov | Fallback Mean |r| | Normal Mean |r| |
|---|---|---|---|---|---|---|---|---|
| `yolo11s_640` | +0.0199 | +0.0063 | **-0.0136** | +0.04974 | 8 | **0.0000** | 2.56 | 0.09 |
| `yolov8s_640` | +0.0208 | +0.0077 | **-0.0131** | +0.06382 | 7 | **0.0000** | 2.59 | 0.08 |
| `yolov5su_640` | +0.0213 | +0.0129 | **-0.0084** | +0.05166 | 10 | **0.0000** | 2.56 | 0.08 |

## 3. Coverage theo dải cự ly Z (D3, E1)

> Đánh giá độ phủ có điều kiện theo dải cự ly: theo Z thật (chẩn đoán) và theo Ẑ dự đoán (midpoint).

### Detector `yolo11s_640`

| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | n (Ẑ dự đoán) | Coverage (theo Ẑ dự đoán) |
|---|---|---|---|---|
| `0-10m` | 136 | 0.6103 | 137 | 0.6058 |
| `10-20m` | 430 | 0.9581 | 417 | 0.9664 |
| `20-30m` | 457 | 0.8972 | 438 | 0.9247 |
| `30-50m` | 466 | 0.8433 | 489 | 0.8262 |
| `>50m` | 0 | N/A (n=0) | 8 | 0.3750 |

### Detector `yolov8s_640`

| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | n (Ẑ dự đoán) | Coverage (theo Ẑ dự đoán) |
|---|---|---|---|---|
| `0-10m` | 135 | 0.5630 | 135 | 0.5556 |
| `10-20m` | 429 | 0.9604 | 420 | 0.9667 |
| `20-30m` | 459 | 0.9129 | 445 | 0.9393 |
| `30-50m` | 422 | 0.8341 | 435 | 0.8276 |
| `>50m` | 0 | N/A (n=0) | 10 | 0.0000 |

### Detector `yolov5su_640`

| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | n (Ẑ dự đoán) | Coverage (theo Ẑ dự đoán) |
|---|---|---|---|---|
| `0-10m` | 135 | 0.6667 | 136 | 0.6691 |
| `10-20m` | 423 | 0.9598 | 409 | 0.9633 |
| `20-30m` | 453 | 0.8786 | 429 | 0.9277 |
| `30-50m` | 419 | 0.8687 | 454 | 0.8216 |
| `>50m` | 0 | N/A (n=0) | 2 | 1.0000 |

## 4. Chi tiết per-drive (10 cụm Car Hard trên Split C)

> Lưu ý hiện tượng Drive Heterogeneity (D50): 8 drive over-cover (96–100%), trong khi 2 drive lớn
> `0057` (n=250–260) và `0004` (n=309–325) chiếm ~39% mẫu bị under-cover (67–79%), kéo pooled coverage xuống 87.1–88.0%.

### Detector `yolo11s_640`

| Drive ID | n | Q_hat (LODO) | Coverage | Mean Width (Z_hi/Z_lo) | Mean Winkler | Crossings |
|---|---|---|---|---|---|---|
| `2011_09_26_drive_0005_sync` | 55 | +0.05174 | 0.9636 | 1.355 | 1.8475 | 0 |
| `2011_09_26_drive_0011_sync` | 120 | +0.05283 | 0.9917 | 1.353 | 0.6841 | 0 |
| `2011_09_26_drive_0027_sync` | 25 | +0.05167 | 1.0000 | 1.308 | 0.2660 | 0 |
| `2011_09_26_drive_0035_sync` | 271 | +0.05379 | 0.9594 | 1.350 | 0.9160 | 0 |
| `2011_09_26_drive_0057_sync` | 260 | +0.04163 | 0.7423 | 1.344 | 0.4071 | 0 |
| `2011_09_26_drive_0079_sync` | 7 | +0.05116 | 1.0000 | 1.328 | 0.2830 | 0 |
| `2011_09_26_drive_0087_sync` | 51 | +0.05187 | 0.9804 | 1.287 | 0.2514 | 0 |
| `2011_09_26_drive_0096_sync` | 369 | +0.05736 | 0.9892 | 1.365 | 0.3247 | 0 |
| `2011_09_28_drive_0047_sync` | 6 | +0.05105 | 0.6667 | 1.349 | 0.4396 | 0 |
| `2011_09_29_drive_0004_sync` | 325 | +0.03427 | 0.6831 | 1.257 | 0.5528 | 0 |

### Detector `yolov8s_640`

| Drive ID | n | Q_hat (LODO) | Coverage | Mean Width (Z_hi/Z_lo) | Mean Winkler | Crossings |
|---|---|---|---|---|---|---|
| `2011_09_26_drive_0005_sync` | 53 | +0.06568 | 0.9623 | 1.392 | 1.9336 | 0 |
| `2011_09_26_drive_0011_sync` | 112 | +0.06866 | 0.9911 | 1.356 | 0.7147 | 0 |
| `2011_09_26_drive_0027_sync` | 24 | +0.06476 | 1.0000 | 1.356 | 0.3039 | 0 |
| `2011_09_26_drive_0035_sync` | 259 | +0.07243 | 0.9730 | 1.399 | 0.8340 | 0 |
| `2011_09_26_drive_0057_sync` | 250 | +0.04470 | 0.6680 | 1.279 | 0.8026 | 0 |
| `2011_09_26_drive_0079_sync` | 4 | +0.06390 | 1.0000 | 1.359 | 0.3064 | 0 |
| `2011_09_26_drive_0087_sync` | 59 | +0.06767 | 1.0000 | 1.350 | 0.2981 | 0 |
| `2011_09_26_drive_0096_sync` | 366 | +0.08153 | 0.9891 | 1.404 | 0.3440 | 0 |
| `2011_09_28_drive_0047_sync` | 7 | +0.06390 | 0.8571 | 1.368 | 0.3873 | 0 |
| `2011_09_29_drive_0004_sync` | 311 | +0.04500 | 0.7170 | 1.304 | 0.5976 | 0 |

### Detector `yolov5su_640`

| Drive ID | n | Q_hat (LODO) | Coverage | Mean Width (Z_hi/Z_lo) | Mean Winkler | Crossings |
|---|---|---|---|---|---|---|
| `2011_09_26_drive_0005_sync` | 54 | +0.05261 | 0.9630 | 1.355 | 1.9084 | 0 |
| `2011_09_26_drive_0011_sync` | 112 | +0.05541 | 0.9911 | 1.344 | 0.7104 | 0 |
| `2011_09_26_drive_0027_sync` | 25 | +0.05253 | 0.9600 | 1.329 | 1.9180 | 0 |
| `2011_09_26_drive_0035_sync` | 265 | +0.05742 | 0.9660 | 1.377 | 1.1165 | 0 |
| `2011_09_26_drive_0057_sync` | 252 | +0.04521 | 0.7937 | 1.348 | 0.3786 | 0 |
| `2011_09_26_drive_0079_sync` | 4 | +0.05252 | 1.0000 | 1.325 | 0.2815 | 0 |
| `2011_09_26_drive_0087_sync` | 49 | +0.05252 | 0.9184 | 1.298 | 0.2803 | 0 |
| `2011_09_26_drive_0096_sync` | 356 | +0.06126 | 0.9944 | 1.372 | 0.3169 | 0 |
| `2011_09_28_drive_0047_sync` | 4 | +0.05246 | 0.7500 | 1.325 | 0.3275 | 0 |
| `2011_09_29_drive_0004_sync` | 309 | +0.03462 | 0.6764 | 1.286 | 0.6438 | 0 |
