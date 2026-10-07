# Kết quả Conformal Quantile Regression (CQR) trên Dev Split C (Chuẩn hóa v1.1)

> **Ghi chú chuẩn hóa (Decisions D71, D74)**: Báo cáo này cập nhật sau khi khắc phục lỗi cú pháp `base_score` trong Model (e). Đánh giá hoàn toàn read-only, giữ nguyên 100% mã băm SHA-256 của các mô hình quantile.

## 1. Bảng tổng hợp các detector (LODO Calibration trên Split C - v1.1)

| Detector | N (C) | Raw Cov (Q̂=0) | Pooled Cov | Macro Cov (10 cụm) | Macro Cov (n >= 30) | Mean Width | Winkler | Crossings | Q_hat (full C) | Gate [85%, 95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| `yolo11s_640` | 1489 | 0.7018 | **0.8717** | 0.8997 | 0.9044 | 1.324 | 0.3521 | 0 | +0.04931 | ✅ PASS |
| `yolov8s_640` | 1445 | 0.6561 | **0.8713** | 0.9197 | 0.9057 | 1.342 | 0.4404 | 0 | +0.06143 | ✅ PASS |
| `yolov5su_640` | 1430 | 0.6979 | **0.8734** | 0.9038 | 0.8983 | 1.336 | 0.3666 | 0 | +0.04977 | ✅ PASS |

## 2. Khắc phục Hiện tượng Fallback (Thay thế Decision D51/D58 theo D74)

> **Phát hiện quan trọng**: Hiện tượng 'Model e under-predict Z_e ≈ 0.5m, độ phủ fallback = 0.000' ghi nhận trước đây là hệ quả của bug cú pháp `base_score: '[...]'` trong XGBoost JSON khiến parser fallback về 0.5. Sau khi chuẩn hóa cú pháp, Model (e) dự đoán chính xác và độ phủ fallback trên Split C không còn bằng 0.

| Detector | Fallback n | Fallback Cov (v1.1) | Fallback Cov (v1 cũ) | Fallback Mean |r| (v1.1) | Normal Mean |r| |
|---|---|---|---|---|---|
| `yolo11s_640` | 8 | **100.0%** | 0.0% (buggy) | 0.0979 | 0.0850 |
| `yolov8s_640` | 7 | **100.0%** | 0.0% (buggy) | 0.0810 | 0.0827 |
| `yolov5su_640` | 10 | **90.0%** | 0.0% (buggy) | 0.0903 | 0.0811 |

## 3. Độ phủ theo Dải cự ly Z (Split C)

### Detector `yolo11s_640`

| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | Mean Width |
|---|---|---|---|
| `0-10m` | 136 | 0.6618 | 1.431x |
| `10-20m` | 430 | 0.9558 | 1.317x |
| `20-30m` | 457 | 0.8972 | 1.309x |
| `30-50m` | 466 | 0.8305 | 1.315x |

### Detector `yolov8s_640`

| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | Mean Width |
|---|---|---|---|
| `0-10m` | 135 | 0.6148 | 1.356x |
| `10-20m` | 429 | 0.9604 | 1.345x |
| `20-30m` | 459 | 0.9020 | 1.333x |
| `30-50m` | 422 | 0.8294 | 1.343x |

### Detector `yolov5su_640`

| Dải cự ly | n (Z thật) | Coverage (theo Z thật) | Mean Width |
|---|---|---|---|
| `0-10m` | 135 | 0.7111 | 1.396x |
| `10-20m` | 423 | 0.9433 | 1.327x |
| `20-30m` | 453 | 0.8653 | 1.319x |
| `30-50m` | 419 | 0.8640 | 1.344x |

