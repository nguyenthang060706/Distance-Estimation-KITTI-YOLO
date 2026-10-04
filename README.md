# Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors

## Overview
Monocular vehicle distance estimation using pinhole geometry cues (width, height, ground plane) fused in log-space, with a learned residual corrector and Conformal Quantile Regression (CQR) for calibrated prediction intervals.

**Dataset:** KITTI Object Detection (7,481 labeled images)  
**Detectors:** YOLOv5su, YOLOv8s, YOLO11s (Ultralytics)  
**Timeline:** 3 weeks

## Project Structure

```
Distance-Estimation-KITTI-YOLO/
│
├── data/                          # Dữ liệu (không commit lên git)
│   ├── kitti/                     # Dữ liệu KITTI gốc
│   │   ├── image_2/              # Ảnh training (7,481 ảnh)
│   │   ├── label_2/              # Nhãn 2D/3D
│   │   ├── calib/                # File calibration (P2, R0_rect)
│   │   └── devkit/               # Devkit: train_mapping.txt, train_rand.txt
│   └── yolo_format/              # Dữ liệu đã chuyển sang định dạng YOLO
│
├── splits/                        # Split A/V/B/C/T đóng băng (có hash)
│
├── configs/                       # Cấu hình thí nghiệm
│   ├── detector/                 # Cấu hình huấn luyện YOLO (chung cho 3 detector)
│   └── residual/                 # Cấu hình XGBoost/MLP
│
├── src/                           # Source code chính
│   ├── detection/                # Fine-tune, suy luận YOLO, khớp Greedy (D16)
│   ├── geometry/                 # 3 cue hình học (Z_w, Z_h, Z_g), hợp nhất log-space
│   ├── residual/                 # Hiệu chỉnh residual (XGBoost, MLP)
│   ├── uncertainty/              # CQR, Mondrian CQR, conformalization
│   ├── evaluation/               # eval.py: MAE, RMSE, AbsRel, δ<1.25, cluster bootstrap
│   └── utils/                    # Loader KITTI, bbox mapping, helpers
│
├── scripts/                       # Scripts chạy pipeline end-to-end
│
├── tests/                         # Unit tests
│
├── runs/                          # Checkpoint & log huấn luyện detector
│   ├── yolov5su/
│   ├── yolov8s/
│   └── yolo11s/
│
├── results/                       # Kết quả thí nghiệm
│   ├── tables/                   # Bảng CSV/LaTeX
│   ├── figures/                  # Biểu đồ
│   └── predictions/              # Dự đoán raw
│
├── notebooks/                     # Jupyter notebooks (EDA, visualization)
│
├── docs/                          # Tài liệu dự án
│   ├── KE_HOACH_V4.md            # Kế hoạch chi tiết v4
│   ├── NHAT_KY_QUYET_DINH.md    # Nhật ký quyết định
│   └── ke_hoach_distance_estimation_v3.docx  # Kế hoạch v3 (tham khảo)
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Pipeline

```
Ảnh → YOLO detector → bbox (map về ảnh gốc)
    → Lọc / Khớp Greedy (D16)
    → 3 cue: Z_w (chiều rộng), Z_h (chiều cao), Z_g (cạnh dưới)
    → Hợp nhất log-space (trọng số theo hiệp phương sai)
    → Z_d
    → Residual r̂ = f(x) trên log-tỉ lệ
    → Ẑ = Z_d · exp(r̂)
    → CQR trên r → khoảng [Z_lo, Z_hi]
```

## Data Splits (by drive)

| Split | Ratio | Detector sees? | Role |
|-------|-------|----------------|------|
| **A** | 50%   | Yes            | Fine-tune detectors, estimate priors |
| **V** | 5%    | Yes (early stop)| Validation, choose imgsz & conf threshold |
| **B** | 20%   | No             | Fit fusion weights, residual, quantile functions |
| **C** | 10%   | No             | Conformalize CQR only |
| **T** | 15%   | No             | Final test (run once, frozen config) |

## Research Questions
- **RQ1:** How much does the residual improve over pure pinhole at each distance range?
- **RQ2:** How do different YOLO detectors affect ranging accuracy?
- **RQ3:** Does CQR achieve nominal coverage, and how does it vary by conditions?

## Setup
```bash
pip install -r requirements.txt
```

## References
See [docs/KE_HOACH_V4.md](docs/KE_HOACH_V4.md) §12 for full reference list.
