# Báo cáo Tác vụ T02: Hình học trên Bbox Detector (D17, D29)

## 1. Kiểm tra Regression trên GT Bbox (Toàn bộ Split B, 4.776 xe)
- **Số lượng mẫu:** 4776 Car Hard
- **In-sample AbsRel:** 0.0605 (chuẩn Day 4: 0.0605, chênh lệch: 0.0000 <= 0.002) -> **ĐẠT**
- **LODO OOF AbsRel:** 0.0610 (chuẩn Day 6: 0.0609, chênh lệch: 0.0001 <= 0.002) -> **ĐẠT**

## 2. Bảng Tỷ lệ Mask Viền Ảnh (Mask Rate) giữa GT Bbox vs Detector Bbox trên tập TP

| Detector | N (TP) | Mask $Z_w$ (GT / Det) | Mask $Z_h$ (GT / Det) | Mask $Z_g$ (GT / Det) | All Invalid (GT / Det) |
|---|---|---|---|---|---|
| `yolov8s_640` | 3427 | 0.036 / **0.034** | 0.029 / **0.024** | 0.029 / **0.024** | 0.0137 / **0.0114** |
| `yolo11s_640` | 3523 | 0.035 / **0.031** | 0.027 / **0.025** | 0.027 / **0.025** | 0.0128 / **0.0105** |
| `yolov5su_640` | 3496 | 0.035 / **0.033** | 0.028 / **0.023** | 0.028 / **0.023** | 0.0132 / **0.0114** |

> [!NOTE]
> Tỷ lệ mask của detector bbox thấp hơn GT bbox do detector dự đoán viền cách lề ảnh 1-3 px thay vì 0 px.
> Theo quy định T02 mục Cấm: báo cáo trung thực hiện tượng này, KHÔNG tự ý đổi eps (eps=2.0 px giữ nguyên).

## 3. Trọng số Hợp nhất: GT-Bbox Đối chứng vs Refit Detector (D17)

| Detector | $w_w$ (Width) | $w_h$ (Height) | $w_g$ (Ground) | Ghi chú |
|---|---|---|---|---|
| **GT-Bbox (Chuẩn D10)** | 0.0807 | 0.6634 | 0.2560 | Cận dưới lý thuyết trên GT |
| `yolov8s_640` (Refit) | 0.0000 | 0.7151 | 0.2849 | Refit trên TP Split B |
| `yolo11s_640` (Refit) | 0.0000 | 0.6984 | 0.3016 | Refit trên TP Split B |
| `yolov5su_640` (Refit) | 0.0000 | 0.6673 | 0.3327 | Refit trên TP Split B |

## 4. Hiệu năng Ước lượng (a)–(d) trên Split B (Pooled vs Macro by Drive)

| Model | Phương pháp | N | N valid | Valid % | Pooled AbsRel | Macro AbsRel | Pooled RMSE | Pooled $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| `yolov8s_640` | (d) Det Refit | 3427 | 3388 | 98.9% | **0.0592** | 0.0674 | 1.81 m | 98.1% |
| `yolov8s_640` | (d) Det GT-weights | 3427 | 3388 | 98.9% | **0.0635** | 0.0617 | 1.86 m | 98.2% |
| `yolov8s_640` | (b) $Z_h$ Det | 3427 | 3345 | 97.6% | **0.0629** | 0.0703 | 1.93 m | 97.7% |
| `yolov8s_640` | (c) $Z_g$ Det | 3427 | 3345 | 97.6% | **0.0940** | 0.1302 | 3.61 m | 94.2% |
| `yolov8s_640` | (a) $Z_w$ Det | 3427 | 3312 | 96.6% | **0.2413** | 0.2530 | 8.30 m | 48.8% |
| `yolov8s_640` | (d) GT Bbox | 3427 | 3380 | 98.6% | **0.0618** | 0.0645 | 1.84 m | 98.2% |
| `yolo11s_640` | (d) Det Refit | 3523 | 3486 | 98.9% | **0.0607** | 0.0759 | 1.89 m | 97.7% |
| `yolo11s_640` | (d) Det GT-weights | 3523 | 3486 | 98.9% | **0.0659** | 0.0685 | 1.97 m | 97.9% |
| `yolo11s_640` | (b) $Z_h$ Det | 3523 | 3434 | 97.5% | **0.0650** | 0.0775 | 2.00 m | 97.8% |
| `yolo11s_640` | (c) $Z_g$ Det | 3523 | 3434 | 97.5% | **0.0931** | 0.1302 | 3.72 m | 93.9% |
| `yolo11s_640` | (a) $Z_w$ Det | 3523 | 3413 | 96.9% | **0.2462** | 0.2436 | 8.91 m | 48.4% |
| `yolo11s_640` | (d) GT Bbox | 3523 | 3478 | 98.7% | **0.0601** | 0.0623 | 1.82 m | 98.4% |
| `yolov5su_640` | (d) Det Refit | 3496 | 3456 | 98.9% | **0.0613** | 0.0742 | 2.03 m | 97.6% |
| `yolov5su_640` | (d) Det GT-weights | 3496 | 3456 | 98.9% | **0.0653** | 0.0681 | 2.01 m | 97.7% |
| `yolov5su_640` | (b) $Z_h$ Det | 3496 | 3416 | 97.7% | **0.0668** | 0.0786 | 2.20 m | 97.1% |
| `yolov5su_640` | (c) $Z_g$ Det | 3496 | 3416 | 97.7% | **0.0923** | 0.1301 | 4.15 m | 94.2% |
| `yolov5su_640` | (a) $Z_w$ Det | 3496 | 3382 | 96.7% | **0.2435** | 0.2376 | 8.54 m | 48.7% |
| `yolov5su_640` | (d) GT Bbox | 3496 | 3450 | 98.7% | **0.0608** | 0.0635 | 1.90 m | 98.3% |

## 5. Paired Cluster Bootstrap: (d)-Detector vs (d)-GT Bbox (Phân rã sai số)

| Model | N chung | Cụm | Ước lượng $\Delta$ (Det - GT) | 95% CI thô | Khác biệt loại 0? |
|---|---|---|---|---|---|
| `yolov8s_640` | 3378 | 12 | +-0.0030 | [-0.0114, +0.0060] | Không (CI thô (12 cụm)) |
| `yolo11s_640` | 3476 | 12 | +0.0000 | [-0.0081, +0.0092] | Không (CI thô (12 cụm)) |
| `yolov5su_640` | 3449 | 12 | +0.0002 | [-0.0092, +0.0104] | Không (CI thô (12 cụm)) |

## 6. False Negatives (FN) và Recall theo Dải Khoảng cách trên Split B

| Model | Dải khoảng cách | GT Cars | TP | FN | Recall | Cờ n < 100 |
|---|---|---|---|---|---|---|
| `yolov8s_640` | 0-10 | 288 | 284 | 4 | 98.6% |  |
| `yolov8s_640` | 10-20 | 1218 | 1089 | 129 | 89.4% |  |
| `yolov8s_640` | 20-30 | 1546 | 1204 | 342 | 77.9% |  |
| `yolov8s_640` | 30-50 | 1705 | 848 | 857 | 49.7% |  |
| `yolov8s_640` | >50 | 19* | 2 | 17 | 10.5% | * |
| `yolov8s_640` | >30 | 1724 | 850 | 874 | 49.3% |  |
| `yolo11s_640` | 0-10 | 288 | 279 | 9 | 96.9% |  |
| `yolo11s_640` | 10-20 | 1218 | 1089 | 129 | 89.4% |  |
| `yolo11s_640` | 20-30 | 1546 | 1225 | 321 | 79.2% |  |
| `yolo11s_640` | 30-50 | 1705 | 924 | 781 | 54.2% |  |
| `yolo11s_640` | >50 | 19* | 6 | 13 | 31.6% | * |
| `yolo11s_640` | >30 | 1724 | 930 | 794 | 53.9% |  |
| `yolov5su_640` | 0-10 | 288 | 280 | 8 | 97.2% |  |
| `yolov5su_640` | 10-20 | 1218 | 1083 | 135 | 88.9% |  |
| `yolov5su_640` | 20-30 | 1546 | 1198 | 348 | 77.5% |  |
| `yolov5su_640` | 30-50 | 1705 | 931 | 774 | 54.6% |  |
| `yolov5su_640` | >50 | 19* | 4 | 15 | 21.1% | * |
| `yolov5su_640` | >30 | 1724 | 935 | 789 | 54.2% |  |
