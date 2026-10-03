# Đánh giá Hiệu năng Detector trên KITTI Split B và Split C (§5.1, §6)

Báo cáo đánh giá 3 mô hình detector (`yolov8s_640`, `yolo11s_640`, `yolov5su_640`) trên:
- **Split B (Residual Fitting):** 1.499 ảnh, 4.776 Car Hard.
- **Split C (Conformal Calibration):** 766 ảnh, 1.826 Car Hard.
- **Cấu hình:** FP32, imgsz=640, matching IoU 0.5 (Greedy theo điểm số, Decision D15 & D16), DontCare mode `iou`.

## 1. Hiệu năng Tổng thể trên Split B

| Detector | Conf Thr | Preds (Pass) | TP | FP | Ign Non-Hard | Ign DontCare | Precision (%) | Recall (%) | F1 | Recall Floor (0.05) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8s** | 0.790 | 4,141 | 3,427 | 195 | 500 | 19 | **94.62%** | **71.75%** | **0.8161** | 83.46% |
| **yolo11s** | 0.700 | 4,339 | 3,523 | 200 | 599 | 17 | **94.63%** | **73.76%** | **0.8290** | 82.45% |
| **yolov5su** | 0.740 | 4,275 | 3,496 | 204 | 563 | 12 | **94.49%** | **73.20%** | **0.8249** | 83.56% |

### 1.1. Phân rã Recall theo Dải Khoảng cách trên Split B

| Dải Khoảng cách | Tổng GT Car Hard | Recall yolov8s | Recall yolo11s | Recall yolov5su | Tập chung (Cả 3 bắt được) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0-10m** | 288 | 98.61% | 96.88% | 97.22% | **96.18%** (277) |
| **10-20m** | 1,218 | 89.41% | 89.41% | 88.92% | **85.88%** (1,046) |
| **20-30m** | 1,546 | 77.88% | 79.24% | 77.49% | **72.64%** (1,123) |
| **30-50m** | 1,705 | 49.74% | 54.19% | 54.60% | **42.99%** (733) |
| **>50m** * | 19 | 10.53% | 31.58% | 21.05% | **10.53%** (2) |

### 1.2. Phân rã Recall theo Mức độ Khó KITTI trên Split B (Cumulative)

| Độ khó KITTI | Tổng GT | Recall yolov8s | Recall yolo11s | Recall yolov5su |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 1,311 | 99.08% | 98.40% | 99.39% |
| **Moderate** | 3,248 | 86.79% | 89.87% | 89.16% |
| **Hard** | 4,776 | 71.75% | 73.76% | 73.20% |

### 1.3. Phân tích Tập Hỗ trợ Chung (Common Support) trên Split B
- Số GT Car Hard được cả 3 detector phát hiện đồng thời: **3,181** / 4,776 (**66.60%**).
- Hợp (Union) của các GT được ít nhất 1 detector phát hiện: **3,772** (**78.98%**).

## 1. Hiệu năng Tổng thể trên Split C

| Detector | Conf Thr | Preds (Pass) | TP | FP | Ign Non-Hard | Ign DontCare | Precision (%) | Recall (%) | F1 | Recall Floor (0.05) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8s** | 0.790 | 1,880 | 1,445 | 137 | 264 | 34 | **91.34%** | **79.13%** | **0.8480** | 88.88% |
| **yolo11s** | 0.700 | 2,065 | 1,489 | 165 | 376 | 35 | **90.02%** | **81.54%** | **0.8557** | 89.27% |
| **yolov5su** | 0.740 | 1,883 | 1,430 | 156 | 276 | 21 | **90.16%** | **78.31%** | **0.8382** | 88.88% |

### 1.1. Phân rã Recall theo Dải Khoảng cách trên Split C

| Dải Khoảng cách | Tổng GT Car Hard | Recall yolov8s | Recall yolo11s | Recall yolov5su | Tập chung (Cả 3 bắt được) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0-10m** | 138 | 97.83% | 98.55% | 97.83% | **97.10%** (134) |
| **10-20m** | 472 | 90.89% | 91.10% | 89.62% | **88.56%** (418) |
| **20-30m** | 559 | 82.11% | 81.75% | 81.04% | **77.46%** (433) |
| **30-50m** | 657 | 64.23% | 70.93% | 63.77% | **56.93%** (374) |
| **>50m** * | 0 | 0.00% | 0.00% | 0.00% | **0.00%** (0) |

### 1.2. Phân rã Recall theo Mức độ Khó KITTI trên Split C (Cumulative)

| Độ khó KITTI | Tổng GT | Recall yolov8s | Recall yolo11s | Recall yolov5su |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 605 | 97.19% | 96.36% | 95.21% |
| **Moderate** | 1,441 | 88.62% | 90.15% | 87.16% |
| **Hard** | 1,826 | 79.13% | 81.54% | 78.31% |

### 1.3. Phân tích Tập Hỗ trợ Chung (Common Support) trên Split C
- Số GT Car Hard được cả 3 detector phát hiện đồng thời: **1,359** / 1,826 (**74.42%**).
- Hợp (Union) của các GT được ít nhất 1 detector phát hiện: **1,543** (**84.50%**).

## 2. Ghi chú Thảo luận & Hạn chế cho Paper
1. **Độ sụt giảm Recall ở ngưỡng F1 tối ưu:** Ngưỡng F1 chọn trên Split V tối ưu hóa F1 nhưng làm giảm Recall ~10–12% so với trần ở sàn 0.05. Sự sụt giảm tập trung chủ yếu ở dải cự ly xa (30–50 m và >50 m), nơi kích thước vật thể nhỏ và độ tin cậy detector thấp hơn.
2. **Đặc thù Split C v2:** Toàn bộ mẫu xe Car Hard trên Split C đều có cự ly $\le 50$ m ($n_{>50m} = 0$). Đánh giá detector trên Split C phản ánh chân thực phân bố đô thị cự ly gần và trung bình.
3. **Tập khớp chung (Common Support):** Việc đánh giá ranging trên tập khớp chung loại bỏ hoàn toàn thiên lệch do recall detector khác nhau, bảo đảm so sánh công bằng giữa các thuật toán ước lượng cự ly.
