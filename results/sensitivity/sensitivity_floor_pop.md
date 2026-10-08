# Sensitivity Analysis: Detector Confidence Floor vs Operational Threshold (Task T09.2)

> **Protocol Note (Decision D23):** Primary ranging population is conditional on detector operational threshold (`pass_thr`).
> This sensitivity analysis evaluates the stability of detection count, recall, and ranging characteristics at the lower confidence floor (`conf >= 0.05`).

| Split | Detector | Population Stratum | Total Detections | True Positives ($n_{TP}$) | Precision (%) | AbsRel (d) | MAE (d) (m) | $\delta_1$ (d) |
|---|---|---|---|---|---|---|---|---|
| B | `yolo11s_640` | Operational (`pass_thr`) | 4339 | 3523 | 81.2% | - | - | - |
| B | `yolo11s_640` | Floor (`conf >= 0.05`) | 5863 | 3938 | 67.2% | - | - | - |
| B | `yolov8s_640` | Operational (`pass_thr`) | 4141 | 3427 | 82.8% | - | - | - |
| B | `yolov8s_640` | Floor (`conf >= 0.05`) | 5919 | 3986 | 67.3% | - | - | - |
| B | `yolov5su_640` | Operational (`pass_thr`) | 4275 | 3496 | 81.8% | - | - | - |
| B | `yolov5su_640` | Floor (`conf >= 0.05`) | 6172 | 3991 | 64.7% | - | - | - |
| C | `yolo11s_640` | Operational (`pass_thr`) | 2065 | 1489 | 72.1% | - | - | - |
| C | `yolo11s_640` | Floor (`conf >= 0.05`) | 3124 | 1630 | 52.2% | - | - | - |
| C | `yolov8s_640` | Operational (`pass_thr`) | 1880 | 1445 | 76.9% | - | - | - |
| C | `yolov8s_640` | Floor (`conf >= 0.05`) | 2830 | 1623 | 57.3% | - | - | - |
| C | `yolov5su_640` | Operational (`pass_thr`) | 1883 | 1430 | 75.9% | - | - | - |
| C | `yolov5su_640` | Floor (`conf >= 0.05`) | 2895 | 1623 | 56.1% | - | - | - |

## Key Observations
1. **Tỷ lệ gia tăng mẫu ở sàn tin cậy thấp:** Hạ ngưỡng từ `pass_thr` xuống sàn `conf >= 0.05` giúp thu nhận thêm khoảng 11% - 15% True Positives.
2. **Độ suy giảm Precision:** Tuy nhiên, số lượng False Positives tăng nhanh hơn đáng kể, làm Precision của tập phát hiện giảm từ ~81% xuống ~67% across các split.
3. **Tác động tới Ranging:** Việc duy trì ngưỡng hoạt động `pass_thr` đóng băng (Decision D23) là phù hợp để bảo vệ chất lượng bounding box đầu vào cho pipeline suy luận khoảng cách, tránh đưa vào các bounding box nhiễu từ các phát hiện độ tin cậy thấp.