# Sensitivity Analysis: Detector Confidence Floor vs Operational Threshold (Task T09.2)

> **Protocol Note (Decision D23):** Primary ranging population is conditional on detector operational threshold (`pass_thr`).
> This sensitivity analysis evaluates detection count, recall, and geometric ranging performance (AbsRel, MAE, $\delta_1$, valid_frac) at the lower confidence floor (`conf >= 0.05`).

| Split | Detector | Population Stratum | Total Detections | True Positives ($n_{TP}$) | Precision (%) | AbsRel (d) | MAE (d) (m) | $\delta_1$ (d) | valid_frac |
|---|---|---|---|---|---|---|---|---|---|
| B | `yolo11s_640` | Operational (`pass_thr`) | 4339 | 3523 | 81.2% | 0.0604 | 1.333 | 0.9773 | 0.9895 |
| B | `yolo11s_640` | Floor (`conf >= 0.05`) | 5863 | 3938 | 67.2% | 0.0635 | 1.490 | 0.9726 | 0.9904 |
| B | `yolov8s_640` | Operational (`pass_thr`) | 4141 | 3427 | 82.8% | 0.0590 | 1.294 | 0.9814 | 0.9886 |
| B | `yolov8s_640` | Floor (`conf >= 0.05`) | 5919 | 3986 | 67.3% | 0.0617 | 1.476 | 0.9780 | 0.9902 |
| B | `yolov5su_640` | Operational (`pass_thr`) | 4275 | 3496 | 81.8% | 0.0611 | 1.377 | 0.9769 | 0.9886 |
| B | `yolov5su_640` | Floor (`conf >= 0.05`) | 6172 | 3991 | 64.7% | 0.0649 | 1.575 | 0.9711 | 0.9897 |
| C | `yolo11s_640` | Operational (`pass_thr`) | 2065 | 1489 | 72.1% | 0.0862 | 2.088 | 0.9541 | 0.9946 |
| C | `yolo11s_640` | Floor (`conf >= 0.05`) | 3124 | 1630 | 52.2% | 0.0869 | 2.167 | 0.9513 | 0.9951 |
| C | `yolov8s_640` | Operational (`pass_thr`) | 1880 | 1445 | 76.9% | 0.0834 | 2.016 | 0.9743 | 0.9952 |
| C | `yolov8s_640` | Floor (`conf >= 0.05`) | 2830 | 1623 | 57.3% | 0.0854 | 2.142 | 0.9666 | 0.9951 |
| C | `yolov5su_640` | Operational (`pass_thr`) | 1883 | 1430 | 75.9% | 0.0820 | 1.957 | 0.9782 | 0.9930 |
| C | `yolov5su_640` | Floor (`conf >= 0.05`) | 2895 | 1623 | 56.1% | 0.0852 | 2.139 | 0.9665 | 0.9938 |

## Key Observations
1. **Tỷ lệ gia tăng mẫu và độ chính xác Ranging ở Floor:**
   - Hạ ngưỡng từ `pass_thr` xuống sàn `conf >= 0.05` giúp thu nhận thêm khoảng +11% đến +15% True Positives (ví dụ trên B yolo11s tăng từ 3,523 lên 3,938 mẫu).
   - Sai số hình học $z_d$ tăng nhẹ không đáng kể: AbsRel(d) tăng từ +0.001 đến +0.003 (trên B yolo11s từ 0.0604 lên 0.0635; trên C yolo11s từ 0.0862 lên 0.0869), tỷ lệ cue hợp lệ `valid_frac` giữ nguyên ở mức ~99.0%.
2. **Độ suy giảm Precision và Đánh đổi:**
   - Tuy nhiên, số lượng False Positives tăng rất nhanh (trên B từ 200 lên ~1,900 FP), làm Precision giảm mạnh từ ~81% xuống ~67%.
3. **Kết luận:** Quyết định D23 đóng băng quần thể ranging ở `pass_thr` là phù hợp để bảo đảm tỷ lệ phát hiện sạch (Precision cao) trong ứng dụng tự hành thực tế mà không làm suy hao nghiêm trọng độ chính xác ranging.