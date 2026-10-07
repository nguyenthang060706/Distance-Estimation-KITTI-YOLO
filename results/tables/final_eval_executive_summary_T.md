# Split T Final Evaluation Executive Summary (Post-hoc Verified v1.1)

## 1. Tóm tắt Kiểm toán và Khắc phục Lỗi Triển khai (Decisions D71, D74)

- **Phát hiện lỗi**: Trong các tệp artifact đóng băng `model_f.json` và `model_e.json`, trường `learner.learner_model_param.base_score` bị serialize dưới dạng chuỗi mảng (ví dụ `'[1.9926282E-2]'`). Parser C++ của XGBoost khi nạp gặp lỗi parse nên silently fallback về giá trị mặc định 0.5, gây lệch hằng số +0.5 trong log residuals $\hat{r}$ (AbsRel Model f tăng lên ~0.62, Model e lên ~0.92, Split Conformal nở rộng 3.37x). Đồng thời, hiện tượng fallback pattern 000 ở T07/T08 có coverage = 0% thực chất là do cùng bug này khi nạp `model_e.json` (D74).
- **Khắc phục (D71, D74)**: Đã xóa ngoặc vuông để khôi phục chuỗi số vô hướng gốc. Không huấn luyện lại, không chỉnh sửa tham số, không chạy lại inference trên Split T (`runs/final_T.lock` được giữ nguyên vẹn 100%). Dữ liệu v1.1 được tính lại post-hoc từ các file parquet tĩnh.
- **Minh bạch học thuật**: Toàn bộ bảng dưới đây báo cáo song song kết quả v1 (có bug) và v1.1 (đã sửa).

## 2. Ước lượng Điểm trên Split T (Bản v1.1 Chuẩn hóa)

| Detector | Model (d) Fused | Model (f0) Ridge | Model (f) Residual | Model (e) Direct | Delta1 (f) |
|---|---|---|---|---|---|
| yolo11s_640 | 0.0640 | 0.0553 | **0.0463** | 0.0465 | 0.9967 |
| yolov8s_640 | 0.0641 | 0.0551 | **0.0461** | 0.0459 | 0.9970 |
| yolov5su_640 | 0.0642 | 0.0560 | **0.0474** | 0.0474 | 0.9951 |

## 3. Khoảng Tin cậy Conformal (Mức danh nghĩa 90%)

| Detector | CQR Pooled | CQR Macro (10 drives) | CQR Macro (n≥30, 8 drives) | CQR Width | SC Width | Mondrian Width |
|---|---|---|---|---|---|---|
| yolo11s_640 | 96.4% | 87.9% | 97.4% | 1.319x | 1.322x | 1.294x |
| yolov8s_640 | 97.1% | 96.9% | 97.5% | 1.352x | 1.345x | 1.353x |
| yolov5su_640 | 96.6% | 97.9% | 97.4% | 1.332x | 1.351x | 1.316x |

## 4. Phân tích Thống kê và Lưu ý Phương pháp luận (Decisions D68, D73, D75)

- **Điều kiện hóa trên True Positives**: Toàn bộ chỉ số điểm và khoảng được tính trên các phát hiện TP vượt ngưỡng tin cậy (Recall 82.8%–84.4%). Số lượng False Negatives tương ứng của 3 detector là 500 / 552 / 538 mẫu GT.
- **Độ phủ thực nghiệm & Tính chất bảo thủ**: Standard CQR đạt độ phủ tổng gộp 96.4%–97.1%, cao hơn mức danh nghĩa 90% khoảng 6–7 điểm phần trăm. Đây là khoảng bảo thủ (over-coverage) ngoài mẫu, không phải khoảng thắt chặt.
- **Độ phủ dải gần 0–10m (RQ3)**: Đạt 91.6% / 93.6% / 89.4% (dải 89.4%–93.6%), vẫn đạt xấp xỉ và duy trì quanh mức danh nghĩa 90%.
- **Cụm cỡ mẫu nhỏ và Macro Coverage (D75)**: Cụm `drive_0002` chỉ có $n=2$ mẫu TP. Trên `yolo11s`, cả 2 mẫu đều không được cover (0/2), kéo macro coverage trung bình không trọng số của yolo11s xuống 87.9%. Khi tính macro trên 8 cụm có $n \ge 30$, độ phủ đạt 97.4% đồng đều ở cả 3 detector.
- **So sánh Cặp Bootstrap (10 cụm drive, B=1000)**:
  * Model (f) vs Model (d): CI thô loại trừ 0 (ước lượng $\Delta \approx -0.018$) $\to$ Residual phi tuyến cải thiện rõ so với mô hình hình học thuần túy (d).
  * Model (f) vs Model (f0): CI thô loại trừ 0 (ước lượng $\Delta \approx -0.009$) $\to$ Residual phi tuyến cải thiện so với baseline tuyến tính (f0).
  * Model (f) vs Model (e): CI thô chứa 0 (ước lượng $\Delta \approx -0.0002$, 95% CI [-0.0012, +0.0023]) $\to$ Residual (f) không phân biệt được với hồi quy trực tiếp (e) trên Split T (Limitations #8).
- **Tương quan RQ2**: Tương quan giữa sai số AbsRel và các đặc trưng phát hiện là rất yếu ($|r| \le 0.15$). Tương quan với kích thước bbox bị nhiễu mạnh bởi cự ly $Z$ thực tế (hiệu ứng phối cảnh $h \propto 1/Z$ theo D21). Sai số tiếp đất $\Delta y_2$ có tương quan thực nghiệm rất nhỏ.
