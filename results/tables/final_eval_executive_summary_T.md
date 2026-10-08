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

## 4. Phân tích Thống kê và Lưu ý Phương pháp luận (Decisions D68, D73, D75, D78, D79, D81)

- **Điều kiện hóa trên True Positives**: Toàn bộ chỉ số điểm và khoảng được tính trên các phát hiện TP vượt ngưỡng tin cậy (Recall 82.8%–84.4%). Số lượng False Negatives tương ứng của 3 detector là 500 / 552 / 538 mẫu GT.
- **Độ phủ thực nghiệm & Tính chất bảo thủ (D79, D87)**: Standard CQR đạt độ phủ tổng gộp 96.4%–97.1%, cao hơn mức danh nghĩa 90% khoảng +6.4 đến +7.1 điểm phần trăm. Đây là khoảng bảo thủ (over-coverage) ngoài mẫu, mang tính diễn giải hậu nghiệm (post-hoc exploratory theo D87, đối chiếu với tiên đoán under-coverage ban đầu tại D68). Chẩn đoán tính khả hoán mô tả (D87) cho thấy sai số log-residual $|r|$ trên C lớn hơn T (KS stat = 0.14–0.16; Mean $|r|$ trên C là ~0.08 vs ~0.07 trên T; Split C tập trung hai drive khó 0057 và 0004 chiếm ~39% mẫu), khiến ngưỡng nonconformity $\hat{Q}$ từ C mở rộng và tạo tính bảo thủ khi áp dụng sang T. Không tính nominal p-values để tránh lỗi pseudo-replication theo D20/D87.
- **Độ phủ dải gần 0–10m (RQ3)**: Đạt 91.6% / 93.6% / 89.4% (dải 89.4%–93.6%), cao hơn mức 61%–71% ghi nhận trên Split C. Độ phủ dải gần duy trì ở mức cao nhờ ngưỡng sai số log-residual $\hat{Q}$ toàn cục kế thừa từ C bao trùm an toàn (hiện tượng bảo thủ cự ly gần mang tính exploratory). Dải xa >50m có cỡ mẫu rất nhỏ ($n \le 9$ xe) được gắn cờ `*` cảnh báo theo D3/D54.
- **Cụm cỡ mẫu nhỏ và Macro Coverage (D75)**: Cụm `drive_0002` chỉ có $n = 2$ mẫu TP. Trên `yolo11s`, cả hai mẫu đều không được bao phủ (0/2), kéo macro coverage (10 cụm) xuống 87.9%. Khi đánh giá trên 8 cụm có $n \ge 30$, macro coverage đạt 97.4%–97.5% đồng đều ở cả 3 detector.
- **So sánh Cặp Bootstrap (10 cụm drive, B=1000) (D68, D78)**:
  * Model (f) vs Model (d): CI thô loại trừ 0 ở cả 3 detector (yolo11s_640: -0.0188 (95% CI [-0.0264, -0.0107]) | yolov8s_640: -0.0191 (95% CI [-0.0259, -0.0111]) | yolov5su_640: -0.0178 (95% CI [-0.0264, -0.0076])). Sign test cấp drive xác nhận Model (f) thắng (d) ở 8–9/10 drive (p_binom <= 0.0547).
  * Model (f) vs Model (f0): CI thô loại trừ 0 ở cả 3 detector (yolo11s_640: -0.0090 (95% CI [-0.0124, -0.0064]) | yolov8s_640: -0.0090 (95% CI [-0.0125, -0.0066]) | yolov5su_640: -0.0087 (95% CI [-0.0148, -0.0034])).
  * Model (f) vs Model (e): CI thô chứa 0 ở cả 3 detector (yolo11s_640: -0.0002 (95% CI [-0.0012, 0.0023]) | yolov8s_640: +0.0002 (95% CI [-0.0008, 0.0018]) | yolov5su_640: +0.0000 (95% CI [-0.0015, 0.0022])). Không có bằng chứng thực nghiệm phân tách giữa Model (f) và Model (e) trên Split T (D78, Limitations #8).
- **Tương quan RQ2**: Hệ số tương quan hạng Spearman $\rho$ nằm trong khoảng [-0.0867, +0.1040], và toàn bộ khoảng tin cậy cluster bootstrap 95% đều chứa 0 (không phân biệt được với 0 ở mức 10 cụm). Hệ số Pearson $r$ đạt tới 0.1889 nhưng nhạy với outlier và hiệu ứng phối cảnh cự ly $Z$ ($h \propto 1/Z$ theo D21). Sai số tiếp đất $\Delta y_2$ có tương quan thực nghiệm rất nhỏ quanh 0.
