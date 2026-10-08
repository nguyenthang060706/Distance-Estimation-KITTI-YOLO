# Chẩn Đoán Tính Khả Hoán $C \leftrightarrow T$ (Exchangeability Diagnostics)

> **Cơ Sở Lý Thuyết & Bản Chất Post-hoc / Exploratory của Hiện Tượng Over-coverage (Decisions D68, D79, D87):**
> - **Đối chiếu tiên đoán Tiền đăng ký (D68):** Ban đầu, D68 dự báo nguy cơ *under-coverage* ngoài mẫu do phân tích độ ổn định 20 resplits (T08) chỉ đạt 84–87%. Tuy nhiên, kết quả thực tế trên Split T đạt độ phủ danh nghĩa vượt mức: **96.4%–97.1%** (over-coverage).
> - **Tính chất Diễn giải Hậu nghiệm (Post-hoc / Exploratory):** Giả thuyết *'Split C có độ khó cao hơn Split T khiến ngưỡng sai số không tương đồng $\hat{Q}$ bị nới rộng, dẫn đến bảo thủ ngoài mẫu'* là suy luận post-hoc được hình thành sau khi quan sát dữ liệu Split T, không phải kiểm chứng tiên nghiệm. Thông điệp phương pháp luận chính của RQ3 là: *Độ phủ biên không chuyển giao ổn định giữa các cụm khi số lượng cụm drive còn nhỏ (~10 cụm); hướng lệch (under hay over) phụ thuộc vào thành phần drive của tập hiệu chuẩn C so với tập kiểm định T.*
> - **Cảnh báo Phương pháp luận về Tránh Lỗi Pseudo-replication (AGENT_RULES §6.2, Decisions D20, D73):** Bounding box trong KITTI gom theo các cụm driving sequence có tương quan chuỗi mạnh. Việc tính p-value giả định các hàng độc lập (i.i.d) tạo ra p-value ngụy tạo ($p \approx 10^{-20}$). Do đó, bảng bên dưới chỉ báo cáo chỉ số thống kê Kolmogorov-Smirnov $D_{\text{KS}} = \sup |F_C(x) - F_T(x)|$ mang tính **mô tả phân kỳ phân bố thực nghiệm (descriptive empirical divergence)**; đối với các cờ nhị phân (`valid_*`), chỉ báo cáo tỷ lệ trung bình (mean proportion).

## Detector `yolo11s_640`

| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Stat (mô tả) | Ghi chú diễn giải |
|---|---|---|---|---|---|---|---|---|
| **Độ sâu dự đoán Ẑ (m)** | 1,489 | 2,712 | 24.690 | 23.901 | 10.476 | 10.717 | **0.0543** | KS mô tả (hướng: C > T) |
| **Độ sâu thực tế Z_gt (m)** | 1,489 | 2,712 | 24.368 | 23.830 | 10.309 | 10.530 | **0.0485** | KS mô tả (hướng: C > T) |
| **Độ tin cậy detector (confidence)** | 1,489 | 2,712 | 0.900 | 0.901 | 0.043 | 0.044 | **0.0480** | KS mô tả (hướng: T > C) |
| **Cờ hợp lệ Cue Chiều rộng (valid_w)** | 1,489 | 2,712 | 0.956 | 0.957 | 0.204 | 0.204 | --- | Biến nhị phân (chênh lệch 0.01%) |
| **Cờ hợp lệ Cue Chiều cao (valid_h)** | 1,489 | 2,712 | 0.944 | 0.959 | 0.231 | 0.199 | --- | Biến nhị phân (chênh lệch 1.51%) |
| **Cờ hợp lệ Cue Cạnh dưới (valid_g)** | 1,489 | 2,712 | 0.944 | 0.959 | 0.231 | 0.199 | --- | Biến nhị phân (chênh lệch 1.51%) |
| **Sai số log-residual |r| = |ln Z_gt - ln Z_base|** | 1,489 | 2,712 | 0.085 | 0.067 | 0.068 | 0.067 | **0.1536** | KS mô tả (hướng: C > T) |

## Detector `yolov8s_640`

| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Stat (mô tả) | Ghi chú diễn giải |
|---|---|---|---|---|---|---|---|---|
| **Độ sâu dự đoán Ẑ (m)** | 1,445 | 2,660 | 24.180 | 23.536 | 10.240 | 10.643 | **0.0548** | KS mô tả (hướng: C > T) |
| **Độ sâu thực tế Z_gt (m)** | 1,445 | 2,660 | 23.875 | 23.462 | 9.986 | 10.343 | **0.0536** | KS mô tả (hướng: C > T) |
| **Độ tin cậy detector (confidence)** | 1,445 | 2,660 | 0.910 | 0.911 | 0.033 | 0.033 | **0.0545** | KS mô tả (hướng: T > C) |
| **Cờ hợp lệ Cue Chiều rộng (valid_w)** | 1,445 | 2,660 | 0.956 | 0.958 | 0.206 | 0.200 | --- | Biến nhị phân (chênh lệch 0.26%) |
| **Cờ hợp lệ Cue Chiều cao (valid_h)** | 1,445 | 2,660 | 0.945 | 0.959 | 0.229 | 0.197 | --- | Biến nhị phân (chênh lệch 1.48%) |
| **Cờ hợp lệ Cue Cạnh dưới (valid_g)** | 1,445 | 2,660 | 0.945 | 0.959 | 0.229 | 0.197 | --- | Biến nhị phân (chênh lệch 1.48%) |
| **Sai số log-residual |r| = |ln Z_gt - ln Z_base|** | 1,445 | 2,660 | 0.083 | 0.067 | 0.064 | 0.068 | **0.1621** | KS mô tả (hướng: C > T) |

## Detector `yolov5su_640`

| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Stat (mô tả) | Ghi chú diễn giải |
|---|---|---|---|---|---|---|---|---|
| **Độ sâu dự đoán Ẑ (m)** | 1,430 | 2,674 | 24.165 | 23.613 | 10.084 | 10.559 | **0.0535** | KS mô tả (hướng: C > T) |
| **Độ sâu thực tế Z_gt (m)** | 1,430 | 2,674 | 23.878 | 23.632 | 10.024 | 10.507 | **0.0507** | KS mô tả (hướng: C > T) |
| **Độ tin cậy detector (confidence)** | 1,430 | 2,674 | 0.903 | 0.904 | 0.039 | 0.039 | **0.0446** | KS mô tả (hướng: T > C) |
| **Cờ hợp lệ Cue Chiều rộng (valid_w)** | 1,430 | 2,674 | 0.952 | 0.957 | 0.213 | 0.202 | --- | Biến nhị phân (chênh lệch 0.49%) |
| **Cờ hợp lệ Cue Chiều cao (valid_h)** | 1,430 | 2,674 | 0.948 | 0.960 | 0.223 | 0.195 | --- | Biến nhị phân (chênh lệch 1.28%) |
| **Cờ hợp lệ Cue Cạnh dưới (valid_g)** | 1,430 | 2,674 | 0.948 | 0.960 | 0.223 | 0.195 | --- | Biến nhị phân (chênh lệch 1.28%) |
| **Sai số log-residual |r| = |ln Z_gt - ln Z_base|** | 1,430 | 2,674 | 0.081 | 0.067 | 0.065 | 0.069 | **0.1427** | KS mô tả (hướng: C > T) |
