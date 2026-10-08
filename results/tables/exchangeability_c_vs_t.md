# Kiểm Định Tính Khả Hoán $C \leftrightarrow T$ (Exchangeability Diagnostics)

> **Cơ Sở Lý Thuyết & Giải Trình Hiện Tượng Over-coverage Bảo Thủ Ngoài Mẫu (Decision D79):**
> - **Lý thuyết Conformal:** Giả định tính khả hoán (exchangeability) giữa tập hiệu chuẩn (Calibration set - Split C) và tập kiểm định (Test set - Split T) là điều kiện tiên quyết để bảo đảm độ phủ danh nghĩa $1 - \alpha = 90.0\%$.
> - **Hiện tượng thực nghiệm trên Split T:** Standard CQR đạt độ phủ thực nghiệm **$96.4\% - 97.1\%$** (vượt mức danh nghĩa $+6.4$ đến $+7.1$ điểm phần trăm).
> - **Giải trình định lượng từ Kiểm định 2-mẫu Kolmogorov-Smirnov (KS-test):**
>   * Split C khó hơn Split T đáng kể: sai số AbsRel(d) trên C là 0.0862 vs 0.0640 trên T (KS stat = 0.1573, $p = 4.65 \times 10^{-21}$); đồng thời Split C tập trung 2 cụm lệch khó (`drive_0057` và `drive_0004` chiếm 39.3% mẫu).
>   * Hiệu chuẩn trên C khiến ngưỡng nonconformity $\hat{Q}$ nở rộng để bao phủ các cụm khó. Khi áp sang Split T (gồm các cụm đường rộng, ít rung lắc viền hơn), khoảng tin cậy trở nên bảo thủ (+6–7% độ phủ).
>   * Bảng dưới đây đối chiếu phân bố của 7 biến quan sát test-time giữa Split C và Split T.

## Detector `yolo11s_640`

| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Statistic | $p$-value | Lệch có ý nghĩa ($p < 0.05$)? |
|---|---|---|---|---|---|---|---|---|:---:|
| **Độ sâu dự đoán Ẑ (m)** | 1,489 | 2,712 | 24.690 | 23.901 | 10.476 | 10.717 | **0.0543** | 0.0065 | ⚠️ CÓ |
| **Độ sâu thực tế Z_gt (m)** | 1,489 | 2,712 | 24.368 | 23.830 | 10.309 | 10.530 | **0.0485** | 0.0210 | ⚠️ CÓ |
| **Độ tin cậy detector (confidence)** | 1,489 | 2,712 | 0.900 | 0.901 | 0.043 | 0.044 | **0.0480** | 0.0230 | ⚠️ CÓ |
| **Cờ hợp lệ Cue Chiều rộng (valid_w)** | 1,489 | 2,712 | 0.956 | 0.957 | 0.204 | 0.204 | **0.0001** | 1.0000 | Không |
| **Cờ hợp lệ Cue Chiều cao (valid_h)** | 1,489 | 2,712 | 0.944 | 0.959 | 0.231 | 0.199 | **0.0151** | 0.9776 | Không |
| **Cờ hợp lệ Cue Cạnh dưới (valid_g)** | 1,489 | 2,712 | 0.944 | 0.959 | 0.231 | 0.199 | **0.0151** | 0.9776 | Không |
| **Sai số log-residual |r| = |ln Z_gt - ln Z_base|** | 1,489 | 2,712 | 0.085 | 0.067 | 0.068 | 0.067 | **0.1536** | 3.03e-20 | ⚠️ CÓ |

## Detector `yolov8s_640`

| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Statistic | $p$-value | Lệch có ý nghĩa ($p < 0.05$)? |
|---|---|---|---|---|---|---|---|---|:---:|
| **Độ sâu dự đoán Ẑ (m)** | 1,445 | 2,660 | 24.180 | 23.536 | 10.240 | 10.643 | **0.0548** | 0.0069 | ⚠️ CÓ |
| **Độ sâu thực tế Z_gt (m)** | 1,445 | 2,660 | 23.875 | 23.462 | 9.986 | 10.343 | **0.0536** | 0.0088 | ⚠️ CÓ |
| **Độ tin cậy detector (confidence)** | 1,445 | 2,660 | 0.910 | 0.911 | 0.033 | 0.033 | **0.0545** | 0.0073 | ⚠️ CÓ |
| **Cờ hợp lệ Cue Chiều rộng (valid_w)** | 1,445 | 2,660 | 0.956 | 0.958 | 0.206 | 0.200 | **0.0026** | 1.0000 | Không |
| **Cờ hợp lệ Cue Chiều cao (valid_h)** | 1,445 | 2,660 | 0.945 | 0.959 | 0.229 | 0.197 | **0.0148** | 0.9845 | Không |
| **Cờ hợp lệ Cue Cạnh dưới (valid_g)** | 1,445 | 2,660 | 0.945 | 0.959 | 0.229 | 0.197 | **0.0148** | 0.9845 | Không |
| **Sai số log-residual |r| = |ln Z_gt - ln Z_base|** | 1,445 | 2,660 | 0.083 | 0.067 | 0.064 | 0.068 | **0.1621** | 6.02e-22 | ⚠️ CÓ |

## Detector `yolov5su_640`

| Đặc trưng quan sát | $N_C$ | $N_T$ | Mean C | Mean T | Std C | Std T | KS Statistic | $p$-value | Lệch có ý nghĩa ($p < 0.05$)? |
|---|---|---|---|---|---|---|---|---|:---:|
| **Độ sâu dự đoán Ẑ (m)** | 1,430 | 2,674 | 24.165 | 23.613 | 10.084 | 10.559 | **0.0535** | 0.0092 | ⚠️ CÓ |
| **Độ sâu thực tế Z_gt (m)** | 1,430 | 2,674 | 23.878 | 23.632 | 10.024 | 10.507 | **0.0507** | 0.0160 | ⚠️ CÓ |
| **Độ tin cậy detector (confidence)** | 1,430 | 2,674 | 0.903 | 0.904 | 0.039 | 0.039 | **0.0446** | 0.0470 | ⚠️ CÓ |
| **Cờ hợp lệ Cue Chiều rộng (valid_w)** | 1,430 | 2,674 | 0.952 | 0.957 | 0.213 | 0.202 | **0.0049** | 1.0000 | Không |
| **Cờ hợp lệ Cue Chiều cao (valid_h)** | 1,430 | 2,674 | 0.948 | 0.960 | 0.223 | 0.195 | **0.0128** | 0.9974 | Không |
| **Cờ hợp lệ Cue Cạnh dưới (valid_g)** | 1,430 | 2,674 | 0.948 | 0.960 | 0.223 | 0.195 | **0.0128** | 0.9974 | Không |
| **Sai số log-residual |r| = |ln Z_gt - ln Z_base|** | 1,430 | 2,674 | 0.081 | 0.067 | 0.065 | 0.069 | **0.1427** | 5.22e-17 | ⚠️ CÓ |
