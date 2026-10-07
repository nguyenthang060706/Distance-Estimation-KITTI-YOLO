# Đánh giá Độ ổn định Conformal qua 20 Lần chia lại B∪C (T08)

> Đánh giá out-of-sample trên 20 phân hoạch ngẫu nhiên (seed 0–19) của 22 cụm drive có Car Hard.
> Tỷ lệ phân hoạch: fit ≈ 50%, calib ≈ 25%, eval ≈ 25% (mỗi phần >= 4 cụm, top1_share <= 0.50).
> Tuân thủ Decisions D24, D26, D46, D47, D48, D50, D51, D52, D53, D54.

---

## 1. Phát hiện chính: Under-coverage hệ thống ngoài mẫu (RQ3 / H3)

Mọi phương án và detector trên tập eval giữ lại đều **dưới mức danh nghĩa 90%**:
- **Pooled coverage:** 83.8% – 87.4%
- **Macro coverage theo drive:** 77.9% – 84.5% (thấp hơn pooled 3–7 điểm %, cho thấy các drive nhỏ bị under-cover nặng hơn)
- **Khoảng dao động per-seed:** 63.6% – 99.6%
- **Độ lệch chuẩn giữa các seed:** $\sigma \approx 7.4\% – 9.0\%$ (phản ánh độ nhạy với cách chia, không phải khoảng tin cậy CI).

*Ghi chú về tính bi quan của resplit:* Tập calibration trong mỗi seed chỉ có khoảng 5–6 drive (≈ 25% số xe, $n_{\text{calib}} \approx 1100\text{--}1800$), trong khi cấu hình chuẩn ở T07 được calibration trên toàn bộ Split C (10 drive, $n=1489$). Do đó, kết quả 20 resplits là một ước lượng bi quan hơn so với LODO trên C của T07 (T07 đạt 87.1% – 88.0% pooled, và 90.0% – 90.1% macro trên các drive $n \ge 30$).

---

## 2. Bảng tổng hợp Mean ± Std qua 20 Resplits

> Ràng buộc D52: Chênh lệch giữa 3 phương án ($\le 3$ điểm phần trăm) nhỏ hơn nhiều so với độ lệch chuẩn ($7\text{--}9$ điểm phần trăm), do đó không xếp hạng giữa Split Conformal, Standard CQR và Mondrian CQR.

| Detector | Phương án | Mean Pooled Cov ± Std | [Min, Max] Cov | Mean Macro Cov ± Std | Mean Width ($Z_{hi}/Z_{lo}$) | Mean Winkler ($r$-space) | Crossings |
|---|---|---|---|---|---|---|---|
| `yolo11s_640` | Split Conformal | 85.56% ± 9.03% | [63.62%, 98.51%] | 80.17% ± 10.20% | 1.241 | 0.3834 | 0 |
| `yolo11s_640` | Standard CQR | 85.15% ± 8.62% | [69.80%, 98.75%] | 77.91% ± 11.18% | 1.247 | 0.3366 | 0 |
| `yolo11s_640` | Mondrian CQR | 84.53% ± 8.76% | [67.56%, 97.43%] | 77.89% ± 11.33% | 1.247 | 0.3287 | 0 |
| `yolov8s_640` | Split Conformal | 87.41% ± 7.41% | [69.25%, 98.50%] | 84.48% ± 8.83% | 1.266 | 0.3772 | 0 |
| `yolov8s_640` | Standard CQR | 86.14% ± 8.67% | [69.59%, 99.56%] | 82.94% ± 9.47% | 1.256 | 0.3764 | 0 |
| `yolov8s_640` | Mondrian CQR | 84.49% ± 7.83% | [71.41%, 98.56%] | 81.28% ± 8.75% | 1.255 | 0.3739 | 0 |
| `yolov5su_640` | Split Conformal | 85.85% ± 8.11% | [68.35%, 98.51%] | 82.96% ± 10.02% | 1.265 | 0.3915 | 0 |
| `yolov5su_640` | Standard CQR | 84.99% ± 8.37% | [68.97%, 97.91%] | 80.21% ± 10.16% | 1.264 | 0.3545 | 0 |
| `yolov5su_640` | Mondrian CQR | 83.79% ± 8.06% | [69.04%, 97.49%] | 79.82% ± 10.44% | 1.264 | 0.3474 | 0 |

---

## 3. Bảng so sánh cặp theo seed (Paired Differences qua 20 Seeds)

| Detector | So sánh cặp | Mean $\Delta$ | Std $\Delta$ | Khoảng $[\text{Min}, \text{Max}] \Delta$ |
|---|---|---|---|---|
| `yolo11s_640` | $\Delta(\text{CQR} - \text{Split})$ | -0.41% | 4.06% | [-7.41%, +11.63%] |
| `yolo11s_640` | $\Delta(\text{Mondrian} - \text{CQR})$ | -0.62% | 1.74% | [-4.95%, +1.48%] |
| `yolo11s_640` | $\Delta(\text{Mondrian} - \text{Split})$ | -1.03% | 4.68% | [-10.32%, +11.91%] |
| `yolov8s_640` | $\Delta(\text{CQR} - \text{Split})$ | -1.27% | 3.20% | [-9.91%, +3.91%] |
| `yolov8s_640` | $\Delta(\text{Mondrian} - \text{CQR})$ | -1.65% | 2.16% | [-6.49%, +2.19%] |
| `yolov8s_640` | $\Delta(\text{Mondrian} - \text{Split})$ | -2.92% | 2.78% | [-9.12%, +3.18%] |
| `yolov5su_640` | $\Delta(\text{CQR} - \text{Split})$ | -0.86% | 2.52% | [-6.50%, +4.35%] |
| `yolov5su_640` | $\Delta(\text{Mondrian} - \text{CQR})$ | -1.20% | 3.50% | [-9.35%, +5.57%] |
| `yolov5su_640` | $\Delta(\text{Mondrian} - \text{Split})$ | -2.06% | 3.92% | [-12.61%, +3.85%] |

---

## 4. Bằng chứng định lượng về ảnh hưởng của Cụm Heterogeneity (`0057` & `0004`)

Phân tích trên `yolo11s_640` theo sự hiện diện của 2 cụm có under-cover đặc thù trong tập `eval`:

| Nhóm vị trí của `0057` và `0004` | Số seed | CQR Cov trung bình | Khoảng CQR Cov | Split Cov trung bình | Mondrian Cov trung bình |
|---|---|---|---|---|---|
| **Cả 2 cụm nằm trong `eval`** | 5 (seeds 2, 3, 11, 14, 17) | **72.38%** | [69.80%, 75.25%] | 72.85% | 71.76% |
| **Chỉ 1 trong 2 cụm nằm trong `eval`** | 11 (seeds 0, 1, 6, 7, 8, 9, 10, 12, 13, 18, 19) | **87.58%** | [80.11%, 93.08%] | 87.89% | 87.05% |
| **Không có cụm nào nằm trong `eval`** | 4 (seeds 4, 5, 15, 16) | **94.44%** | [89.93%, 98.75%] | 94.14% | 93.24% |

> **Nhận xét nhân quả:** Độ phủ ngoài mẫu phân hóa thành 3 bậc rõ rệt phụ thuộc trực tiếp vào việc hai cụm lệch `0057` và `0004` rơi vào tập nào ($72.4\% \to 87.6\% \to 94.4\%$). Điều này cung cấp bằng chứng định lượng vững chắc cho Quyết định D50: drive-level cluster heterogeneity là nguồn gốc chính tạo ra dao động độ phủ ngoài mẫu.

---

## 5. Chi tiết 20 Giá trị Coverage cho từng Seed (Bao gồm chẩn đoán cụm)

- **Tần suất bin Mondrian gộp ($n < 50$):** 0 / 20 seed (0.0%). Cả 20 seed đều giữ nguyên 4 bins `[0,10)`, `[10,20)`, `[20,30)`, `[30,∞)` do cỡ mẫu calibration $n_{\text{calib}} \ge 1000$ đủ lớn.

### Detector `yolo11s_640`

| Seed | Redraws | $k_{\text{calib}}$ | $k_{\text{eval}}$ | $\text{top1}_{\text{eval}}$ | Có `0057`? | Có `0004`? | $n_{\text{eval}}$ | Split Conformal | Standard CQR | Mondrian CQR | $\hat{Q}_{\text{cqr}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 9 | 6 | 0.497 | Có | Không | 1408 | 88.99% | 85.01% | 85.09% | +0.0147 |
| 1 | 5 | 6 | 8 | 0.277 | Có | Không | 1143 | 87.58% | 87.84% | 88.89% | +0.0055 |
| 2 | 11 | 9 | 6 | 0.494 | Có | Có | 1424 | 69.03% | 69.80% | 67.56% | +0.0075 |
| 3 | 14 | 6 | 7 | 0.302 | Có | Có | 1410 | 63.62% | 75.25% | 75.53% | +0.0109 |
| 4 | 1 | 6 | 9 | 0.453 | Không | Không | 1500 | 89.93% | 89.93% | 86.80% | +0.0115 |
| 5 | 6 | 6 | 10 | 0.407 | Không | Không | 1094 | 96.25% | 93.97% | 94.97% | +0.0186 |
| 6 | 1 | 9 | 7 | 0.491 | Không | Có | 1426 | 84.43% | 84.78% | 83.66% | +0.0066 |
| 7 | 30 | 6 | 6 | 0.397 | Có | Không | 1049 | 85.99% | 88.37% | 88.66% | +0.0464 |
| 8 | 8 | 4 | 7 | 0.339 | Có | Không | 1410 | 91.91% | 91.63% | 92.70% | +0.0425 |
| 9 | 0 | 5 | 7 | 0.310 | Không | Có | 1378 | 82.95% | 84.83% | 84.62% | +0.0053 |
| 10 | 4 | 6 | 7 | 0.499 | Không | Có | 1880 | 81.49% | 80.11% | 77.34% | -0.0045 |
| 11 | 6 | 7 | 8 | 0.357 | Có | Có | 1376 | 80.81% | 73.40% | 70.49% | +0.0110 |
| 12 | 49 | 9 | 9 | 0.326 | Có | Không | 1446 | 91.70% | 93.08% | 93.50% | +0.0347 |
| 13 | 9 | 9 | 5 | 0.491 | Không | Có | 1434 | 94.91% | 92.12% | 87.17% | +0.0529 |
| 14 | 19 | 7 | 8 | 0.306 | Có | Có | 1468 | 80.38% | 73.02% | 73.50% | +0.0091 |
| 15 | 14 | 6 | 5 | 0.327 | Không | Không | 1267 | 91.87% | 95.11% | 93.76% | +0.0299 |
| 16 | 0 | 8 | 6 | 0.430 | Không | Không | 1674 | 98.51% | 98.75% | 97.43% | +0.0682 |
| 17 | 8 | 5 | 7 | 0.339 | Có | Có | 1383 | 74.40% | 70.43% | 71.73% | +0.0152 |
| 18 | 1 | 6 | 11 | 0.248 | Không | Có | 1214 | 79.90% | 82.54% | 84.02% | +0.0077 |
| 19 | 10 | 8 | 5 | 0.454 | Không | Có | 1574 | 96.63% | 93.07% | 93.20% | +0.0507 |

*(Số liệu của `yolov8s_640` và `yolov5su_640` có xu hướng bậc 3 tương đồng hoàn toàn, xem đầy đủ trong file JSON).*
