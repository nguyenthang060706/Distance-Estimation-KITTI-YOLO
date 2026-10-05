# Kết quả Đánh giá Mô hình Residual (T04, Split B OOF)

> [!IMPORTANT]
> Đánh giá theo đúng cấu hình pre-registration `prereg-residual-v1` (D25, D33, D34).
> - Mọi tuning và selection chỉ thực hiện trên Split B (D24).
> - Baseline (d) là Z_d OOF LODO theo drive (D29), không in-sample.
> - So sánh chính thực hiện trên tập chung nơi (d) hợp lệ (pattern != 000).

## 1. Bảng Tổng hợp Gate T04 & Các Phương pháp (Tập chung nơi d hợp lệ)

| Detector | Chỉ số | (d) Z_d LODO | (f0) Ridge | (f) XGBoost | (e) Direct | Δ(f − d) | 95% CI thô (12 cụm) | Gate (f < d) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolo11s_640** | **Pooled AbsRel** | 0.0607 | 0.0554 | **0.0462** | 0.0462 | -0.0144 | [-0.0205, -0.0090] | ✅ ĐẠT |
| | **Macro AbsRel** | 0.0759 | 0.0694 | **0.0541** | 0.0532 | -0.0218 | — | |
| | Macro (n≥30) | 0.0607 | 0.0509 | 0.0417 | 0.0411 | — | — | |
| **yolov8s_640** | **Pooled AbsRel** | 0.0592 | 0.0543 | **0.0467** | 0.0468 | -0.0125 | [-0.0208, -0.0042] | ✅ ĐẠT |
| | **Macro AbsRel** | 0.0674 | 0.0629 | **0.0516** | 0.0490 | -0.0158 | — | |
| | Macro (n≥30) | 0.0601 | 0.0515 | 0.0429 | 0.0421 | — | — | |
| **yolov5su_640** | **Pooled AbsRel** | 0.0613 | 0.0581 | **0.0484** | 0.0482 | -0.0129 | [-0.0215, -0.0047] | ✅ ĐẠT |
| | **Macro AbsRel** | 0.0742 | 0.0688 | **0.0519** | 0.0526 | -0.0222 | — | |
| | Macro (n≥30) | 0.0636 | 0.0556 | 0.0457 | 0.0432 | — | — | |

## 2. Kiểm định f0 ≈ f (Mô hình tuyến tính đơn giản)

| Detector | Δ AbsRel(f − f0) | 95% CI thô (12 cụm) | CI chứa 0 | |Δ| < 0.005 | Kết luận f0 ≈ f |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **yolo11s_640** | -0.0091 | [-0.0117, -0.0071] | Không | Không | **Không** |
| **yolov8s_640** | -0.0076 | [-0.0106, -0.0042] | Không | Không | **Không** |
| **yolov5su_640** | -0.0097 | [-0.0127, -0.0063] | Không | Không | **Không** |

## 3. Kiểm định Sign Test theo Drive ((f) vs (d))

| Detector | Số Drive Thắng (Wins) | Số Drive Thua (Losses) | Hòa (Ties) | p-value (one-sided binom) | Đạt p < 0.05 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **yolo11s_640** | 12 | 0 | 0 | 0.0002 | Đạt |
| **yolov8s_640** | 11 | 1 | 0 | 0.0032 | Đạt |
| **yolov5su_640** | 11 | 1 | 0 | 0.0032 | Đạt |

## 4. Dòng Phụ Đối Chiếu: Toàn bộ Mẫu Matched (N = 3.523..3.427, d + fallback Z_e)

| Detector | N matched / N GT | (d + Z_e fallback) AbsRel | (f0) Ridge AbsRel | (f) XGBoost AbsRel | (e) Direct AbsRel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **yolo11s_640** | 3523 / 4776 | 0.0615 | 0.0577 | **0.0470** | 0.0472 |
| **yolov8s_640** | 3427 / 4776 | 0.0598 | 0.0565 | **0.0474** | 0.0476 |
| **yolov5su_640** | 3496 / 4776 | 0.0620 | 0.0602 | **0.0492** | 0.0491 |

## 5. Phân rã theo Dải Cự ly (Tập chung nơi d hợp lệ)

| Detector | Dải (m) | n | Cờ low_n (*) | Baseline (d) AbsRel | Residual (f) AbsRel | Δ(f − d) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| yolo11s_640 | 0-10 | 242 |  | 0.1371 | 0.0542 | -0.0829 |
| yolo11s_640 | 10-20 | 1089 |  | 0.0606 | 0.0495 | -0.0111 |
| yolo11s_640 | 20-30 | 1225 |  | 0.0509 | 0.0443 | -0.0066 |
| yolo11s_640 | 30-50 | 924 |  | 0.0535 | 0.0429 | -0.0106 |
| yolo11s_640 | >50 | 6 | \* | 0.0848 | 0.0473 | -0.0376 |
| yolo11s_640 | >30 | 930 |  | 0.0537 | 0.0429 | -0.0107 |
| yolov8s_640 | 0-10 | 245 |  | 0.1233 | 0.0551 | -0.0682 |
| yolov8s_640 | 10-20 | 1089 |  | 0.0604 | 0.0512 | -0.0092 |
| yolov8s_640 | 20-30 | 1204 |  | 0.0474 | 0.0400 | -0.0074 |
| yolov8s_640 | 30-50 | 848 |  | 0.0558 | 0.0482 | -0.0076 |
| yolov8s_640 | >50 | 2 | \* | 0.0881 | 0.0411 | -0.0470 |
| yolov8s_640 | >30 | 850 |  | 0.0559 | 0.0482 | -0.0077 |
| yolov5su_640 | 0-10 | 240 |  | 0.1244 | 0.0521 | -0.0723 |
| yolov5su_640 | 10-20 | 1083 |  | 0.0626 | 0.0514 | -0.0112 |
| yolov5su_640 | 20-30 | 1198 |  | 0.0501 | 0.0444 | -0.0056 |
| yolov5su_640 | 30-50 | 931 |  | 0.0579 | 0.0489 | -0.0090 |
| yolov5su_640 | >50 | 4 | \* | 0.0949 | 0.0983 | +0.0034 |
| yolov5su_640 | >30 | 935 |  | 0.0580 | 0.0491 | -0.0090 |
