# Đánh giá Độ Phủ Có Điều Kiện trên Split T (Tác vụ T15)

> **Bối cảnh phương pháp luận (Decisions D19, D47, D50, D54, D55, D70, D74, D75, D79):**
> - **Zero-Touch Split T (D70):** Toàn bộ phân tích đọc trực tiếp từ các artifact nghiệm thu tĩnh tại `results/final/`.
> - **Standard CQR là phương án chính tiên nghiệm (D55):** Split Conformal và Mondrian CQR đóng vai trò baseline đối chứng.
> - **Minh bạch thiên lệch kẻ sống sót (D32, D70):** Mọi bảng phân rã đều công bố song song $n_{\text{TP}}, n_{\text{FN}}, n_{\text{GT}}$ và Recall.
> - **Cảnh báo cỡ mẫu nhỏ (D54):** Gắn cờ `*` khi $n_{\text{TP}} < 100$ và ghi rõ số cụm drive $k$.
> - **Mức ý nghĩa danh nghĩa:** $\alpha = 0.10$ (độ phủ danh nghĩa $90.0\%$).

## 1. Detector `yolo11s_640`

> - **Bootstrap 95% CI Pooled Coverage (10 cụm):**
>   * Standard CQR: [94.72%, 98.01%] (Mean Width: [1.296, 1.343]) — *CI thô (10 cụm)*
>   * Split Conformal: [94.52%, 97.99%] (Mean Width: [1.322, 1.322]) — *CI thô (10 cụm)*
>   * Mondrian CQR: [93.47%, 98.01%] (Mean Width: [1.280, 1.313]) — *CI thô (10 cụm)*

### 1.1 Dải cự ly theo Ẑ dự đoán (Prospective Distance Bins)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0-10` | 256 | --- | --- | --- | **91.8%** | 1.429 | 0.4149 | 89.5% | 1.322 | 94.1% | 1.473 | 8 |  |
| `10-20` | 835 | --- | --- | --- | **97.7%** | 1.319 | 0.2880 | 98.1% | 1.322 | 95.1% | 1.242 | 9 |  |
| `20-30` | 801 | --- | --- | --- | **96.8%** | 1.311 | 0.2969 | 96.1% | 1.322 | 94.8% | 1.259 | 8 |  |
| `30-50` | 815 | --- | --- | --- | **96.3%** | 1.293 | 0.2748 | 97.2% | 1.322 | 97.4% | 1.323 | 8 |  |
| `>50` | 5 | --- | --- | --- | **60.0%** | 1.381 | 0.5174 | 40.0% | 1.322 | 60.0% | 1.414 | 2 | * |
| `>=30 (grouped)` | 820 | --- | --- | --- | **96.1%** | 1.293 | 0.2763 | 96.8% | 1.322 | 97.2% | 1.324 | 8 |  |

### 1.2 Dải cự ly theo Z thật (Retrospective Distance Bins — Chẩn đoán, v4 §6)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0-10` | 261 | 14 | 275 | 94.9% | **91.6%** | 1.428 | 0.4116 | 88.9% | 1.322 | 92.7% | 1.464 | 8 |  |
| `10-20` | 831 | 65 | 896 | 92.8% | **97.6%** | 1.320 | 0.2881 | 98.0% | 1.322 | 95.1% | 1.245 | 9 |  |
| `20-30` | 804 | 107 | 911 | 88.2% | **97.3%** | 1.315 | 0.2942 | 96.5% | 1.322 | 95.7% | 1.267 | 8 |  |
| `30-50` | 809 | 288 | 1,097 | 73.8% | **96.2%** | 1.289 | 0.2762 | 97.0% | 1.322 | 97.2% | 1.315 | 8 |  |
| `>50` | 7 | 26 | 33 | 21.2% | **57.1%** | 1.317 | 0.6059 | 71.4% | 1.322 | 57.1% | 1.348 | 3 | * |
| `>=30 (grouped)` | 816 | 314 | 1,130 | 72.2% | **95.8%** | 1.289 | 0.2790 | 96.8% | 1.322 | 96.8% | 1.315 | 8 |  |

### 1.3 Mức độ cắt biên (Truncation) & Chạm viền ảnh (Touch Edges, D84)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `No Truncation (0.0)` | 2,516 | 483 | 2,999 | 83.9% | **96.9%** | 1.305 | 0.2841 | 97.3% | 1.322 | 96.0% | 1.276 | 10 |  |
| `Mild (0.0 < t <= 0.15)` | 71 | 5 | 76 | 93.4% | **91.5%** | 1.468 | 0.4240 | 88.7% | 1.322 | 95.8% | 1.485 | 7 | * |
| `Moderate/Severe (0.15 < t <= 0.50)` | 125 | 12 | 137 | 91.2% | **88.0%** | 1.521 | 0.5289 | 80.0% | 1.322 | 86.4% | 1.536 | 8 |  |
| `No Edge Touch (Pattern 111)` | 2,518 | --- | --- | --- | **97.0%** | 1.305 | 0.2830 | 97.4% | 1.322 | 96.1% | 1.276 | 10 |  |
| `Touch Horizontal (valid_w=0)` | 82 | --- | --- | --- | **87.8%** | 1.408 | 0.4175 | 80.5% | 1.322 | 85.4% | 1.378 | 7 | * |
| `Touch Vertical (valid_h=0 or valid_g=0)` | 76 | --- | --- | --- | **93.4%** | 1.587 | 0.5234 | 92.1% | 1.322 | 96.0% | 1.636 | 8 | * |
| `Touch Multi-edge (valid_w=0 and (valid_h=0 or valid_g=0))` | 36 | --- | --- | --- | **77.8%** | 1.572 | 0.6823 | 63.9% | 1.322 | 80.6% | 1.620 | 6 | * |

### 1.4 Mức độ che khuất (Occlusion Levels)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0 (Fully visible)` | 1,624 | 65 | 1,689 | 96.2% | **96.6%** | 1.304 | 0.2886 | 97.0% | 1.322 | 96.0% | 1.284 | 10 |  |
| `1 (Partly occluded)` | 790 | 231 | 1,021 | 77.4% | **96.0%** | 1.335 | 0.3088 | 95.1% | 1.322 | 94.6% | 1.306 | 8 |  |
| `2 (Largely occluded)` | 298 | 204 | 502 | 59.4% | **96.3%** | 1.365 | 0.3303 | 95.6% | 1.322 | 95.6% | 1.314 | 8 |  |

### 1.5 Góc hướng quan sát θ (Viewing Angle Bins, D19)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Side (<30°)` | 362 | 99 | 461 | 78.5% | **93.7%** | 1.363 | 0.3473 | 93.1% | 1.322 | 92.8% | 1.304 | 5 |  |
| `Diagonal (30–60°)` | 249 | 81 | 330 | 75.4% | **96.0%** | 1.375 | 0.3501 | 92.8% | 1.322 | 94.8% | 1.360 | 8 |  |
| `Front/Rear (>60°)` | 2,101 | 320 | 2,421 | 86.8% | **96.9%** | 1.305 | 0.2847 | 97.3% | 1.322 | 96.1% | 1.284 | 9 |  |

### 1.6 Mức độ khó KITTI (Nested & Disjoint Difficulty)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Easy (nested)` | 1,004 | 20 | 1,024 | 98.0% | **98.1%** | 1.298 | 0.2759 | 98.5% | 1.322 | 96.6% | 1.249 | 9 |  |
| `Moderate (nested)` | 2,338 | 291 | 2,629 | 88.9% | **96.8%** | 1.307 | 0.2862 | 97.2% | 1.322 | 96.0% | 1.283 | 10 |  |
| `Hard (nested)` | 2,712 | 500 | 3,212 | 84.4% | **96.4%** | 1.319 | 0.2991 | 96.3% | 1.322 | 95.5% | 1.294 | 10 |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | :---: |
| `Easy (disjoint)` | 1,004 | 20 | 1,024 | 98.0% | **98.1%** | 1.298 | 0.2759 | 98.5% | 1.322 | 96.6% | 1.249 | 9 |  |
| `Moderate (disjoint)` | 1,334 | 271 | 1,605 | 83.1% | **95.9%** | 1.314 | 0.2940 | 96.2% | 1.322 | 95.5% | 1.308 | 10 |  |
| `Hard (disjoint)` | 374 | 209 | 583 | 64.1% | **93.6%** | 1.396 | 0.3793 | 90.6% | 1.322 | 92.8% | 1.359 | 9 |  |

### 1.7 Nhóm Fallback Pattern 000 (Decision D74)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Fallback (Pattern 000)` | 36 | --- | --- | --- | **77.8%** | 1.572 | 0.6823 | 63.9% | 1.322 | 80.6% | 1.620 | 6 | * |
| `Fused Geometric (>=1 cue)` | 2,676 | --- | --- | --- | **96.6%** | 1.316 | 0.2939 | 96.8% | 1.322 | 95.7% | 1.289 | 10 |  |

## 1. Detector `yolov8s_640`

> - **Bootstrap 95% CI Pooled Coverage (10 cụm):**
>   * Standard CQR: [95.28%, 98.70%] (Mean Width: [1.325, 1.374]) — *CI thô (10 cụm)*
>   * Split Conformal: [94.94%, 98.46%] (Mean Width: [1.345, 1.345]) — *CI thô (10 cụm)*
>   * Mondrian CQR: [95.22%, 98.58%] (Mean Width: [1.324, 1.376]) — *CI thô (10 cụm)*

### 1.1 Dải cự ly theo Ẑ dự đoán (Prospective Distance Bins)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0-10` | 273 | --- | --- | --- | **93.4%** | 1.447 | 0.4140 | 91.6% | 1.345 | 99.6% | 1.816 | 8 |  |
| `10-20` | 845 | --- | --- | --- | **98.2%** | 1.356 | 0.3117 | 97.8% | 1.345 | 96.0% | 1.268 | 9 |  |
| `20-30` | 754 | --- | --- | --- | **97.8%** | 1.328 | 0.2940 | 98.0% | 1.345 | 96.4% | 1.261 | 8 |  |
| `30-50` | 784 | --- | --- | --- | **97.1%** | 1.339 | 0.3003 | 97.2% | 1.345 | 98.1% | 1.372 | 8 |  |
| `>50` | 4 | --- | --- | --- | **25.0%** | 1.393 | 0.9782 | 25.0% | 1.345 | 25.0% | 1.427 | 2 | * |
| `>=30 (grouped)` | 788 | --- | --- | --- | **96.7%** | 1.339 | 0.3038 | 96.8% | 1.345 | 97.7% | 1.372 | 8 |  |

### 1.2 Dải cự ly theo Z thật (Retrospective Distance Bins — Chẩn đoán, v4 §6)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0-10` | 266 | 9 | 275 | 96.7% | **93.6%** | 1.452 | 0.4161 | 89.8% | 1.345 | 97.7% | 1.796 | 8 |  |
| `10-20` | 836 | 60 | 896 | 93.3% | **97.9%** | 1.356 | 0.3116 | 98.1% | 1.345 | 96.4% | 1.280 | 9 |  |
| `20-30` | 780 | 131 | 911 | 85.6% | **98.0%** | 1.332 | 0.2942 | 97.8% | 1.345 | 97.0% | 1.271 | 8 |  |
| `30-50` | 776 | 321 | 1,097 | 70.7% | **96.8%** | 1.336 | 0.3044 | 97.2% | 1.345 | 97.3% | 1.363 | 8 |  |
| `>50` | 2 | 31 | 33 | 6.1% | **100.0%** | 1.346 | 0.2964 | 100.0% | 1.345 | 100.0% | 1.379 | 1 | * |
| `>=30 (grouped)` | 778 | 352 | 1,130 | 68.8% | **96.8%** | 1.336 | 0.3044 | 97.2% | 1.345 | 97.3% | 1.363 | 8 |  |

### 1.3 Mức độ cắt biên (Truncation) & Chạm viền ảnh (Touch Edges, D84)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `No Truncation (0.0)` | 2,463 | 536 | 2,999 | 82.1% | **97.5%** | 1.339 | 0.3024 | 97.9% | 1.345 | 97.1% | 1.319 | 10 |  |
| `Mild (0.0 < t <= 0.15)` | 72 | 4 | 76 | 94.7% | **88.9%** | 1.464 | 0.5010 | 87.5% | 1.345 | 97.2% | 1.717 | 7 | * |
| `Moderate/Severe (0.15 < t <= 0.50)` | 125 | 12 | 137 | 91.2% | **94.4%** | 1.545 | 0.4534 | 83.2% | 1.345 | 95.2% | 1.809 | 8 |  |
| `No Edge Touch (Pattern 111)` | 2,477 | --- | --- | --- | **97.6%** | 1.339 | 0.3014 | 97.9% | 1.345 | 97.1% | 1.321 | 10 |  |
| `Touch Horizontal (valid_w=0)` | 75 | --- | --- | --- | **92.0%** | 1.405 | 0.3777 | 80.0% | 1.345 | 89.3% | 1.459 | 7 | * |
| `Touch Vertical (valid_h=0 or valid_g=0)` | 72 | --- | --- | --- | **91.7%** | 1.615 | 0.5479 | 93.1% | 1.345 | 100.0% | 2.027 | 8 | * |
| `Touch Multi-edge (valid_w=0 and (valid_h=0 or valid_g=0))` | 36 | --- | --- | --- | **86.1%** | 1.609 | 0.6405 | 69.4% | 1.345 | 97.2% | 2.020 | 6 | * |

### 1.4 Mức độ che khuất (Occlusion Levels)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0 (Fully visible)` | 1,609 | 80 | 1,689 | 95.3% | **97.9%** | 1.340 | 0.3015 | 98.0% | 1.345 | 98.1% | 1.353 | 10 |  |
| `1 (Partly occluded)` | 776 | 245 | 1,021 | 76.0% | **96.3%** | 1.367 | 0.3283 | 94.8% | 1.345 | 95.1% | 1.361 | 8 |  |
| `2 (Largely occluded)` | 275 | 227 | 502 | 54.8% | **95.3%** | 1.383 | 0.3550 | 96.4% | 1.345 | 95.6% | 1.332 | 8 |  |

### 1.5 Góc hướng quan sát θ (Viewing Angle Bins, D19)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Side (<30°)` | 347 | 114 | 461 | 75.3% | **93.7%** | 1.359 | 0.3472 | 94.8% | 1.345 | 93.1% | 1.302 | 5 |  |
| `Diagonal (30–60°)` | 243 | 87 | 330 | 73.6% | **96.7%** | 1.380 | 0.3375 | 91.4% | 1.345 | 95.5% | 1.418 | 8 |  |
| `Front/Rear (>60°)` | 2,070 | 351 | 2,421 | 85.5% | **97.8%** | 1.348 | 0.3068 | 97.9% | 1.345 | 97.8% | 1.354 | 9 |  |

### 1.6 Mức độ khó KITTI (Nested & Disjoint Difficulty)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Easy (nested)` | 1,010 | 14 | 1,024 | 98.6% | **98.5%** | 1.328 | 0.2912 | 99.1% | 1.345 | 98.4% | 1.311 | 9 |  |
| `Moderate (nested)` | 2,311 | 318 | 2,629 | 87.9% | **97.6%** | 1.343 | 0.3051 | 97.7% | 1.345 | 97.3% | 1.341 | 10 |  |
| `Hard (nested)` | 2,660 | 552 | 3,212 | 82.8% | **97.1%** | 1.352 | 0.3148 | 96.9% | 1.345 | 97.0% | 1.353 | 10 |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | :---: |
| `Easy (disjoint)` | 1,010 | 14 | 1,024 | 98.6% | **98.5%** | 1.328 | 0.2912 | 99.1% | 1.345 | 98.4% | 1.311 | 9 |  |
| `Moderate (disjoint)` | 1,301 | 304 | 1,605 | 81.1% | **96.9%** | 1.354 | 0.3159 | 96.5% | 1.345 | 96.4% | 1.364 | 10 |  |
| `Hard (disjoint)` | 349 | 234 | 583 | 59.9% | **94.3%** | 1.418 | 0.3794 | 92.0% | 1.345 | 95.1% | 1.436 | 9 |  |

### 1.7 Nhóm Fallback Pattern 000 (Decision D74)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Fallback (Pattern 000)` | 36 | --- | --- | --- | **86.1%** | 1.609 | 0.6405 | 69.4% | 1.345 | 97.2% | 2.020 | 6 | * |
| `Fused Geometric (>=1 cue)` | 2,624 | --- | --- | --- | **97.3%** | 1.349 | 0.3104 | 97.3% | 1.345 | 97.0% | 1.344 | 10 |  |

## 1. Detector `yolov5su_640`

> - **Bootstrap 95% CI Pooled Coverage (10 cụm):**
>   * Standard CQR: [95.15%, 98.27%] (Mean Width: [1.310, 1.355]) — *CI thô (10 cụm)*
>   * Split Conformal: [95.75%, 98.35%] (Mean Width: [1.351, 1.351]) — *CI thô (10 cụm)*
>   * Mondrian CQR: [94.37%, 98.14%] (Mean Width: [1.302, 1.335]) — *CI thô (10 cụm)*

### 1.1 Dải cự ly theo Ẑ dự đoán (Prospective Distance Bins)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0-10` | 269 | --- | --- | --- | **89.6%** | 1.418 | 0.4327 | 91.1% | 1.351 | 91.8% | 1.464 | 8 |  |
| `10-20` | 837 | --- | --- | --- | **97.6%** | 1.328 | 0.2941 | 97.9% | 1.351 | 96.3% | 1.265 | 9 |  |
| `20-30` | 776 | --- | --- | --- | **96.5%** | 1.314 | 0.2964 | 97.4% | 1.351 | 94.8% | 1.269 | 8 |  |
| `30-50` | 789 | --- | --- | --- | **98.0%** | 1.323 | 0.2877 | 97.9% | 1.351 | 98.6% | 1.366 | 8 |  |
| `>50` | 3 | --- | --- | --- | **100.0%** | 1.483 | 0.3923 | 100.0% | 1.351 | 100.0% | 1.532 | 2 | * |
| `>=30 (grouped)` | 792 | --- | --- | --- | **98.0%** | 1.323 | 0.2881 | 97.9% | 1.351 | 98.6% | 1.367 | 8 |  |

### 1.2 Dải cự ly theo Z thật (Retrospective Distance Bins — Chẩn đoán, v4 §6)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0-10` | 264 | 11 | 275 | 96.0% | **89.4%** | 1.423 | 0.4290 | 89.4% | 1.351 | 90.9% | 1.462 | 8 |  |
| `10-20` | 835 | 61 | 896 | 93.2% | **97.1%** | 1.328 | 0.3034 | 98.2% | 1.351 | 95.9% | 1.268 | 9 |  |
| `20-30` | 781 | 130 | 911 | 85.7% | **98.0%** | 1.317 | 0.2831 | 98.1% | 1.351 | 96.8% | 1.276 | 8 |  |
| `30-50` | 785 | 312 | 1,097 | 71.6% | **97.5%** | 1.318 | 0.2905 | 98.0% | 1.351 | 97.7% | 1.356 | 8 |  |
| `>50` | 9 | 24 | 33 | 27.3% | **66.7%** | 1.425 | 0.5522 | 44.4% | 1.351 | 66.7% | 1.472 | 3 | * |
| `>=30 (grouped)` | 794 | 336 | 1,130 | 70.3% | **97.1%** | 1.319 | 0.2935 | 97.4% | 1.351 | 97.4% | 1.357 | 8 |  |

### 1.3 Mức độ cắt biên (Truncation) & Chạm viền ảnh (Touch Edges, D84)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `No Truncation (0.0)` | 2,478 | 521 | 2,999 | 82.6% | **97.3%** | 1.317 | 0.2901 | 98.1% | 1.351 | 96.8% | 1.299 | 10 |  |
| `Mild (0.0 < t <= 0.15)` | 72 | 4 | 76 | 94.7% | **87.5%** | 1.470 | 0.5064 | 90.3% | 1.351 | 90.3% | 1.490 | 7 | * |
| `Moderate/Severe (0.15 < t <= 0.50)` | 124 | 13 | 137 | 90.5% | **87.1%** | 1.538 | 0.5276 | 80.7% | 1.351 | 86.3% | 1.556 | 8 |  |
| `No Edge Touch (Pattern 111)` | 2,490 | --- | --- | --- | **97.4%** | 1.317 | 0.2904 | 98.1% | 1.351 | 96.8% | 1.299 | 10 |  |
| `Touch Horizontal (valid_w=0)` | 78 | --- | --- | --- | **87.2%** | 1.442 | 0.4084 | 83.3% | 1.351 | 87.2% | 1.421 | 7 | * |
| `Touch Vertical (valid_h=0 or valid_g=0)` | 70 | --- | --- | --- | **84.3%** | 1.591 | 0.6147 | 90.0% | 1.351 | 87.1% | 1.643 | 8 | * |
| `Touch Multi-edge (valid_w=0 and (valid_h=0 or valid_g=0))` | 36 | --- | --- | --- | **86.1%** | 1.581 | 0.6291 | 69.4% | 1.351 | 86.1% | 1.632 | 5 | * |

### 1.4 Mức độ che khuất (Occlusion Levels)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `0 (Fully visible)` | 1,615 | 74 | 1,689 | 95.6% | **97.5%** | 1.316 | 0.2875 | 98.2% | 1.351 | 97.0% | 1.307 | 10 |  |
| `1 (Partly occluded)` | 781 | 240 | 1,021 | 76.5% | **95.3%** | 1.346 | 0.3290 | 95.3% | 1.351 | 94.8% | 1.328 | 9 |  |
| `2 (Largely occluded)` | 278 | 224 | 502 | 55.4% | **95.3%** | 1.379 | 0.3579 | 95.3% | 1.351 | 94.6% | 1.337 | 8 |  |

### 1.5 Góc hướng quan sát θ (Viewing Angle Bins, D19)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Side (<30°)` | 350 | 111 | 461 | 75.9% | **93.7%** | 1.364 | 0.3546 | 94.9% | 1.351 | 92.9% | 1.317 | 5 |  |
| `Diagonal (30–60°)` | 243 | 87 | 330 | 73.6% | **93.8%** | 1.397 | 0.4011 | 91.4% | 1.351 | 93.0% | 1.390 | 8 |  |
| `Front/Rear (>60°)` | 2,081 | 340 | 2,421 | 86.0% | **97.4%** | 1.319 | 0.2879 | 98.1% | 1.351 | 97.0% | 1.308 | 9 |  |

### 1.6 Mức độ khó KITTI (Nested & Disjoint Difficulty)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Easy (nested)` | 1,009 | 15 | 1,024 | 98.5% | **98.0%** | 1.300 | 0.2702 | 99.5% | 1.351 | 97.3% | 1.263 | 9 |  |
| `Moderate (nested)` | 2,322 | 307 | 2,629 | 88.3% | **97.2%** | 1.320 | 0.2934 | 98.0% | 1.351 | 96.7% | 1.306 | 10 |  |
| `Hard (nested)` | 2,674 | 538 | 3,212 | 83.2% | **96.6%** | 1.332 | 0.3069 | 97.0% | 1.351 | 96.1% | 1.316 | 10 |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | :---: |
| `Easy (disjoint)` | 1,009 | 15 | 1,024 | 98.5% | **98.0%** | 1.300 | 0.2702 | 99.5% | 1.351 | 97.3% | 1.263 | 9 |  |
| `Moderate (disjoint)` | 1,313 | 292 | 1,605 | 81.8% | **96.5%** | 1.335 | 0.3113 | 96.9% | 1.351 | 96.3% | 1.340 | 10 |  |
| `Hard (disjoint)` | 352 | 231 | 583 | 60.4% | **92.9%** | 1.410 | 0.3960 | 90.6% | 1.351 | 92.0% | 1.382 | 9 |  |

### 1.7 Nhóm Fallback Pattern 000 (Decision D74)

> - (*) Cờ cảnh báo cỡ mẫu phân nhóm: $n < 100$ theo Decision D54.
> - (**) Recall chỉ áp dụng cho các phân nhóm xác định bằng thuộc tính Ground Truth (Z_gt, Truncation, Occlusion, θ, Difficulty). Các phân nhóm theo thuộc tính suy luận test-time (Ẑ, cờ hợp lệ valid_*, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán tương ứng (Decision D91).

| Phân nhóm | $n_{\text{TP}}$ | $n_{\text{FN}}$ | $n_{\text{GT}}$ | Recall | CQR Cov | CQR Width | CQR Winkler | SC Cov | SC Width | Mondrian Cov | Mondrian Width | $k$ | Cờ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|:---:|
| `Fallback (Pattern 000)` | 36 | --- | --- | --- | **86.1%** | 1.581 | 0.6291 | 69.4% | 1.351 | 86.1% | 1.632 | 5 | * |
| `Fused Geometric (>=1 cue)` | 2,638 | --- | --- | --- | **96.7%** | 1.328 | 0.3025 | 97.4% | 1.351 | 96.2% | 1.312 | 10 |  |
