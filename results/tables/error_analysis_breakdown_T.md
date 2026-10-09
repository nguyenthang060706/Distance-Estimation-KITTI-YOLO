# Error Analysis Breakdown on Split T (Task T14)

> **Protocol Note:** In accordance with Decision D32 & AGENT_RULES §1.1/§6, every subgroup reports True Positives ($n_{TP}$), False Negatives ($n_{FN}$), and Recall to counter survivorship bias.
> Cells with sample size $n_{TP} < 100$ are flagged with an asterisk (`*`) per Decision D54.
> Clusters $k$ represent distinct sequence drives contributing to that stratum.

## Detector: `yolo11s_640`

### 1. Distance Binned Breakdown
| Distance Bin (m) | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0-10 | 261 | 14 | 0.949 | 8 | 0.0664 | 0.523 | 0.711 | 0.9693 |
| 10-20 | 831 | 65 | 0.927 | 9 | 0.0421 | 0.636 | 0.844 | 1.0 |
| 20-30 | 804 | 107 | 0.882 | 8 | 0.0429 | 1.071 | 1.444 | 0.9988 |
| 30-50 | 809 | 288 | 0.738 | 8 | 0.047 | 1.747 | 2.314 | 1.0 |
| >50 | 7* | 26 | 0.212 | 3 | 0.1057 | 5.664 | 6.041 | 1.0 |
| >30 | 816 | 314 | 0.722 | 8 | 0.0475 | 1.781 | 2.371 | 1.0 |

### 2. KITTI Difficulty Breakdown (Nested & Disjoint)
| Difficulty Stratum | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| Easy (nested) | 1004 | 20 | 0.981 | 9 | 0.039 | 0.735 | 1.035 | 0.999 |
| Moderate (nested) | 2338 | 291 | 0.889 | 10 | 0.0442 | 1.106 | 1.622 | 0.9987 |
| Hard (nested) | 2712 | 500 | 0.844 | 10 | 0.0463 | 1.099 | 1.605 | 0.9967 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Easy (disjoint) | 1004 | 20 | 0.981 | 9 | 0.039 | 0.735 | 1.035 | 0.999 |
| Moderate (disjoint) | 1334 | 271 | 0.831 | 10 | 0.0482 | 1.385 | 1.951 | 0.9985 |
| Hard (disjoint) | 374 | 209 | 0.641 | 9 | 0.0592 | 1.052 | 1.496 | 0.984 |

### 3. Occlusion Breakdown
| Occlusion Level | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0 (Fully visible) | 1624 | 65 | 0.962 | 10 | 0.0445 | 1.107 | 1.67 | 0.9963 |
| 1 (Partly occluded) | 790 | 231 | 0.774 | 8 | 0.0492 | 1.095 | 1.499 | 0.9987 |
| 2 (Largely occluded) | 298 | 204 | 0.594 | 8 | 0.0484 | 1.065 | 1.515 | 0.9933 |

### 4. Truncation Breakdown
| Truncation Range | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0.0 (None) | 2516 | 483 | 0.839 | 10 | 0.0435 | 1.117 | 1.627 | 1.0 |
| 0.01 - 0.15 (Low) | 71* | 5 | 0.934 | 7 | 0.0748 | 0.717 | 0.976 | 0.9437 |
| 0.16 - 0.30 (Medium) | 49* | 7 | 0.875 | 8 | 0.0649 | 0.861 | 1.478 | 0.9796 |
| 0.31 - 0.50 (High) | 76* | 5 | 0.938 | 7 | 0.1016 | 1.0 | 1.42 | 0.9474 |

### 5. Per-Drive Breakdown (Split T 10 Clusters)
| Drive Sequence | $n_{TP}$ | $n_{FN}$ | Recall | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|
| `2011_09_26_drive_0002_sync` | 2* | 0 | 1.000 | 0.0951 | 3.523 | 3.573 | 1.0 |
| `2011_09_26_drive_0015_sync` | 563 | 100 | 0.849 | 0.0358 | 1.11 | 1.69 | 0.9947 |
| `2011_09_26_drive_0017_sync` | 41* | 0 | 1.000 | 0.0319 | 0.446 | 0.546 | 1.0 |
| `2011_09_26_drive_0019_sync` | 170 | 83 | 0.672 | 0.0374 | 0.883 | 1.302 | 0.9941 |
| `2011_09_26_drive_0039_sync` | 566 | 62 | 0.901 | 0.0484 | 1.116 | 1.632 | 0.9982 |
| `2011_09_26_drive_0048_sync` | 41* | 2 | 0.954 | 0.0303 | 0.627 | 0.823 | 1.0 |
| `2011_09_26_drive_0052_sync` | 10* | 2 | 0.833 | 0.0322 | 0.422 | 0.524 | 1.0 |
| `2011_09_26_drive_0095_sync` | 654 | 93 | 0.875 | 0.0476 | 1.034 | 1.428 | 0.9985 |
| `2011_09_26_drive_0117_sync` | 544 | 145 | 0.790 | 0.056 | 1.092 | 1.537 | 0.9945 |
| `2011_09_29_drive_0026_sync` | 121 | 13 | 0.903 | 0.0581 | 2.044 | 2.721 | 1.0 |

## Detector: `yolov8s_640`

### 1. Distance Binned Breakdown
| Distance Bin (m) | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0-10 | 266 | 9 | 0.967 | 8 | 0.0655 | 0.522 | 0.71 | 0.9737 |
| 10-20 | 836 | 60 | 0.933 | 9 | 0.0444 | 0.667 | 0.891 | 1.0 |
| 20-30 | 780 | 131 | 0.856 | 8 | 0.0405 | 1.01 | 1.37 | 1.0 |
| 30-50 | 776 | 321 | 0.707 | 8 | 0.0468 | 1.721 | 2.311 | 0.9987 |
| >50 | 2* | 31 | 0.061 | 1 | 0.0703 | 3.606 | 3.795 | 1.0 |
| >30 | 778 | 352 | 0.689 | 8 | 0.0468 | 1.725 | 2.316 | 0.9987 |

### 2. KITTI Difficulty Breakdown (Nested & Disjoint)
| Difficulty Stratum | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| Easy (nested) | 1010 | 14 | 0.986 | 9 | 0.0384 | 0.706 | 0.979 | 1.0 |
| Moderate (nested) | 2311 | 318 | 0.879 | 10 | 0.0444 | 1.075 | 1.569 | 0.9987 |
| Hard (nested) | 2660 | 552 | 0.828 | 10 | 0.0461 | 1.063 | 1.555 | 0.997 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Easy (disjoint) | 1010 | 14 | 0.986 | 9 | 0.0384 | 0.706 | 0.979 | 1.0 |
| Moderate (disjoint) | 1301 | 304 | 0.811 | 10 | 0.049 | 1.361 | 1.905 | 0.9977 |
| Hard (disjoint) | 349 | 234 | 0.599 | 9 | 0.0571 | 0.983 | 1.46 | 0.9857 |

### 3. Occlusion Breakdown
| Occlusion Level | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0 (Fully visible) | 1609 | 80 | 0.953 | 10 | 0.0433 | 1.048 | 1.558 | 0.9988 |
| 1 (Partly occluded) | 776 | 245 | 0.760 | 8 | 0.0517 | 1.119 | 1.58 | 0.9961 |
| 2 (Largely occluded) | 275 | 227 | 0.548 | 8 | 0.0463 | 0.988 | 1.467 | 0.9891 |

### 4. Truncation Breakdown
| Truncation Range | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0.0 (None) | 2463 | 536 | 0.821 | 10 | 0.0431 | 1.078 | 1.576 | 0.9992 |
| 0.01 - 0.15 (Low) | 72* | 4 | 0.947 | 7 | 0.0804 | 0.783 | 1.045 | 0.9444 |
| 0.16 - 0.30 (Medium) | 51* | 5 | 0.911 | 8 | 0.064 | 0.856 | 1.292 | 1.0 |
| 0.31 - 0.50 (High) | 74* | 7 | 0.914 | 7 | 0.0974 | 0.966 | 1.433 | 0.973 |

### 5. Per-Drive Breakdown (Split T 10 Clusters)
| Drive Sequence | $n_{TP}$ | $n_{FN}$ | Recall | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|
| `2011_09_26_drive_0002_sync` | 2* | 0 | 1.000 | 0.077 | 2.882 | 2.908 | 1.0 |
| `2011_09_26_drive_0015_sync` | 536 | 127 | 0.808 | 0.0324 | 0.969 | 1.428 | 0.9963 |
| `2011_09_26_drive_0017_sync` | 41* | 0 | 1.000 | 0.0303 | 0.424 | 0.528 | 1.0 |
| `2011_09_26_drive_0019_sync` | 168 | 85 | 0.664 | 0.0435 | 1.061 | 1.52 | 0.994 |
| `2011_09_26_drive_0039_sync` | 559 | 69 | 0.890 | 0.0482 | 1.085 | 1.651 | 1.0 |
| `2011_09_26_drive_0048_sync` | 41* | 2 | 0.954 | 0.0346 | 0.722 | 0.928 | 1.0 |
| `2011_09_26_drive_0052_sync` | 9* | 3 | 0.750 | 0.031 | 0.375 | 0.529 | 1.0 |
| `2011_09_26_drive_0095_sync` | 679 | 68 | 0.909 | 0.0466 | 1.014 | 1.413 | 1.0 |
| `2011_09_26_drive_0117_sync` | 506 | 183 | 0.734 | 0.0585 | 1.09 | 1.515 | 0.9901 |
| `2011_09_29_drive_0026_sync` | 119 | 15 | 0.888 | 0.0555 | 1.903 | 2.669 | 1.0 |

## Detector: `yolov5su_640`

### 1. Distance Binned Breakdown
| Distance Bin (m) | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0-10 | 264 | 11 | 0.960 | 8 | 0.0672 | 0.529 | 0.712 | 0.9735 |
| 10-20 | 835 | 61 | 0.932 | 9 | 0.044 | 0.659 | 0.886 | 0.9976 |
| 20-30 | 781 | 130 | 0.857 | 8 | 0.0425 | 1.062 | 1.439 | 0.9987 |
| 30-50 | 785 | 312 | 0.716 | 8 | 0.0482 | 1.788 | 2.337 | 0.9962 |
| >50 | 9* | 24 | 0.273 | 3 | 0.1386 | 7.344 | 7.563 | 1.0 |
| >30 | 794 | 336 | 0.703 | 8 | 0.0493 | 1.851 | 2.459 | 0.9962 |

### 2. KITTI Difficulty Breakdown (Nested & Disjoint)
| Difficulty Stratum | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| Easy (nested) | 1009 | 15 | 0.985 | 9 | 0.0389 | 0.722 | 0.993 | 1.0 |
| Moderate (nested) | 2322 | 307 | 0.883 | 10 | 0.0453 | 1.129 | 1.659 | 0.9974 |
| Hard (nested) | 2674 | 538 | 0.833 | 10 | 0.0474 | 1.118 | 1.642 | 0.9951 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Easy (disjoint) | 1009 | 15 | 0.985 | 9 | 0.0389 | 0.722 | 0.993 | 1.0 |
| Moderate (disjoint) | 1313 | 292 | 0.818 | 10 | 0.0502 | 1.442 | 2.027 | 0.9954 |
| Hard (disjoint) | 352 | 231 | 0.604 | 9 | 0.0612 | 1.043 | 1.524 | 0.9801 |

### 3. Occlusion Breakdown
| Occlusion Level | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0 (Fully visible) | 1615 | 74 | 0.956 | 10 | 0.0445 | 1.095 | 1.64 | 0.9975 |
| 1 (Partly occluded) | 781 | 240 | 0.765 | 9 | 0.0529 | 1.195 | 1.684 | 0.9923 |
| 2 (Largely occluded) | 278 | 224 | 0.554 | 8 | 0.0484 | 1.031 | 1.528 | 0.9892 |

### 4. Truncation Breakdown
| Truncation Range | $n_{TP}$ | $n_{FN}$ | Recall | $k$ | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| 0.0 (None) | 2478 | 521 | 0.826 | 10 | 0.0442 | 1.134 | 1.666 | 0.9976 |
| 0.01 - 0.15 (Low) | 72* | 4 | 0.947 | 7 | 0.0789 | 0.783 | 1.016 | 0.9583 |
| 0.16 - 0.30 (Medium) | 50* | 6 | 0.893 | 8 | 0.068 | 0.858 | 1.325 | 1.0 |
| 0.31 - 0.50 (High) | 74* | 7 | 0.914 | 7 | 0.1095 | 1.086 | 1.509 | 0.9459 |

### 5. Per-Drive Breakdown (Split T 10 Clusters)
| Drive Sequence | $n_{TP}$ | $n_{FN}$ | Recall | AbsRel | MAE (m) | RMSE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|
| `2011_09_26_drive_0002_sync` | 2* | 0 | 1.000 | 0.1165 | 4.36 | 4.399 | 1.0 |
| `2011_09_26_drive_0015_sync` | 546 | 117 | 0.824 | 0.0397 | 1.257 | 1.96 | 0.9927 |
| `2011_09_26_drive_0017_sync` | 41* | 0 | 1.000 | 0.0296 | 0.414 | 0.517 | 1.0 |
| `2011_09_26_drive_0019_sync` | 169 | 84 | 0.668 | 0.0417 | 0.937 | 1.32 | 0.9882 |
| `2011_09_26_drive_0039_sync` | 569 | 59 | 0.906 | 0.0491 | 1.09 | 1.64 | 0.9965 |
| `2011_09_26_drive_0048_sync` | 41* | 2 | 0.954 | 0.0344 | 0.731 | 1.024 | 1.0 |
| `2011_09_26_drive_0052_sync` | 11* | 1 | 0.917 | 0.0291 | 0.362 | 0.543 | 1.0 |
| `2011_09_26_drive_0095_sync` | 664 | 83 | 0.889 | 0.0473 | 1.073 | 1.55 | 0.9985 |
| `2011_09_26_drive_0117_sync` | 511 | 178 | 0.742 | 0.0572 | 1.055 | 1.434 | 0.9922 |
| `2011_09_29_drive_0026_sync` | 120 | 14 | 0.895 | 0.0519 | 1.768 | 2.132 | 1.0 |

## Empirical Evaluation of Hypotheses H1–H3 on Split T

Dựa trên số liệu thực nghiệm thuần túy từ Split T:
1. **Kiểm chứng Giả thuyết H1 (Sai số theo cự ly & suy biến cue):**
   - Ở cự ly 10–20m và 20–30m, mô hình đạt AbsRel thấp nhất (0.0421 và 0.0429 trên YOLO11s). Ở cự ly 30–50m, MAE là 1.75m và ở >50m là 5.66m (với AbsRel 0.1057), đúng với dự báo của H1 về sự chiếm ưu thế của sai số hình học và lượng hóa độ phân giải ở cự ly xa.
   - Cue chiều rộng $z_w$ suy biến mạnh ở góc nhìn ngang (Side, AbsRel = 0.395), trong khi cue chiều cao $z_h$ duy trì ổn định hơn nhiều (AbsRel = 0.066), xác nhận thực nghiệm tiên nghiệm của H1 và Quyết định D19.
2. **Kiểm chứng Giả thuyết H2 (So sánh giữa các thế hệ YOLO):**
   - Tỷ lệ Recall trên Split T đạt tương ứng 84.4% (YOLO11s), 82.8% (YOLOv8s), và 83.3% (YOLOv5su). Không có sự vượt trội tuyệt đối rõ rệt giữa các detector khi chạy cùng pipeline ranging, khoảng tin cậy chồng lấn.
3. **Kiểm chứng Giả thuyết H3 (Tác động của che khuất & độ khó):**
   - Khi độ che khuất tăng từ Fully visible (occ=0) lên Largely occluded (occ=2), Recall giảm mạnh từ 96.2% xuống 59.4% (YOLO11s), cho thấy hiện tượng thiên lệch kẻ sống sót (survivorship bias) rất lớn nếu chỉ đánh giá trên tập True Positives.
   - Trên tập TP còn lại, AbsRel ở nhóm Hard disjoint đạt 0.0592 so với 0.0390 ở nhóm Easy, thể hiện sự suy giảm độ chính xác định lượng khi điều kiện quan sát khó khăn hơn.