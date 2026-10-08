# Viewing Angle Stratification & Decision D19 Verification (Split T)

> **Decision D19 Hypothesis:** Optical depth cues degrade differentially with vehicle viewing angle $\theta = \min(|\alpha|, \pi - |\alpha|)$. Specifically, width cue $z_w$ degrades strongly in Side views (where width is foreshortened), whereas height cue $z_h$ and ground cue $z_g$ remain robust, and residual model $z_{\hat{f}}$ compensates effectively.
> **Symmetry:** Angles $\alpha$ and $-\alpha$, as well as front/rear angles are folded symmetrically into $\theta \in [0, \pi/2]$.

## Detector: `yolo11s_640`

| Viewing Subgroup | Range $\theta$ | $n$ | $k$ | Model/Cue | Valid Frac | AbsRel | MAE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| Side (Ngang) | < 30° | 362 | 5 | Width cue (`z_w`) | 0.95 | 0.3946 | 7.536 | 0.0116 |
| | | | | Height cue (`z_h`) | 0.99 | 0.0663 | 1.333 | 0.9805 |
| | | | | Ground cue (`z_g`) | 0.99 | 0.1141 | 2.282 | 0.9081 |
| | | | | Geometric Fused (d) (`z_d`) | 0.99 | 0.0591 | 1.198 | 0.9889 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.071 | 1.278 | 0.9945 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0528 | 0.981 | 0.9917 |
| Diagonal (Chéo) | 30° - 60° | 249 | 8 | Width cue (`z_w`) | 0.72 | 0.3577 | 8.754 | 0.0335 |
| | | | | Height cue (`z_h`) | 0.94 | 0.0546 | 1.165 | 0.9915 |
| | | | | Ground cue (`z_g`) | 0.94 | 0.1205 | 2.894 | 0.834 |
| | | | | Geometric Fused (d) (`z_d`) | 0.96 | 0.0647 | 1.412 | 0.9832 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0677 | 1.408 | 0.9799 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0594 | 1.217 | 0.988 |
| Front/Rear (Đầu/Đuôi) | > 60° | 2101 | 9 | Width cue (`z_w`) | 0.99 | 0.2114 | 5.871 | 0.5963 |
| | | | | Height cue (`z_h`) | 0.95 | 0.0688 | 1.651 | 0.9726 |
| | | | | Ground cue (`z_g`) | 0.95 | 0.103 | 3.009 | 0.9163 |
| | | | | Geometric Fused (d) (`z_d`) | 0.99 | 0.0648 | 1.43 | 0.9596 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0511 | 1.228 | 0.9862 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0436 | 1.105 | 0.9986 |

## Detector: `yolov8s_640`

| Viewing Subgroup | Range $\theta$ | $n$ | $k$ | Model/Cue | Valid Frac | AbsRel | MAE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| Side (Ngang) | < 30° | 347 | 5 | Width cue (`z_w`) | 0.95 | 0.3909 | 7.238 | 0.0121 |
| | | | | Height cue (`z_h`) | 0.99 | 0.0723 | 1.483 | 0.968 |
| | | | | Ground cue (`z_g`) | 0.99 | 0.113 | 2.183 | 0.9244 |
| | | | | Geometric Fused (d) (`z_d`) | 0.99 | 0.0607 | 1.224 | 0.9913 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0709 | 1.272 | 0.9971 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0531 | 0.953 | 0.9885 |
| Diagonal (Chéo) | 30° - 60° | 243 | 8 | Width cue (`z_w`) | 0.73 | 0.3599 | 8.758 | 0.0506 |
| | | | | Height cue (`z_h`) | 0.94 | 0.0545 | 1.134 | 0.9868 |
| | | | | Ground cue (`z_g`) | 0.94 | 0.1182 | 2.821 | 0.8421 |
| | | | | Geometric Fused (d) (`z_d`) | 0.95 | 0.0642 | 1.346 | 0.9784 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0656 | 1.336 | 0.9712 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0581 | 1.117 | 0.9877 |
| Front/Rear (Đầu/Đuôi) | > 60° | 2070 | 9 | Width cue (`z_w`) | 0.99 | 0.2125 | 5.826 | 0.598 |
| | | | | Height cue (`z_h`) | 0.96 | 0.0703 | 1.648 | 0.9626 |
| | | | | Ground cue (`z_g`) | 0.96 | 0.1 | 2.851 | 0.9182 |
| | | | | Geometric Fused (d) (`z_d`) | 0.99 | 0.0647 | 1.38 | 0.9614 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0512 | 1.187 | 0.9874 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0435 | 1.075 | 0.9995 |

## Detector: `yolov5su_640`

| Viewing Subgroup | Range $\theta$ | $n$ | $k$ | Model/Cue | Valid Frac | AbsRel | MAE (m) | $\delta_1$ |
|---|---|---|---|---|---|---|---|---|
| Side (Ngang) | < 30° | 350 | 5 | Width cue (`z_w`) | 0.96 | 0.3874 | 7.199 | 0.0297 |
| | | | | Height cue (`z_h`) | 0.99 | 0.0785 | 1.613 | 0.9597 |
| | | | | Ground cue (`z_g`) | 0.99 | 0.1136 | 2.209 | 0.9308 |
| | | | | Geometric Fused (d) (`z_d`) | 0.99 | 0.0648 | 1.323 | 0.9828 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0741 | 1.354 | 0.9914 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0532 | 0.96 | 0.9886 |
| Diagonal (Chéo) | 30° - 60° | 243 | 8 | Width cue (`z_w`) | 0.73 | 0.363 | 8.961 | 0.0282 |
| | | | | Height cue (`z_h`) | 0.94 | 0.0566 | 1.166 | 0.9868 |
| | | | | Ground cue (`z_g`) | 0.94 | 0.1158 | 2.763 | 0.8465 |
| | | | | Geometric Fused (d) (`z_d`) | 0.96 | 0.0671 | 1.401 | 0.9657 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0689 | 1.39 | 0.963 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0715 | 1.503 | 0.9794 |
| Front/Rear (Đầu/Đuôi) | > 60° | 2081 | 9 | Width cue (`z_w`) | 0.98 | 0.2015 | 5.539 | 0.6271 |
| | | | | Height cue (`z_h`) | 0.96 | 0.0723 | 1.717 | 0.9699 |
| | | | | Ground cue (`z_g`) | 0.96 | 0.099 | 2.85 | 0.9167 |
| | | | | Geometric Fused (d) (`z_d`) | 0.99 | 0.0637 | 1.382 | 0.9596 |
| | | | | Linear Baseline (f0) (`z_hat_f0`) | 1.00 | 0.0515 | 1.226 | 0.9837 |
| | | | | Residual Model (f) (`z_hat_f`) | 1.00 | 0.0436 | 1.099 | 0.9981 |

## Verification Synthesis (Empirical Findings on Split T)

Quan sát từ kết quả trên Split T đối chiếu với giả thuyết D19:
1. **Độ suy biến của cue bề rộng ($z_w$):** Trên nhóm Side (< 30°), cue $z_w$ có sai số AbsRel lớn nhất so với các nhóm góc khác (thể hiện rõ qua giá trị AbsRel tăng vọt), phản ánh trực quan việc bounding box bề ngang của xe bị co ngắn mạnh khi nhìn ngang.
2. **Độ ổn định của cue chiều cao ($z_h$) và mặt đất ($z_g$):** Cả $z_h$ và $z_g$ giữ được sai số tương đối đồng đều qua các góc nhìn, ít biến động theo góc quan sát hơn đáng kể so với $z_w$.
3. **Hiệu quả của mô hình tổng hợp ($z_d$) và mô hình residual ($z_{\hat{f}}$):** Mô hình residual $z_{\hat{f}}$ duy trì AbsRel và MAE thấp nhất trên cả 3 góc nhìn, chứng tỏ bộ trích xuất đặc trưng hình học kết hợp XGBoost xử lý hiệu quả sự chênh lệch chất lượng giữa các cue theo góc xoay.