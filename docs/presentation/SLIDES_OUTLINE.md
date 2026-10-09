# Khung Bài Thuyết Trình Bảo Vệ Đồ Án (Presentation Slides Outline)
## Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors
**Môn học:** DSR301m — Fall 2026 | **Hội đồng:** Khoa Trí tuệ Nhân tạo / Khoa học Dữ liệu  
**Tác giả:** Nhóm Nghiên cứu Đề tài | **Thời lượng dự kiến:** 15–20 phút (16 Slides)

---

## Slide 1: Giới Thiệu Đề Tài (Title Slide)
- **Tiêu đề lớn:** Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors
- **Tiêu đề phụ:** Ước Lượng Khoảng Cách Xe Đơn Kính Kết Hợp Hình Học Phối Cảnh, Học Máy Dư Sai & Hiệu Chuẩn Bất Định Conformal
- **Thông tin đề tài:**
  * Khóa học: DSR301m — Data Science Research Project (FALL 2026)
  * Bộ dữ liệu chuẩn: KITTI Vision Benchmark (Object Detection 3D)
  * Trọng tâm ứng dụng: Hệ thống Hỗ trợ Lái xe Nâng cao (ADAS: FCW, AEB, ACC)
- **Hình ảnh minh họa:** Banner kiến trúc hệ thống (`results/figures/final/fig_01_hybrid_architecture.png`).
- 🗣️ **Speaker Notes:**
  > "Kính thưa quý thầy cô trong Hội đồng, hôm nay nhóm xin phép bảo vệ đề tài: Ước lượng khoảng cách xe phía trước bằng một camera đơn sắc kết hợp giữa mô hình hình học pinhole, mô hình học máy residual và khung lý thuyết định lượng độ bất định Conformal Prediction."

---

## Slide 2: Đặt Vấn Đề & Động Lực Nghiên Cứu (Motivation)
- **Bối cảnh thực tế:**
  * Cảm biến chủ động (LiDAR, Radar) đo độ sâu chính xác nhưng chi phí cao, cồng kềnh, tiêu thụ điện lớn.
  * Camera đơn (Monocular Camera) có chi phí cực thấp, phổ biến trên 100% phương tiện thông minh, cung cấp ngữ nghĩa thị giác phong phú nhưng **mất thông tin chiều sâu 3D**.
- **Thách thức cốt lõi:**
  1. *Hình học quang học thuần túy (Pinhole):* Dễ giải thích nhưng chịu sai số hệ thống lớn do góc xoay xe, đường mấp mô và cắt viền ảnh.
  2. *Học sâu thuần túy (End-to-end / Direct Regression):* Độ chính xác số học cao nhưng là "hộp đen", thiếu căn cứ vật lý, suy biến khó lường khi gặp ca biên.
  3. *Quyết định an toàn tự hành:* Cần không chỉ dự đoán điểm ($\hat{Z}$ mét) mà bắt buộc phải có **khoảng tin cậy an toàn** $[Z_{\text{lo}}, Z_{\text{hi}}]$ đạt độ phủ danh nghĩa được bảo đảm.
- 🗣️ **Speaker Notes:**
  > "Camera đơn sắc rất rẻ nhưng bài toán phục hồi chiều sâu từ ảnh 2D là bài toán ill-posed. Các giải pháp học sâu trực tiếp cho kết quả điểm số đẹp nhưng thiếu kiểm chứng vật lý, trong khi các hệ thống an toàn ADAS đòi hỏi mô hình phải có khả năng giải thích và phải biết rõ mức độ tin cậy của khoảng dự báo."

---

## Slide 3: Ba Khoảng Trống Nghiên Cứu (Research Gaps G1–G3)
- **Gap 1 (Thiếu phân rã sai số hình học vs detector):**
  * Hầu hết các nghiên cứu trước đây đánh giá gộp, không tách biệt được sai số nào đến từ giả định hình học (kích thước xe, mặt đường) và sai số nào do rung lắc (jitter) bounding box của detector.
- **Gap 2 (Thiếu so sánh công bằng đa thế hệ detector):**
  * Các nghiên cứu trước đây hiếm khi so sánh có kiểm soát giữa YOLOv5, YOLOv8 và YOLO11 trên cùng một công thức huấn luyện và đánh giá trên **tập khớp chung (Common Support)**.
- **Gap 3 (Thiếu định lượng bất định có bảo đảm - Calibrated UQ):**
  * Các khoảng tin cậy của mạng nơ-ron thường bị quá tự tin (overconfident); thiếu cơ chế conformal prediction phân tích hành vi độ phủ theo điều kiện trên dữ liệu lái xe tự nhiên.
- 🗣️ **Speaker Notes:**
  > "Nhóm xác định rõ 3 khoảng trống lớn: G1 là sự mập mờ giữa lỗi hình học và lỗi detector; G2 là thiếu đối sánh công bằng cùng giao thức qua 3 thế hệ YOLO; và G3 là sự thiếu vắng của một khung định lượng bất định có bảo chứng toán học."

---

## Slide 4: Kiến Trúc Đề Xuất Tổng Thể (Modular Architecture)
- **Sơ đồ 4 tầng tích hợp (Fig. 1):**
  ```
  Ảnh KITTI ──> [Tầng 1: YOLO Detector] ──> Bounding box (x1, y1, x2, y2)
                      │
                      ▼
               [Tầng 2: Multi-Cue Geometry] ──> Z_w, Z_h, Z_g
                      │ (Covariance Shrinkage Fusion)
                      ▼
               Baseline Vật lý Z_base (Z_d hoặc Z_e fallback)
                      │
                      ▼
               [Tầng 3: Residual XGBoost] ──> Dự đoán r_hat ──> Z_hat = Z_base * exp(r_hat)
                      │
                      ▼
               [Tầng 4: Conformal UQ (CQR)] ──> Khoảng tin cậy [Z_lo, Z_hi]
  ```
- **Điểm sáng thiết kế:** Tách rời độc lập từng khâu, có cơ chế Fallback Pattern 000 an toàn khi mất sạch cue hình học.
- 🗣️ **Speaker Notes:**
  > "Đây là kiến trúc 4 tầng đề xuất. Detector phát hiện xe; Tầng hình học tính 3 cue độc lập và hợp nhất trong log-space; Tầng 3 dùng XGBoost bù trừ sai số quang học; và Tầng 4 xây dựng khoảng tin cậy CQR."

---

## Slide 5: Thiết Kế Dữ Liệu Tách Biệt Chống Rò Rỉ (Splits-v2)
- **Nguyên tắc chia theo chuỗi hành trình (Drive-based Partitioning):**
  * Tuyệt đối không chia ngẫu nhiên theo ảnh để tránh rò rỉ bối cảnh cảnh quay.
  * 7.481 ảnh KITTI chia thành 5 tập độc lập qua thuật toán Simulated Annealing (Seed 85, $n_{\text{eff}} \approx 5.28$):
    - **Split A (50% - 3.740 frames, 35 drives):** Huấn luyện 3 detector YOLO; ước lượng tham số camera hiệu dụng.
    - **Split V (5% - 374 frames, 25 drives):** Chọn checkpoint `last.pt` và ngưỡng conf theo F1 làm trơn.
    - **Split B (20% - 1.499 frames, 34 drives):** Fit trọng số hợp nhất, huấn luyện mạng residual (LODO 12 folds).
    - **Split C (10% - 766 frames, 18 drives):** Hiệu chuẩn phân vị Conformalize CQR.
    - **Split T (15% - 1.102 frames, 29 drives, 3.212 Car Hard):** Nghiệm thu độc lập, **chạy đúng 1 lần duy nhất** có lock bảo vệ.
- **Hình ảnh minh họa:** `results/figures/final/fig_02_splits_spatial_distribution.png`.
- 🗣️ **Speaker Notes:**
  > "Nhóm thiết kế quy trình 5 split cực kỳ nghiêm ngặt. Split B, C, T là các tập xe mà detector chưa từng nhìn thấy trong lúc train, đảm bảo phân bố lỗi bounding box lúc triển khai là hoàn toàn thực tế và đồng nhất."

---

## Slide 6: Mô Hình Hình Học Đa Cue & Hợp Nhất Co Thắt Covariance (RQ1)
- **Ba Cue Hình Học Pinhole:**
  1. *Chiều rộng ($Z_w = f_x \cdot W_{\text{eff}} / w$):* Nhạy với góc quay xe $\theta$; nhìn ngang xe phình to thành chiều dài $\implies$ Pooled AbsRel lên tới **0.2462** (cue yếu nhất, D19).
  2. *Cạnh đáy ($Z_g = f_y \cdot H_{\text{cam}} / (y_2 - (c_y + \delta))$):* Phụ thuộc mặt phẳng đường $\implies$ Pooled AbsRel **0.0931**.
  3. *Chiều cao ($Z_h = f_y \cdot H_{\text{obj}} / h$):* Ổn định nhất qua mọi góc nhìn $\implies$ Pooled AbsRel **0.0650**.
- **Hiện tượng kẹp trọng số $w_w \to 0$ (D31):**
  * Trên bbox detector, phương sai $\Sigma_{ww}$ gấp 10 lần $\Sigma_{hh}$. Thuật toán NNLS tự động kẹp $w_w \to 0.000$, dồn trọng số sang $[w_h, w_g] \approx [0.70, 0.30]$.
- **Hợp nhất log-space ($Z_d$):** Giảm AbsRel xuống **0.0607** trên B OOF (MAE 1.34 m).
- 🗣️ **Speaker Notes:**
  > "Trên dữ liệu thực nghiệm, cue chiều cao là trụ cột vững chắc nhất. Khi chuyển từ nhãn GT sang bounding box của YOLO, thuật toán NNLS tự động triệt tiêu trọng số chiều rộng về 0 để loại bỏ nhiễu góc nhìn xe."

---

## Slide 7: Hiệu Chuẩn Dư Sai (Residual) & Cơ Chế Fallback
- **Mục tiêu học:** $r = \ln Z_{\text{gt}} - \ln Z_{\text{base}}$ trong không gian log (sai số tương đối đồng đều theo khoảng cách).
- **Bộ đặc trưng quan sát được (17 features):** Tọa độ bbox, tỷ lệ khung hình $w/h$, độ lệch tâm camera, confidence detector, cờ chạm biên ảnh (touch flags), và các cue $\ln Z_k$ hợp lệ. **Zero GT leakage (D11).**
- **Cơ chế Fallback Pattern 000 (D13, D74):**
  * Khi xe bị cắt góc viền nặng làm mất toàn bộ 3 cue hình học, pipeline tự động chuyển sang mô hình hồi quy trực tiếp $Z_e$.
  * Phục hồi độ phủ từ 0% lên **77.8% – 86.1%** trên Split T ($n=36$ xe).
- 🗣️ **Speaker Notes:**
  > "Mô hình residual XGBoost sử dụng 17 đặc trưng có sẵn lúc suy luận để bù trừ sai số phối cảnh. Đặc biệt, cơ chế Fallback Pattern 000 bảo vệ an toàn cho hệ thống khi xe chạm mép ảnh bị mất sạch cue hình học."

---

## Slide 8: Định Lượng Bất Định Hiệu Chuẩn CQR (RQ3)
- **Nguyên lý Conformalized Quantile Regression:**
  * Huấn luyện 2 mô hình XGBoost phân vị $\hat{q}_{0.05}$ và $\hat{q}_{0.95}$ trên Split B.
  * Hiệu chuẩn trên Split C ($n=1.489$ TP) tại thống kê thứ tự $k = \lceil (n+1)(1-\alpha) \rceil$ để tìm ngưỡng không tuân thủ $\hat{Q}$.
  * Khoảng tin cậy: $[Z_{\text{lo}}, Z_{\text{hi}}] = [Z_{\text{base}} \exp(\hat{q}_{0.05} - \hat{Q}), \; Z_{\text{base}} \exp(\hat{q}_{0.95} + \hat{Q})]$.
- **Đặc tính kỹ thuật:** Xây dựng trong log-space nên biến đổi exp đơn điệu bảo toàn độ phủ 100% và **không bao giờ bị crossing** ($Z_{\text{lo}} \le Z_{\text{hi}}$).
- 🗣️ **Speaker Notes:**
  > "Khác với các phương pháp heuristic, CQR cung cấp bảo đảm độ phủ mẫu hữu hạn dưới giả định exchangeability. Khoảng tin cậy được tính toán co giãn tự nhiên theo cự ly và độ che khuất."

---

## Slide 9: Kết Quả Thực Nghiệm Chính Trên Split T (RQ1)
- **Bảng số liệu tổng hợp (Trích xuất từ Bảng 3 `tab_03_main_benchmark_split_t`):**

| Phương pháp | YOLO11s AbsRel | YOLOv8s AbsRel | YOLOv5su AbsRel | $\delta_1$ Accuracy | MAE (m) |
|---|:---:|:---:|:---:|:---:|:---:|
| (a) Pinhole Width $Z_w$ | 0.2458 | 0.2451 | 0.2461 | 51.0% | 6.42 |
| (b) Pinhole Height $Z_h$ | 0.0681 | 0.0681 | 0.0694 | 97.4% | 1.55 |
| (c) Ground Contact $Z_g$ | 0.0965 | 0.0969 | 0.0984 | 93.6% | 2.52 |
| (d) Fused Geometry $Z_d$ | 0.0640 | 0.0643 | 0.0655 | 98.0% | 1.41 |
| (f) **Hybrid Residual $\hat{Z}_f$** | **0.0463** | **0.0461** | **0.0474** | **99.7%** | **1.10** |

- **Phân tích RQ1:** Mạng Residual $\hat{Z}_f$ cải thiện rõ rệt so với hình học thuần $Z_d$ ($\Delta = -0.0188$, 95% Cluster Bootstrap CI $[-0.0264, -0.0107]$, loại trừ 0).
- **Hình ảnh minh họa:** `results/figures/final/fig_03_ranging_error_by_distance.png` (sai số AbsRel giảm mạnh ở mọi dải khoảng cách).
- 🗣️ **Speaker Notes:**
  > "Trên tập kiểm định Split T bị khóa, mô hình lai Hybrid (f) giảm sai số tương đối AbsRel từ 6.4% của hình học xuống 4.63%, độ chính xác delta1 đạt 99.7%, kiểm định Bootstrap 10 cụm cho thấy khoảng tin cậy hiệu số loại trừ 0 so với hình học thuần."

---

## Slide 10: Phát Hiện Cốt Lõi: Hybrid (f) vs Direct Regression (e)
- **Hiện tượng thực nghiệm (D78 & Bảng 3):**
  * Mô hình Hồi quy trực tiếp từ bounding box (e) đạt AbsRel **0.0465** trên YOLO11s.
  * Hiệu số $\Delta (f - e) = \mathbf{-0.0002}$ với Paired Bootstrap 95% CI là **$[-0.0012, +0.0023]$ (chứa số 0)**.
  * $\implies$ **Không có bằng chứng thực nghiệm về sự vượt trội số học giữa Residual (f) và Direct Regression (e) trên 10 cụm drive của Split T.**
- **Định vị lại Giá trị Khoa học của Mô hình Lai (Hybrid Value):**
  1. *Khả năng giải thích vật lý (Explainability):* Cung cấp điểm neo vật lý $Z_d$ minh bạch cho hệ thống an toàn.
  2. *Phân rã sai số:* Tách bạch lỗi phối cảnh quang học và lỗi bounding box detector.
  3. *An toàn hệ thống:* Cơ chế suy thoái êm dịu (graceful fallback) khi mất cue.
- 🗣️ **Speaker Notes:**
  > "Đây là phát hiện trung thực quan trọng nhất của đề tài: Mô hình lai không vượt trội số học so với hồi quy trực tiếp. Tuy nhiên, giá trị của nó không nằm ở vài phần vạn sai số, mà nằm ở tính khả giải thích vật lý và khả năng phòng thủ an toàn khi suy thoái."

---

## Slide 11: So Sánh Đa Thế Hệ YOLO Trên Tập Khớp Chung (RQ2)
- **Tập Khớp Chung (Common Support Set):** Đánh giá trên đúng **2.528 xe** được cả 3 detector phát hiện đồng thời để triệt tiêu thiên lệch Recall.
  * YOLO11s: AbsRel = **0.0446**
  * YOLOv8s: AbsRel = **0.0449**
  * YOLOv5su: AbsRel = **0.0457**
  * Khoảng tin cậy Paired Bootstrap CI giữa các cặp detector đều chứa 0 $\implies$ Ba detector có độ chính xác khoảng cách tương đương nhau.
- **Phân tích Tương quan Bounding Box (RQ2):**
  * Hệ số tương quan Rank Spearman giữa IoU / độ lệch cạnh đáy với sai số khoảng cách gần như bằng 0 ($\rho \in [-0.087, +0.104]$ với $|\rho| \le 0.11$, 23/24 cấu hình chứa 0).
  * $\implies$ Sai số ước lượng khoảng cách chủ yếu do quy mô tỉ lệ chiều sâu phối cảnh, không bị chi phối bởi rung lắc pixel 2D cục bộ.
- 🗣️ **Speaker Notes:**
  > "Trên 2.528 xe chung, cả 3 thế hệ YOLO đều cho độ chính xác tương đồng. Phân tích tương quan chỉ ra rằng rung lắc cạnh đáy 2D không phải nguyên nhân chính gây sai số khoảng cách mà bản chất nằm ở bài toán tỉ lệ phối cảnh chiều sâu."

---

## Slide 12: Phân Tích Độ Phủ & Bất Định Conformal (RQ3)
- **Kết quả Độ Phủ Thực Nghiệm (Bảng 5 & Fig. 4):**
  * Mức danh nghĩa: $90.0\%$
  * Thực tế trên Split T: YOLO11s đạt **$96.39\%$**, YOLOv8s đạt **$97.14\%$**, YOLOv5su đạt **$96.60\%$**.
  * Tỷ lệ độ rộng khoảng trung bình: $Z_{\text{hi}} / Z_{\text{lo}} \approx 1.32\times\text{--}1.35\times$ (toàn Split T).
  * Vi phạm cắt chéo (Crossing): **0 trường hợp**.
- **Giải mã hiện tượng Over-coverage (D79, D87):**
  * Phát hiện post-hoc: Split C tập trung 2 drive khó chiếm 39.3% mẫu ($D_{\text{KS}} = 0.15$), làm ngưỡng $\hat{Q}$ bị nở rộng $\implies$ Tạo tính bảo thủ an toàn khi kiểm định trên Split T.
- **Đánh đổi của Mondrian CQR:** Khôi phục độ phủ cự ly gần 0–10m (từ ~91% lên ~97%) nhưng làm phồng độ rộng khoảng tin cậy từ $1.45\times$ lên $1.80\times$ ở dải 0–10m ($1.33\times \to 1.80\times$ toàn tập).
- 🗣️ **Speaker Notes:**
  > "CQR đạt độ phủ thực nghiệm 96.4%, vượt mức danh nghĩa 90%. Đây là phát hiện hậu nghiệm xuất phát từ việc tập hiệu chuẩn Split C có độ khó cao hơn Split T, mang lại độ phủ thực nghiệm bảo thủ trên tập kiểm định Split T."

---

## Slide 13: Đánh Giá Tính Khả Thi Thời Gian Thực Edge Deployment (RQ4)
- **Kết quả Benchmark Độ Trễ Tier 1 In-Memory (Bảng 7, Nhãn `PRELIMINARY-v2`):**

| Khâu xử lý | GPU RTX 5060 (FP16 CUDA) | CPU Multi-core (ONNX FP32) | Tỷ trọng trên GPU |
|---|:---:|:---:|:---:|
| Preprocess in-memory | ~9.6 ms | ~6.1 ms | ~25% |
| YOLO Forward + NMS | ~26.8 ms | ~108.7 ms | ~71% |
| **Geometry Stage** | **0.18 ms** | **0.18 ms** | **0.5%** |
| **Residual XGBoost** | **0.57 ms** | **0.57 ms** | **1.5%** |
| **Conformal CQR** | **0.58 ms** | **0.58 ms** | **1.5%** |
| **Tổng End-to-End** | **37.99 ms (26.3 FPS)** | **116.17 ms (8.6 FPS)** | **100.0%** |

- **Kết luận:** Ba khâu hậu detector chỉ tốn **$\approx 1.35$ ms/ảnh ($< 4.2\%$ GPU)** $\implies$ Chi phí tính toán của phương pháp lai và conformal là không đáng kể so với detector.
- 🗣️ **Speaker Notes:**
  > "Toàn bộ khâu hình học, residual và conformal chỉ mất 1.35 ms trên GPU, chiếm chưa đầy 4.2% thời gian xử lý. Mô hình đạt 26.3 FPS trên GPU RTX 5060 Laptop và 8.6 FPS trên CPU thuần túy."

---

## Slide 14: Nghiên Cứu Điển Hình Định Tính & Phân Tích Lỗi (Fig. 5)
- **Minh họa 4 ca thành công nổi bật:**
  1. *Nhìn ngang (D19, Frame 000006):* $Z_w = 9.8\text{m}$ bị co rút, Residual bù chuẩn lên $\hat{Z}_f = 20.6\text{m}$ vs $Z_{\text{gt}} = 19.7\text{m}$ (AbsRel 4.5%, Cover=True).
  2. *Lệch tâm vật lý dải gần (D21, Frame 000385):* $Z_d$ lệch âm -11.2% do tâm 3D vs mặt cản gần nhất, Residual triệt tiêu độ lệch về $\hat{Z}_f = 7.99\text{m}$ vs $Z_{\text{gt}} = 7.91\text{m}$ (AbsRel 0.96%).
  3. *Cắt mép ảnh đáy (Frame 000152):* CQR nới rộng khoảng $[4.08, 7.07]\text{m}$ chứa an toàn $Z_{\text{gt}} = 6.37\text{m}$.
  4. *Fallback thành công (Frame 000211):* Mất sạch cue, model direct đưa về $\hat{Z}_f = 8.23\text{m}$ vs $Z_{\text{gt}} = 7.91\text{m}$ (Cover=True).
- **Phân tích ca lỗi lớn nhất (Top-1 Outlier, Frame 001414):** Xe bị cắt góc viền chéo quá nặng, mô hình trực tiếp gặp lỗi ngoại suy phối cảnh $\implies$ AbsRel $42.9\%$ (Cover=False).
- 🗣️ **Speaker Notes:**
  > "Các hình ảnh định tính chứng minh cơ chế bù trừ sai số vật lý hoạt động chính xác: bù chiều rộng khi xe quay ngang và triệt tiêu độ lệch tâm xe 1m ở cự ly gần."

---

## Slide 15: Liêm Chính Khoa Học & 14 Hạn Chế Cốt Lõi (Limitations)
- **Nhóm nghiên cứu công khai toàn diện 14 hạn chế phương pháp luận:**
  1. *Survivorship Bias:* Kết quả đánh giá trên True Positives; ở cự ly xa $30\text{--}50\text{m}$, detector bỏ sót $\approx 48\text{--}50\%$ xe.
  2. *Dải xa $>50\text{m}$ thưa thớt:* GT chỉ có 33 xe, TP chỉ có $\le 9$ xe trên Split T.
  3. *Tương quan cụm:* Số cụm drive $k \le 12 < 20$, khoảng Bootstrap CI mang tính chất thô.
  4. *Minh bạch quy trình v1.1:* Sửa lỗi serialization $base\_score$ hậu kiểm post-hoc mà không mở khóa Split T (`final_T.lock` bất biến).
  5. *Không claim "đầu tiên":* Liệt kê đầy đủ tiền lệ trong tài liệu tham khảo (Dist-YOLO, DisNet, DECADE, MonoLoco, CQR, f-Cal).
- 🗣️ **Speaker Notes:**
  > "Chúng tôi cam kết liêm chính học thuật tuyệt đối: Báo cáo trung thực thiên lệch kẻ sống sót ở cự ly xa, sự thưa thớt của xe trên 50m, và minh bạch quy trình khắc phục lỗi kỹ thuật v1.1."

---

## Slide 16: Kết Luận & Định Hướng Phát Triển (Conclusion)
- **Đóng góp chính của đề tài:**
  1. Xây dựng thành công quy trình chia 5 tập `splits-v2` không rò rỉ bối cảnh, bảo đảm tính tái lập 100%.
  2. Khung làm việc lai Hybrid kết hợp pinhole, residual XGBoost và CQR đạt sai số AbsRel **4.63%** và độ phủ an toàn **96.4%**.
  3. Phân rã tách bạch sai số hình học vs detector jitter; chứng minh chi phí tính toán hậu detector chỉ tốn **1.35 ms**.
  4. Báo cáo kiểm toán độc lập `CHECKLIST_AUDIT.md` đạt chuẩn toàn diện (**14/14 PASS**).
- **Hướng phát triển:** Tích hợp bộ lọc Kalman Tracking theo chuỗi thời gian, mở rộng sang xe máy và người đi bộ.
- **Lời cảm ơn:** Nhóm xin chân thành cảm ơn Quý Thầy Cô Hội đồng đã lắng nghe và kính mời Quý Thầy Cô đặt câu hỏi phản biện!
- 🗣️ **Speaker Notes:**
  > "Đề tài đã hoàn thành toàn diện các mục tiêu đề ra với tính tái lập 100%. Nhóm xin trân trọng cảm ơn Hội đồng và sẵn sàng tiếp thu các câu hỏi phản biện."
