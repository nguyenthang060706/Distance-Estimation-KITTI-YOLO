# KẾ HOẠCH HOÀN CHỈNH (v4)

## Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors

*Bản v4: rà soát xung đột và tối ưu lại từ v3 — thiết kế dữ liệu, phương pháp, giao thức đánh giá và lộ trình.*

*Soạn ngày 30/09/2026 · ⚠️ = giả định cần xác nhận hoặc kiểm tra trước khi chạy.*

## 0. Những xung đột và điểm chưa tối ưu trong v3, và cách sửa
Bảng sắp theo mức ảnh hưởng đến tính đúng đắn của kết quả (từ trên xuống). Các mục 1–4 là lỗi thiết kế có thể làm kết quả sai hoặc không bảo vệ được; các mục còn lại là mâu thuẫn nội bộ hoặc tối ưu lộ trình.

| **#** | **Vấn đề / xung đột trong v3** | **Cách sửa trong v4** | **Mục** |
| --- | --- | --- | --- |
| 1 | **Lệch phân bố detector → residual/CQR.** Tập train 60% vừa fine-tune detector vừa fit residual và hai hàm phân vị. Detector đã “thuộc” ảnh train nên bbox trên train chặt hơn bbox trên calib/test; mô hình học sai phân bố lúc triển khai. v3 chỉ giảm nhẹ bằng jitter. | Tách tập: **A** fine-tune detector, **B** (detector chưa từng thấy) fit trọng số hợp nhất, residual và phân vị. Jitter chỉ còn là ablation. | 4, 5.3 |
| 2 | **Rò rỉ nhãn GT vào đặc trưng.** Đặc trưng “cờ truncated” lấy từ nhãn KITTI, không có sẵn lúc suy luận thực tế. | Thay bằng cờ “bbox chạm biên ảnh” (tính từ bbox) và độ tin cậy detector. Nhãn truncated/occluded chỉ dùng để nhóm phân tích lỗi. | 5.3 |
| 3 | **Cue chiều rộng thiên lệch theo hướng xe.** Xe nhìn ngang có bbox rộng ≈ chiều dài xe (~4 m), không phải chiều rộng (~1,8 m); W trung bình theo class gây sai số hệ thống lớn. | Thêm cue **chiều cao Z_h** (ít phụ thuộc yaw); thêm tỉ lệ khung hình làm đặc trưng; báo cáo sai số theo hướng (alpha) để đo tác động. | 5.2 |
| 4 | **Split “theo cảnh” mâu thuẫn với “ưu tiên split Chen”.** Split Chen thường được biết là chia theo ảnh, không theo chuỗi gốc (⚠️ cần tự kiểm chứng). | Tự dựng split theo drive bằng file mapping của devkit. Chỉ dùng split công bố nếu kiểm chứng không có drive nằm ở hai tập. | 4 |
| 5 | **Ghép detection (IoU ≥ 0.5) loại FP/FN** làm kết quả đẹp hơn thực tế; detector recall thấp còn được lợi vì chỉ bị đánh giá trên ca dễ. | Ghép Hungarian, ngưỡng conf chốt trên V; báo cáo P/R/mAP riêng; so sánh detector trên **tập khớp chung** của cả 3 detector. | 5.1, 6 |
| 6 | **Bộ lọc mẫu chưa định lượng** (“quá nhỏ”, “che nặng”) trong khi calibration và test phải cùng phân bố thì CQR mới có bảo đảm. | Dùng mức lọc **Hard** của KITTI làm quần thể đánh giá cho B, C, T; Easy ⊂ Moderate ⊂ Hard báo cáo như tập con. | 4 |
| 7 | **Hợp nhất nghịch phương sai** giả định các cue độc lập và phương sai không đổi theo Z, nhưng chúng dùng chung nhiễu bbox và sai số tăng theo Z. | Hợp nhất trong không gian log, trọng số theo **hiệp phương sai** sai số log; residual dự đoán log-tỉ lệ nên khoảng CQR co giãn tự nhiên theo Z. | 5.2–5.4 |
| 8 | **Dùng một tiêu cự f** cho mọi cue; bbox từ YOLO (letterbox) chưa nói phải map về ảnh gốc; ảnh KITTI có nhiều kích thước. | fx cho cue ngang, fy cho cue dọc; map bbox về tọa độ ảnh gốc; cue không hợp lệ (bbox chạm biên) bị mask thay vì dùng bừa. | 5.2 |
| 9 | **Thống kê yếu.** 3 seed cho residual/CQR gần như vô nghĩa (đặc trưng cố định, XGBoost ~tất định); bootstrap theo ảnh bỏ qua tương quan giữa các khung cùng cảnh. | Độ phủ CQR: chia lại B∪C theo drive 20 lần. CI: **cluster bootstrap theo drive**. Detector vẫn 1 seed, nêu rõ. | 6 |
| 10 | **Ablation có thể chạm test**, trái nguyên tắc “test chỉ chạy một lần”. | Ablation chạy trên dự đoán out-of-fold (grouped CV) của B∪C. T chỉ chạy một lần với cấu hình đóng băng. | 6, 8.3 |
| 11 | **CQR chỉ bảo đảm độ phủ biên (marginal)** nhưng v3 lại đánh giá theo điều kiện mà không có cơ chế xử lý khi lệch. | Thêm biến thể **Mondrian CQR** (conformalize theo bin của Ẑ, không phải Z thật) làm ablation. | 5.4 |
| 12 | **Không tách được lỗi hình học và lỗi detector**: baseline tuần 1 chỉ chạy trên GT bbox, mô hình chính chạy trên bbox detector. | Chạy (a)–(d) trên cả GT bbox và bbox detector → phân rã sai số. Trực tiếp phục vụ G1, RQ2. | 5.5 |
| 13 | **Lộ trình mất cân bằng.** GPU nhàn rỗi tuần 1, quá tải tuần 2 (3 lần fine-tune tuần tự nhưng ghi “song song”); ngày 7 “đệm” bị chiếm; viết bài dồn 2 ngày cuối. | Xếp hàng fine-tune chạy nền từ tuần 1 ngày 3; đệm thật; viết theo mảng ngay khi có kết quả. | 8 |
| 14 | **Mâu thuẫn phần cứng.** Thầy nhấn mạnh 2 lần nhưng mặc định “chỉ GPU”; §5.6 lượng tử hóa YOLO bản n trong khi so sánh chính dùng bản s. | Tier 1 luôn làm (ONNX CPU, FP16, độ trễ từng khâu). Tier 2 (INT8) nếu xác nhận, dùng YOLO11n huấn luyện cùng giao thức. | 5.6 |
| 15 | **Phạm vi lớp không nhất quán.** Tiêu đề “Vehicle” nhưng mở rộng Pedestrian/Cyclist, trong khi cue chiều rộng và prior W không hợp với người. | Car là chính; Van/Truck tùy chọn, báo cáo riêng. Pedestrian/Cyclist chuyển sang hướng phát triển. | 1, 4 |
| 16 | **Kalman/KITTI Tracking** thêm pipeline dữ liệu thứ hai và có nguy cơ chồng lấn khung với Object (cùng raw drive). | Bỏ khỏi kế hoạch chính, chuyển sang hướng phát triển. | 5.6 |
| 17 | **So sánh 3 detector “cùng giao thức”** nhưng chưa quy định giao thức; 1 seed nên chênh lệch nhỏ có thể chỉ là nhiễu huấn luyện. | Cố định công thức huấn luyện chung; kết luận RQ2 mang tính mô tả, kèm CI; không xếp hạng nếu CI chồng lấn. | 5.1, 3 |

## 1. Tóm tắt nhanh
| **Hạng mục** | **Nội dung** |
| --- | --- |
| Bài toán | Ước lượng khoảng cách (độ sâu Z) tới xe phía trước bằng một camera (ITS/ADAS). |
| Ý tưởng | Hình học pinhole với 3 cue (chiều rộng, chiều cao, cạnh dưới) hợp nhất trong không gian log + bộ hiệu chỉnh residual học được + khoảng tin cậy CQR có bảo đảm độ phủ. |
| Dữ liệu | KITTI Object Detection, 7.481 ảnh có nhãn; chia theo drive thành 5 tập A/V/B/C/T (§4). |
| Detector | YOLOv5su, YOLOv8s, YOLO11s (Ultralytics), cùng công thức huấn luyện. |
| Đóng góp bảo vệ được | (1) Tách vai trò và sai số từng cue hình học, phân rã lỗi hình học vs lỗi detector; (2) so sánh có kiểm soát 3 thế hệ YOLO trên tập khớp chung; (3) CQR trong không gian log với thiết kế tách dữ liệu đúng, phân tích độ phủ theo điều kiện. |
| Thời gian | 3 tuần (bản rút gọn 2 tuần), cộng 1–2 ngày chuẩn bị. |
| Mức tính mới | Tổ hợp và ứng dụng; phù hợp báo cáo môn học/hội nghị ứng dụng; không claim “đầu tiên”. |
| Tài nguyên | RTX 5060 8GB; dự phòng Colab/Kaggle; XGBoost/MLP/CQR chạy CPU. |

**Pipeline tổng thể**

| Ảnh → YOLO → bbox (map về ảnh gốc) → lọc/khớp → 3 cue: **Z_w** (chiều rộng), **Z_h** (chiều cao), **Z_g** (cạnh dưới) → Hợp nhất log-space (trọng số theo hiệp phương sai) → Z_d → Residual r̂ = f(x) trên log-tỉ lệ → Ẑ = Z_d · exp(r̂) → CQR trên r → khoảng [Z_lo, Z_hi] = Z_d · exp([r_lo, r_hi]) |
| --- |

## 2. Cần chốt trước tuần 1
| **#** | **Việc cần chốt** | **Vì sao quan trọng** | **Mặc định nếu chưa có trả lời (v4)** |
| --- | --- | --- | --- |
| 1 | ⚠️ Yêu cầu phần cứng: chỉ GPU, hay cần chạy trên thiết bị giới hạn (ARM/mobile/FPGA)? | Thầy đã nhấn mạnh phần cứng hai lần nên không nên mặc định bỏ qua. | **Tier 1** (ONNX CPU + FP16 + độ trễ từng khâu). Tier 2 (INT8) chỉ khi được xác nhận. |
| 2 | ⚠️ Danh sách lớp. | Prior kích thước và cue chiều rộng chỉ hợp lý cho xe; lớp ít mẫu làm kết quả nhiễu. | Car chính; Van/Truck tùy chọn, báo cáo riêng, không gộp trung bình. Bỏ Pedestrian/Cyclist. |
| 3 | ⚠️ 3 tuần là toàn bộ đề tài hay chỉ giai đoạn đầu? | Quyết định mức rút gọn. | 3 tuần, toàn bộ. |
| 4 | Cách đánh giá: test KITTI (7.518 ảnh) không có nhãn công khai, chỉ chấm qua server và không có nhãn 3D để đánh giá Z. | Ảnh hưởng cách chia dữ liệu. | Đánh giá trên tập T tách từ 7.481 ảnh có nhãn. |

**Tự chốt ngay và ghi vào configs (sau này rất khó đổi)**

- ⚠️ Khoảng cách thật là độ sâu **Z** (location_z trong nhãn KITTI), không phải khoảng cách Euclid, vì công thức pinhole trả về Z.

- Quần thể đánh giá: bỏ DontCare; GT thỏa mức lọc **Hard** của KITTI (chiều cao bbox ≥ 25 px, occluded ≤ 2, truncated ≤ 0,5). Áp dụng đồng nhất cho B, C, T.

- Chọn **YOLOv5su** (anchor-free, Ultralytics) thay vì repo YOLOv5 gốc để cả 3 detector dùng cùng recipe; ghi rõ tên trong bài.

- Công thức huấn luyện chung cho 3 detector: cùng trọng số tiền huấn luyện COCO, imgsz, số epoch, patience, augmentation, ngưỡng NMS. Chốt imgsz trên V (thử 640 rồi 960/1280 nếu VRAM cho phép), dùng một cấu hình cho cả ba.

## 3. Bối cảnh, khoảng trống, câu hỏi nghiên cứu
Paper tham chiếu (Ni et al., 2026) tích hợp YOLOv5 với hình học phối cảnh nhưng thiếu công thức tường minh, phụ thuộc kích thước xe biết trước, baseline yếu, thiếu ablation và thiếu bảng sai số theo dải khoảng cách. Các hướng gần (Dist-YOLO, DisNet, AGL, DECADE) cải thiện độ chính xác điểm nhưng phần lớn chưa đánh giá có hệ thống mức tin cậy của từng dự đoán, trong khi đây là thông tin quan trọng cho quyết định an toàn.

| **Mã** | **Khoảng trống** |
| --- | --- |
| G1 | Hình học thuần chịu sai số hệ thống (prior kích thước, hướng xe, nhiễu cạnh bbox); ít công trình tách sai số từng cue theo dải khoảng cách và tách lỗi hình học khỏi lỗi detector. |
| G2 | Thiếu so sánh công bằng nhiều thế hệ YOLO nhẹ (v5, v8, 11) trên cùng pipeline ranging, cùng giao thức và cùng tập mẫu. |
| G3 | Thiếu ước lượng bất định có hiệu chuẩn cho ranging dựa trên bbox, kèm kiểm tra độ phủ theo điều kiện. |

| **RQ** | **Câu hỏi** | **Giả thuyết kiểm chứng** |
| --- | --- | --- |
| RQ1 | Residual cải thiện bao nhiêu so với pinhole thuần (từng cue và hợp nhất) ở từng dải khoảng cách? Bao nhiêu sai số đến từ hình học, bao nhiêu từ detector? | H1: giảm AbsRel nhiều nhất ở 0–20 m (bbox chuẩn hơn); ở >30 m cải thiện nhỏ hơn vì lỗi hình học và độ phân giải chiếm ưu thế. Cue chiều rộng lệch mạnh với xe nhìn ngang, cue chiều cao ít lệch hơn. |
| RQ2 | Độ chính xác ranging khác nhau thế nào giữa các detector, liên hệ ra sao với IoU và sai lệch cạnh dưới bbox? | H2: detector có IoU và cạnh dưới chính xác hơn cho ranging tốt hơn, không nhất thiết là detector có mAP cao nhất. Vì detector chỉ 1 seed, chỉ kết luận khi CI không chồng lấn. |
| RQ3 | Khoảng CQR có đạt độ phủ danh nghĩa không, và co giãn thế nào theo khoảng cách, che khuất, cắt biên, hướng xe? | H3: độ phủ biên đạt danh nghĩa vì B∪C, T đều chưa được detector thấy và cùng bộ lọc; độ phủ có điều kiện có thể lệch ở >50 m và ca bị che, Mondrian CQR giảm một phần độ lệch. |

## 4. Dữ liệu và giao thức
**Nguồn.** KITTI Object Detection: 7.481 ảnh có nhãn (train) và 7.518 ảnh không có nhãn công khai (test). Toàn bộ phân chia thực hiện trên 7.481 ảnh có nhãn.

### 4.1 Nguyên tắc chia
- **Chia theo drive** (dùng train_mapping.txt trong devkit để suy ra drive gốc), không chia ngẫu nhiên theo ảnh. Gán drive vào tập theo kiểu phân tầng theo nhóm (greedy) để cân bằng phân bố Z và class giữa các tập.

- ⚠️ Số drive không lớn (cỡ vài chục, cần kiểm tra thực tế): kiểm tra mỗi tập có đủ mẫu; nếu một dải khoảng cách có dưới ~100 xe ở C hoặc T thì gộp dải hoặc báo cáo kèm cảnh báo.

- Kiểm tra bắt buộc: không drive nào xuất hiện ở hai tập; so sánh phân bố Z và class giữa các tập (ví dụ KS test). Đóng băng split vào splits/ và ghi hash vào log.

- Split công bố (như Chen et al.) chỉ dùng nếu kiểm chứng bằng mapping không có drive chung; nếu không, tự chia và ghi rõ.

### 4.2 Năm tập dữ liệu
| **Tập** | **Tỉ lệ** | **Detector thấy?** | **Vai trò** |
| --- | --- | --- | --- |
| **A** Detector-train | 50% (~3.700 ảnh) | Có | Fine-tune 3 detector. Ước lượng prior W_eff, H_obj, H_cam từ nhãn. |
| **V** Detector-val | 5% (~370 ảnh) | Có (early stopping) | Early stopping, chọn imgsz, chọn ngưỡng conf. Không dùng để tuning residual. |
| **B** Residual-train | 20% (~1.500 ảnh) | Không | Fit trọng số hợp nhất, residual, hai hàm phân vị. Tuning bằng grouped CV theo drive trong B. |
| **C** Calibration | 10% (~750 ảnh) | Không | Chỉ dùng để conformalize CQR. Không dùng cho việc nào khác. |
| **T** Test | 15% (~1.100 ảnh) | Không | Báo cáo cuối, chạy một lần khi mọi cấu hình đã đóng băng. |

| **Vì sao thiết kế này:** bbox trên B, C, T đều do một detector chưa từng thấy ảnh sinh ra, nên residual học đúng phân bố lỗi bbox lúc triển khai, và B, C, T cùng phân bố (exchangeable) theo drive. Đánh đổi: detector chỉ học trên ~50% dữ liệu nên mAP thấp hơn các paper dùng toàn bộ train; điều này chấp nhận được vì mục tiêu là ranging, không phải mAP. Nếu còn thời gian: cross-fitting 2-fold (huấn luyện detector trên nửa này, suy luận nửa kia) để tận dụng nhiều dữ liệu hơn. |
| --- |

### 4.3 Lớp, camera, tọa độ
- **Lớp.** Car bắt buộc. Van/Truck tùy chọn, có prior W, H riêng và báo cáo riêng. Pedestrian/Cyclist ngoài phạm vi.

- **Calibration camera.** Dùng P2 (và R0_rect) riêng từng ảnh: fx, fy, cx, cy. Không dùng một f cố định.

- **Tọa độ.** Kích thước ảnh KITTI không đồng nhất (ví dụ 1224×370, 1242×375, 1241×376). Mọi bbox của YOLO phải map ngược từ không gian letterbox về pixel ảnh gốc trước khi tính cue.

## 5. Phương pháp
### 5.1 Phát hiện và khớp
- Fine-tune YOLOv5su, YOLOv8s, YOLO11s trên A (early stopping bằng V) với công thức huấn luyện chung (§2). Một seed mỗi detector, nêu rõ trong bài.

- Ngưỡng conf chốt trên V bằng một quy tắc duy nhất (ví dụ F1 tối đa) và dùng cho cả 3 detector.

- **Khớp Greedy** theo confidence giảm dần với IoU ≥ 0,5 (Quyết định D15, thay cho Hungarian): ưu tiên GT Hard, kiểm tra trùng lặp (duplicate -> FP), detection khớp với non-Hard hoặc DontCare bị bỏ qua. Ranging chỉ đánh giá trên detection khớp, nhưng **luôn báo cáo kèm** precision, recall, mAP@0,5 và mAP@0,7 (giao thức KITTI) để người đọc thấy phần bị loại.

- Khi so sánh detector, dùng thêm **tập khớp chung** (những GT mà cả 3 detector đều bắt được) để tránh thiên lệch do recall khác nhau.

### 5.2 Ước lượng hình học
| **Cue** | **Công thức** | **Ghi chú** |
| --- | --- | --- |
| Chiều rộng | Z_w = fx · W_eff / w | W_eff là chiều rộng bbox hiệu dụng trung bình theo class, ước lượng từ nhãn A. Thiên lệch theo hướng xe (xe ngang rộng ≈ dài), nên sai số theo alpha là một phát hiện cần báo cáo. |
| Chiều cao | Z_h = fy · H_obj / h,  h = y_bottom − y_top | H_obj là chiều cao xe trung bình theo class; gần như không đổi theo yaw nên ổn định hơn Z_w. Nhạy với che khuất/cắt biên ở phía trên. |
| Cạnh dưới | Z_g = fy · H_cam / (y_bottom − y_horizon) | Giả định mặt đường phẳng cục bộ. y_horizon lấy từ cy của P2 (camera không nghiêng), có thể tinh chỉnh bằng dữ liệu A. H_cam ước lượng từ median location_y trong nhãn A. |

- **Mask cue không hợp lệ:** bbox chạm biên trái/phải làm Z_w sai; chạm biên trên/dưới làm Z_h và Z_g sai; y_bottom ≤ y_horizon + ε làm Z_g vô nghĩa. Cue không hợp lệ đặt là NaN, kèm cờ hợp lệ làm đặc trưng.

- **Hợp nhất trong không gian log:** ln Z_d = Σ w_k · ln Z_k trên các cue hợp lệ, với w = Σ⁻¹1 / (1ᵀΣ⁻¹1) và Σ là hiệp phương sai của sai số log ước lượng trên B (theo class). Cách này xử lý được tương quan giữa các cue và sai số tăng theo Z.

- **Hạn chế cần nêu:** giả định mặt đường phẳng sai ở dốc và khúc cua; Z_g nhạy với lỗi cạnh dưới; cue chiều rộng phụ thuộc hướng xe.

### 5.3 Hiệu chỉnh residual
- **Mục tiêu:** r = ln Z_gt − ln Z_d (log-tỉ lệ). Dự đoán Ẑ = Z_d · exp(r̂). Sai số tương đối nhờ vậy được học đồng đều theo khoảng cách.

- **Mô hình:** XGBoost (chính), MLP nhỏ (ablation).

- **Đặc trưng (chỉ dùng thông tin có lúc suy luận):** w, h, w/h, y_bottom − cy, (cx − cx0)/fx, class, độ tin cậy detector, cờ bbox chạm biên (4 phía), ln Z_w, ln Z_h, ln Z_g và cờ hợp lệ. **Không** dùng nhãn truncated/occluded của KITTI.

- **Huấn luyện:** chỉ fit trên B, tuning siêu tham số bằng grouped CV theo drive trong B. Không thấy A, V, C, T.

- **Jitter bbox:** không còn là biện pháp chính (lệch phân bố đã xử lý bằng tách tập). Giữ như ablation.

- **Chẩn đoán:** so sánh phân bố sai lệch bbox (IoU, sai lệch cạnh dưới) giữa A và B để minh họa vì sao phải tách tập.

### 5.4 Bất định: CQR
- Fit hai hàm phân vị 5% và 95% của r trên B (XGBoost với hàm mất mát quantile), cùng bộ đặc trưng như §5.3.

- Conformalize trên C: điểm phi tuân thủ E = max(q_lo − r, r − q_hi); hiệu chỉnh khoảng bằng phân vị ⌈(n+1)(1−α)⌉/n của E, α = 0,1. Khoảng cuối trong Z lấy qua biến đổi đơn điệu exp nên giữ nguyên độ phủ.

- **Điều kiện áp dụng:** C phải đi qua đúng pipeline như T (cùng detector, cùng ngưỡng conf, cùng ghép, cùng bộ lọc). Bảo đảm là độ phủ **biên** dưới exchangeability; không suy ra độ phủ có điều kiện.

- **Biến thể Mondrian CQR:** conformalize riêng theo bin của Ẑ (biến quan sát được, không dùng Z thật). Cần đủ mẫu mỗi bin ở C, nếu thiếu thì gộp bin.

- **Đánh giá:** độ phủ thực nghiệm so với 90%; độ rộng khoảng (tỉ lệ Z_hi/Z_lo) và interval score; tất cả theo dải khoảng cách, che khuất, cắt biên, hướng xe.

Lý do chọn CQR (Romano, Patterson, Candès, NeurIPS 2019): có lý thuyết bảo đảm hữu hạn mẫu và thích ứng theo đặc trưng, vừa khung thời gian hơn việc thử nhiều phương pháp bất định.

### 5.5 Baselines và mô hình đề xuất
| **Ký hiệu** | **Mô tả** |
| --- | --- |
| (a) | Pinhole theo chiều rộng (Z_w) |
| (b) | Pinhole theo chiều cao (Z_h) |
| (c) | Pinhole theo cạnh dưới (Z_g) |
| (d) | Hợp nhất log-space (a)+(b)+(c) |
| (e) | Hồi quy trực tiếp ln Z từ đặc trưng bbox, **không** dùng ln Z_k (kiểm tra hình học có thực sự cần) |
| (f) | Đề xuất: (d) + residual |
| (g) | Đề xuất đầy đủ: (f) + CQR (và biến thể Mondrian) |

- **Phân rã sai số:** chạy (a)–(d) trên **cả GT bbox** (chặn dưới của lỗi hình học) **và bbox detector**. Phần chênh lệch là đóng góp của lỗi detector.

- **Ablation** (đánh giá trên dự đoán out-of-fold của B∪C, không chạm T): bỏ từng nhóm đặc trưng; bỏ/bật jitter; XGBoost vs MLP; bỏ từng cue hình học; CQR vs split conformal thường vs Mondrian.

### 5.6 Phần cứng
- **Tier 1 (luôn làm, ~0,5 ngày):** đo độ trễ và FPS từng khâu (detector, hình học, residual, CQR); FP16 trên GPU; xuất ONNX và đo trên CPU cho cả 3 detector.

- **Tier 2 (chỉ khi §2.1 xác nhận cần):** huấn luyện thêm YOLO11n cùng giao thức, lượng tử hóa INT8 (TFLite/NCNN), đo độ chính xác ranging, độ phủ CQR và độ trễ trên board/mobile. Cần thêm khoảng 2–3 ngày, bù bằng cách cắt YOLOv5su.

- **Hướng phát triển (ngoài kế hoạch):** làm mượt theo thời gian (Kalman/EMA) trên KITTI Tracking (⚠️ kiểm tra chồng lấn raw drive với Object), mở rộng lớp Pedestrian/Cyclist.

## 6. Chỉ số và thống kê
| **Nhóm** | **Nội dung** |
| --- | --- |
| Sai số khoảng cách | MAE, RMSE, AbsRel, δ<1,25; theo dải **0–10, 10–20, 20–30, 30–50, ****>****50 m** (thống nhất ở mọi nơi trong tài liệu); theo class; theo Easy/Moderate/Hard. |
| Detector | Precision, recall, mAP@0,5 và mAP@0,7; số GT bị bỏ sót; sai lệch IoU và cạnh dưới bbox. |
| Bất định | Độ phủ và độ rộng khoảng; interval score; theo dải khoảng cách, che khuất, cắt biên, hướng xe. Độ phủ theo Z thật chỉ là chẩn đoán, không phải bảo đảm. |
| Hiệu năng | FPS, độ trễ từng khâu (§5.6). |
| Thống kê | Detector: 1 seed, ghi rõ. Độ phủ CQR: chia lại B∪C theo drive 20 lần, báo cáo mean ± std. So sánh cặp chính (ví dụ (d) vs (f), (f) vs (g), giữa detector): **cluster bootstrap theo drive** (≥1.000 lần) trên T, báo cáo CI 95%. Chỉ dùng cụm “có ý nghĩa thống kê” khi CI của hiệu không chứa 0. |
| Phân tích lỗi | Theo class, occluded, truncated, hướng xe (alpha), dải khoảng cách: phần “chỉ ra khi nào hình học thất bại”. Nhãn occluded/truncated chỉ dùng để nhóm, không đưa vào mô hình. |

## 7. Tài nguyên
- **GPU:** RTX 5060 8GB, đủ fine-tune các bản “s” với batch vừa phải. ⚠️ Cần PyTorch hỗ trợ kiến trúc GPU thế hệ này (thường CUDA ≥ 12.8); kiểm tra trang chính thức PyTorch trước khi cài.

- **Huấn luyện tuần tự** trên 1 GPU (không chạy song song 3 detector trên 8GB). Đo thời gian 1 epoch ở bước chuẩn bị để lên lịch xếp hàng.

- **Dự phòng:** Colab/Kaggle nếu cần chạy song song.

- **XGBoost/MLP/CQR:** chạy trên CPU, không tốn GPU.

## 8. Lộ trình chi tiết theo tuần
### 8.0 Chuẩn bị (1–2 ngày trước tuần 1)
- Cài PyTorch hỗ trợ RTX 5060, ultralytics, xgboost, scikit-learn, numpy, pandas, matplotlib, onnxruntime. Chạy thử 1 epoch fine-tune để đo thời gian và VRAM.

- Tải KITTI Object: image_2, label_2, calib và devkit (có mapping sang KITTI raw: train_mapping.txt, train_rand.txt).

- Dựng thư mục: data/, splits/, configs/, runs/, results/, notebooks/. Dùng git và tag phiên bản.

- Tạo log thí nghiệm (seed, hash split, phiên bản code, cấu hình detector).

- **Với thầy:** gửi một tin nhắn gom 4 câu ở §2. Nếu chưa có trả lời, chạy theo mặc định của §2.

### 8.1 Tuần 1: dữ liệu, split, khởi động GPU, baseline hình học
**Mục tiêu:** split đóng băng; 3 detector đang huấn luyện nền; baseline (a)–(d) trên GT bbox; code đánh giá tái sử dụng được.

| **Ngày** | **Việc** | **Kết quả** |
| --- | --- | --- |
| 1 | Tải, kiểm tra KITTI. Viết loader đọc ảnh, nhãn, P2/R0_rect từng ảnh. Áp bộ lọc Hard, thống kê số xe theo dải. | Loader chạy hết 7.481 ảnh không lỗi. |
| 2 | Dựng split A/V/B/C/T theo drive, kiểm tra không rò rỉ và phân bố Z/class. Đóng băng. Chuyển A, V sang định dạng YOLO (lưu thông tin resize để map ngược). | splits/ + báo cáo kiểm tra; dataset YOLO. |
| 3 | **Khởi động hàng đợi fine-tune** YOLOv8s → YOLO11s → YOLOv5su chạy nền tuần tự. Trong lúc đó: thống kê W_eff, H_obj, H_cam, y_horizon từ nhãn A. | Hàng đợi chạy; bảng thống kê prior. |
| 4 | Cài 3 cue (a)(b)(c) có mask hợp lệ; cài hợp nhất log-space (d) với ước lượng hiệp phương sai trên B. | Bốn baseline chạy trên GT bbox. |
| 5 | Viết eval.py: MAE, RMSE, AbsRel, δ<1,25 theo 5 dải, class, Easy/Moderate/Hard; cluster bootstrap theo drive. Có test đơn vị. | eval.py + test. |
| 6 | Chạy (a)–(d) trên GT bbox của B (out-of-fold nếu cần). Vẽ sai số theo khoảng cách và theo alpha. | Bảng baseline, biểu đồ. |
| 7 | **Ngày đệm thật:** xử lý sự cố, kiểm tra tiến độ hàng đợi GPU. Nếu rảnh: viết nháp mục Data và Related Work. | Nháp bài (mảng dữ liệu). |

| **Điều kiện sang tuần 2:** loader, split, metric chạy đúng; ít nhất 1 detector đã hội tụ; hợp nhất (d) không tệ hơn cue tốt nhất riêng lẻ ở đa số dải. Nếu không: kiểm tra lại y_horizon, H_cam, W_eff, mask cue trước. |
| --- |

### 8.2 Tuần 2: bbox detector, residual, CQR
**Mục tiêu:** bảng (a)–(g) cho ít nhất 2 detector trên dữ liệu dev (B∪C) và đường cong coverage.

| **Ngày** | **Việc** | **Kết quả** |
| --- | --- | --- |
| 1 | Kiểm tra checkpoint và mAP trên V. Chạy suy luận trên B, C, T (và A để chẩn đoán); áp ngưỡng conf, khớp Hungarian, map bbox về ảnh gốc, trích đặc trưng. | File đặc trưng theo detector; P/R/mAP. |
| 2 | Chạy (a)–(d) trên bbox detector; so với GT bbox để phân rã sai số. Chẩn đoán lệch bbox giữa A và B. | Bảng phân rã hình học vs detector. |
| 3 | Fit residual XGBoost trên B (log-tỉ lệ), tuning bằng grouped CV; baseline trực tiếp (e). | Mô hình (f), (e); sai số out-of-fold. |
| 4 | MLP nhỏ; ablation nhóm đặc trưng, bỏ cue, jitter. | Bảng ablation (điểm). |
| 5 | Fit hai phân vị 5%/95% trên B; conformalize trên C; kiểm tra độ phủ tổng. | Khoảng [Z_lo, Z_hi] cho mọi bbox. |
| 6 | Mondrian CQR và split conformal; chia lại B∪C theo drive 20 lần; độ phủ theo điều kiện. | Bảng và biểu đồ coverage (dev). |
| 7 | **Ngày đệm thật:** sửa lỗi, chạy lại phần lỡ. Nếu rảnh: viết nháp mục Method và Experiments. | Nháp bài (mảng phương pháp). |

| **Điều kiện sang tuần 3:** có (a)–(g) cho ít nhất 2 detector; CQR có độ phủ tổng gần 90% trên dev (lệch có giải thích được). **Đóng băng cấu hình** (git tag) trước khi chạm T. Nếu trễ: bỏ YOLOv5su (bản 2 tuần). |
| --- |

### 8.3 Tuần 3: chạy T một lần, phân tích, hiệu năng, hoàn thiện bài
**Mục tiêu:** bản thảo hoàn chỉnh.

| **Ngày** | **Việc** | **Kết quả** |
| --- | --- | --- |
| 1 | Chạy T **một lần** bằng một script duy nhất với cấu hình đã đóng băng. Cluster bootstrap CI cho (d) vs (f), (f) vs (g), giữa detector (kể cả tập khớp chung). | Bảng chính cuối cùng. |
| 2 | Phân tích lỗi theo dải, class, occluded, truncated, alpha. | Bảng và biểu đồ lỗi. |
| 3 | Phân tích độ phủ/độ rộng CQR theo điều kiện trên T; đối chiếu Mondrian; kiểm tra exchangeability có vỡ ở đâu. | Biểu đồ coverage theo điều kiện. |
| 4 | Đo Tier 1 (§5.6): FPS/độ trễ từng khâu, ONNX CPU. Nếu Tier 2 được xác nhận: bắt đầu INT8 với YOLO11n. | Bảng hiệu năng. |
| 5 | Hình định tính: ảnh có bbox, Z dự đoán, khoảng tin cậy; ca hình học thất bại (xe ngang, dốc, bị cắt). | 6–10 hình. |
| 6 | Hoàn thiện Introduction, Experiments, Limitations; bổ sung tài liệu tham khảo. | Bản thảo đầy đủ. |
| 7 | Rà checklist §11, kiểm tra từng trích dẫn từ nguồn, đọc lại và sửa. | Bản nộp. |

**Khi viết:** không claim “đầu tiên”; liệt kê tiền lệ; nêu hạn chế (chỉ KITTI, mặt đường phẳng, phạm vi lớp, detector 1 seed, detector chỉ học trên ~50% dữ liệu); bổ sung 10–15 tài liệu tham khảo và kiểm tra từng trích dẫn.

### 8.4 Bản rút gọn 2 tuần và thứ tự cắt giảm
Bản 2 tuần: giảm còn 2 detector (YOLOv8s, YOLO11s), **giữ nguyên** thiết kế tách tập, residual, CQR và phân tích lỗi. Nếu trễ, cắt theo thứ tự:

- YOLOv5su (nếu cần Tier 2 thì cắt trước).

- MLP trong ablation (giữ XGBoost).

- Mondrian CQR (giữ CQR chuẩn và split conformal).

- Van/Truck (giữ Car).

**Không cắt:** tách tập A/B/C/T theo drive, residual, CQR, phân tích lỗi theo dải khoảng cách, Tier 1 phần cứng.

**Nếu là giai đoạn đầu của đề tài dài hơn (⚠️ §2.3):** tuần 4–5 mở rộng Tier 2, cross-fitting, Pedestrian/Cyclist hoặc làm mượt theo thời gian; tuần 6 viết báo cáo/bài nộp đầy đủ.

## 9. Rủi ro và phương án
| **Rủi ro** | **Mức** | **Biện pháp** |
| --- | --- | --- |
| Bị đánh giá là đóng góp tăng dần | Cao | Không claim “đầu tiên”; nhấn mạnh phân rã sai số, thiết kế tách dữ liệu đúng và CQR có bảo đảm. |
| Thiếu thời gian | Cao | Bản 2 tuần; hàng đợi GPU chạy nền từ tuần 1; thứ tự cắt giảm ở §8.4. |
| Rò rỉ dữ liệu giữa các tập | Trung bình–Cao | Chia theo drive; kiểm tra tự động; C chỉ conformalize; ablation không chạm T; không dùng nhãn GT làm đặc trưng. |
| Ít drive → phân bố các tập lệch, dải xa thiếu mẫu | Trung bình–Cao | Gán drive phân tầng; kiểm tra phân bố; gộp dải >30 m nếu dưới ~100 mẫu; báo cáo cảnh báo. |
| Không đáp ứng yêu cầu phần cứng của thầy | Trung bình | Tier 1 luôn làm; xác nhận §2.1 sớm; Tier 2 nếu cần. |
| Detector học ít dữ liệu (50%) nên yếu | Trung bình | Nêu rõ đánh đổi; cross-fitting 2-fold nếu còn thời gian. |
| Cue chiều rộng thiên lệch theo hướng xe | Trung bình | Cue chiều cao, tỉ lệ khung hình làm đặc trưng, báo cáo theo alpha. |
| Mặt đường phẳng sai ở dốc/cua | Trung bình | Nêu trong hạn chế; residual bù một phần; báo cáo lỗi theo drive để phát hiện ca dốc. |
| CQR không đạt độ phủ có điều kiện | Trung bình | Báo cáo trung thực; Mondrian CQR; đây có thể là phát hiện, không phải thất bại. |
| Chỉ dùng KITTI, khó khái quát | Thấp–Trung bình | Nêu phạm vi (Đức, ban ngày); nếu còn thời gian thử một tập nhỏ khác. |

## 10. Tính mới và định vị
**Đã có tiền lệ (không claim mới)**

- Geometry + học máy cho ranging trên KITTI: Dist-YOLO (2022) dự đoán khoảng cách trực tiếp trong đầu YOLOv3; DisNet dùng MLP từ đặc trưng bbox (rất gần mô-đun residual ở đây); DECADE (2024) so sánh nhiều biến thể YOLOv8; AGL (2026) và Hybrid Dist-YOLOv3 (2025) là cải tiến gần nhất.

- Bất định có hiệu chuẩn: f-Cal (2021) cho object detection và depth; một nghiên cứu 2024 dùng conformal prediction cho khoảng cách trong adaptive cruise control.

**Phần bảo vệ được**

Tổ hợp cụ thể gồm (1) tách vai trò và sai số từng cue hình học, phân rã lỗi hình học vs lỗi detector; (2) so sánh có kiểm soát qua 3 thế hệ YOLO cùng giao thức, trên tập khớp chung; (3) CQR trong không gian log với thiết kế tách dữ liệu đảm bảo B, C, T chưa được detector thấy, kèm phân tích độ phủ theo điều kiện.

**Mức tính mới:** dạng tổ hợp và ứng dụng; phù hợp báo cáo môn học/hội nghị ứng dụng, không đủ cho venue top-tier.

## 11. Checklist trước khi nộp
☐  Xác nhận §2 (phần cứng, danh sách lớp, thời lượng thực) bằng văn bản

☐  Split theo drive, kiểm tra tự động không rò rỉ; A/V/B/C/T đóng băng và có hash trong log

☐  B, C, T chưa từng được detector thấy; C chỉ dùng để conformalize

☐  Không có đặc trưng nào lấy từ nhãn GT (truncated, occluded, alpha) trong mô hình

☐  Dùng P2 riêng từng ảnh (fx, fy, cx, cy); bbox map về ảnh gốc

☐  Cùng bộ lọc Hard cho B, C, T; Easy/Moderate/Hard báo cáo như tập con

☐  Kết quả detector kèm P/R/mAP; so sánh detector trên tập khớp chung

☐  AbsRel/MAE theo dải khoảng cách, theo class riêng, theo hướng xe

☐  Độ phủ CQR kèm điều kiện (khoảng cách, che khuất, cắt biên, hướng), không chỉ một số tổng; mean ± std qua 20 lần chia lại

☐  CI bằng cluster bootstrap theo drive; ghi rõ detector chỉ 1 seed

☐  Ablation chạy trên out-of-fold của B∪C; T chạy đúng một lần; mọi tuning không dùng T

☐  Có bảng độ trễ Tier 1 (và Tier 2 nếu được yêu cầu)

☐  Không claim “đầu tiên”; liệt kê tiền lệ (Dist-YOLO, DisNet, AGL, DECADE, CQR, f-Cal) trong Related Work

☐  Nêu hạn chế: chỉ KITTI, mặt đường phẳng, hướng xe, phạm vi lớp, detector 1 seed và ~50% dữ liệu

## 12. Tài liệu tham khảo
- Z. Ni, J. Shi, L. Li, C. Ni. "Real-time Vehicle Detection and Distance Estimation: Soft-sensor Approach Using Optimized YOLOv5 and Perspective Geometry." Sensors and Materials 38(1), 2026.

- M. Vajgl, P. Hurtik, T. Nejezchleba. "Dist-YOLO: Fast Object Detection with Distance Estimation." Applied Sciences 12(3):1354, 2022.

- M.A. Haseeb et al. "DisNet: A novel method for distance estimation from monocular camera." (workshop paper, dẫn qua DECADE)

- "DECADE: Towards Designing Efficient-yet-Accurate Distance Estimation Modules for Collision Avoidance in Mobile ADAS." arXiv:2410.19336, 2024.

- "Lightweight Monocular Distance Estimation via Anisotropic Geometry Loss for Low-Light Driving Environments." Sensors 26, 2026.

- L. Bertoni, S. Kreiss, A. Alahi. "MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation." ICCV 2019.

- Y. Romano, E. Patterson, E. Candès. "Conformalized Quantile Regression." NeurIPS 2019 (arXiv:1905.03222).

- D. Bhatt et al. "f-Cal: Calibrated aleatoric uncertainty estimation from neural networks for robot perception." arXiv:2109.13913.

- E. Dagan, O. Mano, G.P. Stein, A. Shashua. "Forward Collision Warning with a Single Camera." IEEE IV Symposium, 2004.

*Cần bổ sung 10–15 tài liệu về monocular 3D detection, conformal prediction cho perception (gồm conformal Mondrian/conditional coverage) và ranging trên KITTI sau khi khảo sát Google Scholar/Semantic Scholar; kiểm tra lại trích dẫn trực tiếp từ nguồn (đặc biệt DisNet, MonoLoco) trước khi nộp.*
