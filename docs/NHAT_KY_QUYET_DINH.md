# Nhật ký quyết định: Distance Estimation – KITTI – YOLO

> Nguồn chuẩn của thiết kế là `KE_HOACH_V4.md`. File này ghi lại **những gì đã chốt, đang chờ, và tiến độ**.
> Cuối mỗi phiên làm việc, thêm một mục mới vào phần "Nhật ký theo phiên" (mới nhất ở trên cùng).
> Quy ước: ✅ đã chốt · ⏳ đang chờ xác nhận · ⚠️ cần kiểm chứng trên dữ liệu thật.

---

## 1. Việc cần chốt với thầy (§2 của kế hoạch)

Chưa có trả lời từ thầy. Trong lúc chờ, chạy theo mặc định của kế hoạch.

| # | Câu hỏi | Trả lời của thầy | Mặc định đang dùng |
|---|---------|------------------|--------------------|
| 1 | Phần cứng: chỉ GPU hay cần thiết bị giới hạn (ARM/mobile/FPGA)? | ⏳ chưa có | Tier 1 (ONNX CPU + FP16 + độ trễ từng khâu). Tier 2 (INT8) chỉ khi được xác nhận |
| 2 | Danh sách lớp | ⏳ chưa có | Car là chính; Van/Truck tùy chọn, báo cáo riêng, không gộp trung bình; bỏ Pedestrian/Cyclist |
| 3 | 3 tuần là toàn bộ đề tài hay giai đoạn đầu? | ⏳ chưa có | 3 tuần, toàn bộ |
| 4 | Cách đánh giá | ⏳ chưa có | Tập T tách từ 7.481 ảnh có nhãn (test KITTI không có nhãn công khai) |

---

## 2. Quy ước cố định (đã chốt, không đổi nếu không có lý do mạnh)

- ✅ **Target khoảng cách:** độ sâu **Z = `location_z`** trong nhãn KITTI, không dùng khoảng cách Euclid.
- ✅ **Quần thể đánh giá:** bỏ DontCare; GT thỏa mức lọc **Hard** của KITTI (chiều cao bbox ≥ 25 px, occluded ≤ 2, truncated ≤ 0,5). Áp dụng đồng nhất cho B, C, T. Easy ⊂ Moderate ⊂ Hard báo cáo như tập con.
- ✅ **Detector:** YOLOv5su (Ultralytics, anchor-free), YOLOv8s, YOLO11s; cùng công thức huấn luyện (cùng trọng số COCO, imgsz, epoch, patience, augmentation, NMS). Mỗi detector 1 seed, ghi rõ trong bài.
- ✅ **Chia dữ liệu theo drive**, 5 tập: A (50%) / V (5%) / B (20%) / C (10%) / T (15%). Gán drive greedy theo kích thước để khớp tỉ lệ frame, kiểm tra bằng KS stat Z và chi-square class.
- ✅ **Đóng băng split (D1):** Đã gắn tag git `splits-v1` (seed=42). Split cũ (hash `fd3c...`, V = 6 drive) là **superseded** do bug mapping.
- ✅ **Phạm vi lớp (D2):** Residual, CQR và CI chỉ làm cho **Car**. Van chỉ chạy (a)–(d) với prior riêng, báo cáo mô tả. Truck chỉ báo cáo ở mức detector (P/R) và số lượng.
- ✅ **Dải khoảng cách (D3):** 5 dải thống nhất (0–10, 10–20, 20–30, 30–50, >50 m) kèm n và cờ `*` nếu n < 100, thêm một hàng gộp ">30 m" cho phân tích coverage có điều kiện.
- ✅ **Quy ước chạy T (D4):** Tuần 2 chỉ suy luận trên B, C (và A để chẩn đoán). T chỉ chạy một lần duy nhất trong script cuối tuần 3.
- ✅ **imgsz (D5, sửa lời):** Chốt 640 do giới hạn phần cứng VRAM 8 GB (RTX 5060 Laptop GPU). 960 không được so sánh bằng mAP do chạm trần VRAM ở batch 16 gây thrashing bộ nhớ chia sẻ. Dùng 640 chung cho cả 3 detector. Thử 960 (batch 8) chỉ là ablation tùy chọn trên YOLOv8s, không chặn tiến độ. Ghi Limitations về khả năng phát hiện xe ở xa.
- ✅ **Ngưỡng conf (D6, bổ sung chi tiết):** Cùng quy tắc F1 tối đa của Car Hard trên V, áp riêng cho từng detector. Quần thể V gồm 611 Car Hard trên 6 drive ($n_{\text{eff}} = 3.1$). Để tránh bẫy gai nhiễu cục bộ, đường F1 được làm trơn bằng moving average với cửa sổ 0.05 ($\pm 0.025$). Ngưỡng tối ưu thu được trên `last.pt`: `yolov8s` = 0.790 (smoothed F1 = 0.8355, raw F1 = 0.8368, raw argmax trùng tại 0.790); `yolo11s` = 0.700 (smoothed F1 = 0.8370, raw F1 = 0.8363, raw argmax trùng tại 0.700); `yolov5su` = 0.740 (smoothed F1 = 0.8240, raw F1 = 0.8227, raw argmax trùng tại 0.740). Cả 3 ngưỡng đều nằm sâu bên trong lưới quét [0.05, 0.90], không chạm biên.
- ✅ **Recipe detector (D7):** epochs=100, patience=100 (tắt early stopping) cho cả 3 detector, giữ best.pt theo fitness mặc định và ghi lại epoch được chọn. Lý do: các checkpoint pilot/cũ dừng sớm ở epoch 14–17 là đỉnh nhiễu (winner's curse trên tập V 374 ảnh), LR chưa annealing (còn ~0.0010) và close_mosaic chưa kích hoạt. Các run cũ đổi tên thành `prelim_*` (superseded), không dùng cho B/C/T.
- ✅ **Quần thể khớp (D8):** GT là Car thỏa mức lọc Hard. Khớp GT Hard trước; nếu không khớp, kiểm tra GT ngoài Hard và DontCare. Detection khớp GT ngoài Hard hoặc DontCare bị bỏ qua, không tính TP/FP và không ranging. Áp dụng cho cả chọn ngưỡng D6 và pipeline ranging.
- ✅ **Đặc trưng của mô hình chỉ dùng thông tin có lúc suy luận.** Không dùng nhãn truncated/occluded/alpha của KITTI làm đặc trưng; chỉ dùng để nhóm phân tích lỗi.
- ✅ **Calibration:** dùng P2 (và R0_rect) riêng từng ảnh; bbox của YOLO phải map về tọa độ ảnh gốc.
- ✅ **T chỉ chạy một lần**, với cấu hình đã đóng băng (git tag). Ablation chạy trên dự đoán out-of-fold của B∪C.
- ✅ **Số liệu tập V:** Chỉ dùng chọn checkpoint và ngưỡng conf; không dùng báo cáo hiệu năng detector trong paper (hiệu năng tính trên B∪C và T). Số FPS từ Ultralytics val không dùng làm latency end-to-end (sẽ đo riêng ở Tier 1).
- ✅ **Gate §8.1 & Đánh giá công bằng (D9):** Gate tính trên các dải $n \ge 100$ không chồng lấn (4 dải: 0–10, 10–20, 20–30, 30–50 m; không tính hàng gộp ">30 m"). Báo cáo thêm so sánh cặp trên tập chung (common support) và bảng phân rã theo tổ hợp cue (pattern breakdown).
- ✅ **Đóng băng hình học & Tham số hiệu dụng (D10):** $\delta = -4.6782$ px và $H_{\text{cam}} = 2.0422$ m là tham số hiệu dụng (effective ground-plane parameters) fit trên A (hash A `4402edf8...`) để bù chênh lệch giữa góc tiếp đất gần nhất ($Z_{\text{closest}}$) và tâm xe ($Z_{\text{center}}$). Trọng số hợp nhất hiện hành fit trên Split B-v2 mới (hash B `0f83c354...`, 12 drive có Car Hard, $N=4,776$): $[w_w, w_h, w_g] = [0.0807, 0.6634, 0.2560]$ (tổng = 1.0, shrinkage $\alpha = 0.0008$). Đóng băng vào `configs/geometry_params.yaml` gắn tag `geometry-v2`. *(Ghi chú: Trọng số cũ $[0.0691, 0.6500, 0.2809]$ fit trên Split B-v1 cũ hash `1242...` tag `geometry-v1` đã chính thức bị thay thế / superseded do D14)*.
- ✅ **Phân tách đặc trưng suy luận khỏi GT (D11):** Pipeline suy luận tuyệt đối tách biệt các cột đặc trưng khỏi các trường nhãn ground truth (`gt_*`, `alpha`, `truncated`, `occluded`, `depth`/`location_z`). Phải có unit test chặn và bảo đảm bộ trích đặc trưng không được đọc `gt_*` (đã có trong `tests/test_feature_guard.py`).
- ✅ **Chốt checkpoint detector bằng `last.pt` (D12):** Dùng `last.pt` (epoch 100) cho cả 3 detector, chốt trước khi xem kết quả huấn luyện lại, và ghi epoch của `best.pt` vào log để đối chứng. Lý do: Split V chỉ có 6 drive chứa Car Hard (trong đó 1 drive chiếm 41%), việc chọn `best.pt` theo đỉnh fitness trên V dễ rơi vào winner's curse / đỉnh nhiễu của vài cụm; `last.pt` tại epoch 100 bảo đảm LR annealing hoàn tất và close_mosaic đã kích hoạt đầy đủ trên toàn bộ Split A.
- ✅ **Xử lý xe suy giảm cue & pattern 000 (D13):** Toàn bộ xe thỏa Car Hard (kể cả 13.5% mang pattern `000` ở 0–10m không có cue hình học nào) đều thuộc quần thể đánh giá ranging và được dự đoán bằng mô hình direct ranging (e) làm fallback kèm cờ `fallback_flag=True`. C và T đều đi qua đúng pipeline fallback này để bảo toàn tính exchangeability, không loại bỏ ca khó khỏi bảng tổng hợp. $Z_{\text{base}} = Z_d$ nếu $\ge 1$ cue hợp lệ, ngược lại là $Z_e$. Mô hình (e) fit trên toàn bộ B. Target $r = \ln Z_{gt} - \ln Z_{\text{base}}$. Conformalize chung trên C; coverage theo cờ dùng làm chẩn đoán.
- ✅ **Phân bổ lại drive cho B/C/T, giữ nguyên A và V (D14):** Audit concentration phát hiện Split C cũ bị `drive_0059` chi phối 50.8% ($n_{\text{eff}} = 3.2$), gây rủi ro sập coverage CQR cho RQ3. Quyết định D14: phân bổ lại drive giữa B, C, T qua tìm kiếm tối đa 200 seed simulated annealing, protocol định sẵn trong docstring của script, chỉ đọc nhãn (Car Hard count và Z depth), tuyệt đối không dùng kết quả mô hình (không dùng từ "pre-registered"). Giữ nguyên 100% Split A và V. Tiêu chí: $\text{top1\_share} \le 0.35$; $\ge 10$ drive Car Hard mỗi tập; KS $\le 0.07$ cả 3 cặp; frame ratio lệch $\le 3$ điểm % so với 20/10/15. Seed 85 được chọn tối ưu: $n_{\text{eff}}$ cân bằng ở mức 5.28 cho cả 3 tập, top1 C giảm xuống 28.7%, KS B-C=0.0460, B-T=0.0588, C-T=0.0556 (tất cả đều $\le 0.0588$). Đóng băng `splits-v2` và refit trọng số hình học fit trên B mới thành `geometry-v2`. Xuất bảng chuyển dịch drive chi tiết vào `results/tables/split_migration_v1_to_v2.md`. Thêm phân tích độ nhạy (Sensitivity Analysis): fit lại trọng số hợp nhất và mô hình residual khi loại bỏ lần lượt `drive_0059` (top-2, 1.185 xe) và `drive_0104` (top-1, 1.282 xe) khỏi Split B (hai drive chiếm 51.65% Car Hard). Ghi chú Limitations: 10 drive từ B-v1 sang T-v2 và 3 drive từ B-v1 sang C-v2 có ảnh hưởng gián tiếp ở mức thiết kế định tính trong Day 4 (quy tắc mask viền).
- ✅ **Ngưỡng IoU khớp = 0.5 và Trạng thái đánh giá 4 lớp (D15):** Xây dựng module dùng chung `src/detection/matching.py` trả về 4 trạng thái rõ ràng: `TP`, `FP`, `IGNORED_NONHARD`, `IGNORED_DONTCARE`. Ưu tiên GT Hard (unmatched) trước; nếu trùng GT Hard đã ghép thì gán `FP` (duplicate, kể cả khi nằm trong vùng DontCare); kiểm tra GT non-Hard rồi DontCare để bỏ qua. Ngưỡng IoU khớp mặc định là **0.5** theo đúng kế hoạch v4 §5.1 và ngưỡng F1 của D6 (mAP@0.7 báo cáo riêng). Tiêu chí DontCare mặc định là `iou` (IoU $\ge 0.5$) tái lập 100% kết quả ngưỡng D6; hỗ trợ `area_pred` theo devkit chính thức.
- ✅ **Khớp Greedy theo điểm số giảm dần thay cho Hungarian (D16):** Chốt thuật toán matching là Greedy theo confidence giảm dần (cập nhật v4 §5.1). Lý do: trong Greedy matching, trạng thái gán của một detection chỉ phụ thuộc vào các detection có điểm số cao hơn nó; do đó khớp một lần ở sàn `conf >= 0.05` rồi lọc cờ `pass_thr` tương đương hoàn toàn về mặt toán học với việc lọc ngưỡng trước rồi mới khớp. Thuật toán Hungarian phụ thuộc vào tập ứng viên tổng thể nên không có tính chất bảo toàn phân cấp (nested filtering) này.
- ✅ **Quy tắc Out-of-Fold cho mô hình direct (e) trên Split B (D16b / Out-of-fold rule):** Khi tạo target khoảng cách $r = \ln Z_{gt} - \ln Z_{\text{base}}$ trên Split B, nếu vật thể mang pattern `000` cần fallback sang $Z_e$, thì $Z_e$ phải được tính dạng out-of-fold theo drive (Grouped CV). Mô hình $Z_e$ fit trên toàn bộ B chỉ dùng để suy luận trên C và T.
- ✅ **Quy tắc Out-of-Fold LODO cho $Z_d$ trên Split B khi tạo Target Residual (D16c):** Trọng số hợp nhất $[w_w, w_h, w_g]$ fit trên Split B có độ lệch lạc quan nhỏ in-sample (AbsRel 0.0605 so với 0.0610 LODO OOF). Nhằm bảo đảm target residual $r = \ln Z_{gt} - \ln Z_{\text{base}}$ trên tập huấn luyện B phản ánh đúng sai số thực tế khi suy luận trên C và T (out-of-sample), $Z_d$ trên Split B được tính bằng trọng số LODO OOF qua hàm `fuse_depths_lodo()` trong `src/geometry/fusion.py` (mỗi drive $d$ dùng trọng số $\mathbf{w}_{-d}$ ước lượng từ 11 drive còn lại; kiểm thử bất biến test pass 100%). Trên C và T, $Z_d$ dùng trọng số toàn cục fit trên toàn bộ B (`geometry-v2`). Phân tích độ nhạy D14 (`scripts/eval_sensitivity_d14.py`, báo cáo tại `results/tables/sensitivity_d14_report.md`) mô tả dữ liệu thực nghiệm: thứ tự $w_h$ (66.0–72.6%) $> w_g$ (22.1–25.6%) $> w_w$ (4.1–9.3%) giữ nguyên qua 4 cấu hình; $w_w$ dao động gấp 2.3 lần khi mất 51.7% mẫu; độ lệch AbsRel ngoài mẫu trên `drive_0059` là $+0.0000$ đến $+0.0010$ và trên `drive_0104` là $+0.0004$ đến $+0.0012$ (tức $\le +2.0\%$ tương đối; 12 cụm, CI thô).
- ✅ **Refit trọng số hình học trên detector bboxes (D17):** Trọng số hợp nhất $[w_w, w_h, w_g]$ trong `geometry-v2` là cận dưới lý thuyết trên GT bbox. Khi chuyển sang bbox detector ở Tuần 2, refit trọng số và ma trận hiệp phương sai $\Sigma$ trên Split B riêng cho từng detector (dùng LODO CV để đánh giá). Đồng thời giữ thêm cột đối chứng dùng trọng số fit trên GT bbox để phân rã riêng biệt giữa sai số hình học và sai số detector bbox. Báo cáo tỷ lệ ước lượng hợp lệ (`valid_frac`) và số GT bị bỏ sót (FN) riêng rẽ, không loại bỏ ca khó khỏi bảng tổng hợp.
- ✅ **Báo cáo đồng thời Macro và Pooled, hạ giọng kết luận (D18 & D20):** Báo cáo kết quả phải trình bày song song cả Pooled AbsRel và Macro AbsRel theo drive (unweighted trung bình qua 12 drive). Không sử dụng cụm từ "chứng minh không overfit" hoặc "có ý nghĩa thống kê" khi số lượng drive $\le 12$; ghi chú rõ "CI thô (12 cụm)". Centered covariance giữ nguyên vì kết quả in-sample và OOF tương đương (0.0609 vs 0.0610).
- ✅ **Sửa nhãn góc quan sát $\theta$ theo quy ước KITTI (D19):** Định nghĩa góc nhìn $\theta = \min(|\alpha|, \pi - |\alpha|)$. Theo quy ước KITTI, $|\alpha| \approx \pi/2$ ($\theta > 60^\circ$) là nhìn **Đầu/Đuôi (Front/Rear)** (N=3550, median $w \cdot Z / f_x = 2.279$ m, $Z_w$ AbsRel = 0.2611). Còn $\alpha \approx 0$ hoặc $\pi$ ($\theta < 30^\circ$) là nhìn **Ngang (Side)** (N=183, median $w \cdot Z / f_x = 4.487$ m - tương ứng chiều dài xe, $Z_w$ AbsRel = 0.4241). Ở bin nhìn ngang, $Z_h$ riêng lẻ (0.0482) thắng Fused (0.0800) vì trọng số toàn cục kéo $Z_w$ vào $\to$ Đây là động lực tự nhiên then chốt cho mô hình residual (f) với tỷ số $w/h$ làm proxy cho góc nhìn $\theta$.
- ✅ **Giả thuyết sai số âm dải 0–10m (D21):** Bias của (d) giảm đơn điệu theo khoảng cách (0–10m: $-0.128$, 10–20m: $-0.035$, 20–30m: $+0.014$, 30–50m: $+0.003$). Phép kiểm trên Split B v2 cho thấy Pattern 111 (cả 3 cue hợp lệ, không hề chạm biên) vẫn có bias $-0.1071$ (median $-0.0984$). Do đó, sai số âm ở gần không phải do hiệu ứng biên (border cut) mà chủ yếu do Ground Truth KITTI đo đến tâm hộp 3D ($Z_{\text{center}}$) trong khi các cue thị giác đo đến mặt tiếp xúc gần nhất ($Z_{\text{surface}}$), tạo độ lệch tương đối $\approx l / (2Z)$. Đây là phần residual học được và đưa vào mục Discussion.
- ✅ **Đặc tả hạ tầng suy luận detector `run_inference.py` (D22):**
  - Chạy ở chế độ FP32, `conf=0.05`, `iou_nms=0.7`, `imgsz=640`, matching IoU `--iou-match 0.5`.
  - Cờ `--split` chỉ nhận `A`, `B`, `C`; Split T bị khóa cứng bằng `PermissionError`.
  - Tự động kiểm tra hash split đối chiếu với `split_metadata.json` và kiểm tra mã băm SHA-256 của checkpoint đối chiếu với `configs/detector/checkpoints.yaml` trước khi thực thi.
  - Tách thành **3 artifact** lưu trữ:
    1. `{model}_{split}_detections.parquet`: Chỉ gồm dữ liệu test-time (tọa độ ảnh gốc, `confidence`, `class_id: 0`, cờ `pass_thr`, intrinsics P2, kích thước ảnh) - chuẩn hóa khớp tuyệt đối với `feature_extractor.py`.
    2. `{model}_{split}_matches.parquet`: Phục vụ đánh giá (trạng thái TP/FP/IGNORED, `matched_gt_idx`, `matched_iou`).
    3. `{model}_{split}_gt.parquet`: Toàn bộ GT Car Hard phục vụ tính Recall, False Negative và target residual $Z_{\text{gt}}$.
  - Tự động ghi nhật ký JSONL vào `runs/inference_log.jsonl` kèm phiên bản `ultralytics`, `torch`, `seed: 42`, `iou_match`, `splits_version`.

---

## 3. Quyết định về code

- ✅ Loader **trung lập về split**: nhận danh sách frame ID từ bên ngoài, không chứa logic split.
- ✅ Bỏ `DontCare` khi đọc nhãn xe; **thêm `parse_dontcare()`** trả về bbox DontCare riêng (cần cho §5.1 khớp Greedy D16).
- ✅ Hash split = SHA-256 của danh sách ID đã sắp xếp. Metadata ghi `seed`, tên split, `n_frames` và hash vào `split_metadata.json`.
- ✅ Kiểm tra rò rỉ theo drive bằng `assert_split_disjoint_by_drive` và script `scripts/verify_data.py`.
- ✅ Loader cung cấp cả `depth` (Z) và `distance` (Euclid); **dùng `depth` cho mọi tính toán AbsRel/MAE**.
- ✅ **Mapping fix (02/10):** `read_drive_mapping` dùng `train_rand.txt`. Frame `i` → dòng `rand[i]-1` của `train_mapping.txt`. Xác nhận bằng P2 consistency: H0=0.81, H1=1.0000.
- ✅ **3 lớp huấn luyện YOLO:** Car (0), Van (1), Truck (2). Không áp lọc Hard cho nhãn train.
- ✅ **Guard khóa tập T:** Hàm `load_split(splits_dir, 'T', allow_test=False)` trong `src/utils/split_builder.py` chặn load T nếu chưa bật `allow_test=True` hoặc `ALLOW_TEST_SPLIT=1`. Đã có unit test.
- ✅ **Log JSONL:** `runs/detector/train_log.jsonl` tự động ghi seed, split hashes, git commit, thời gian chạy và metrics.
- ✅ **Prior hình học:** `scripts/compute_priors.py` tính W_eff, H_obj, H_cam, y_horizon chỉ từ nhãn Split A, lưu vào `configs/geometry_priors.yaml`.

---

## 4. Việc cần kiểm chứng trên dữ liệu thật

- [x] Tải KITTI Object: `image_2`, `label_2`, `calib`, devkit. Đã xác nhận 7,481 frames, 141 drives.
- [x] Xác minh mapping: `train_rand.txt` kiểm chứng bằng P2 consistency = **1.0000** (141/141 drives).
- [x] Chạy KS test Car-only (Hard-filtered) trên Splits-v2 (Decision D14, Seed 85):
  - **B vs C:** KS stat = **0.0460** ($p = 7.13e-3 \le 0.0700$) → **ĐẠT!**
  - **B vs T:** KS stat = **0.0588** ($p = 3.17e-6 \le 0.0700$) → **ĐẠT!**
  - **C vs T:** KS stat = **0.0556** ($p = 1.41e-3 \le 0.0700$) → **ĐẠT!**
  *(Ghi chú: Các số KS v1 cũ: B-C=0.0527, B-T=0.0723, C-T=0.0391 đã superseded bởi splits-v2)*.
  - Độ sâu Car Hard: B (mean 25.5, median 24.1), C (mean 25.6, median 25.4), T (mean 26.2, median 26.2).
- [x] Đếm mẫu Car-only dải >50 m (Hard) trên Splits-v2: B=19*, C=0*, T=33* (tổng B∪C∪T = 52 xe; B-v1 cũ là 9*, C-v1 là 8*, T-v1 là 35*). Đánh cờ `*` khi $n < 100$ và có hàng gộp >30 m theo D3.
- [x] Phép thử P2 consistency và split disjointness chuyển thành `scripts/verify_data.py` (chạy tự động trước mỗi khâu).
- [x] Viết `tests/test_splits.py` (5 tests pass: regression mapping H0 vs H1, hash metadata, drive disjoint, frame counts, test split guard).

### 📝 Limitations cần nêu trong paper
1. **Lệch phân bố Truck:** Truck tập trung ở một số drive lớn (A=5.2%, T=0.8%), không thể khắc phục hoàn toàn khi split theo drive với 141 drive.
2. **Split theo drive ≠ theo địa điểm:** Một số drive cùng ngày có thể quay ở cùng khu vực địa lý, nhưng split theo drive là chuẩn cao nhất khả thi trên KITTI raw mapping.
3. **Mẫu dải xa (>50 m) thấp:** Car Hard >50 m chỉ có 19 xe ở B, 0 xe ở C và 33 xe ở T (tổng cộng 52 xe trên B∪C∪T). Khoảng tin cậy ở dải này rộng và đánh giá detector trên C chỉ phản ánh cự ly $\le 50$ m.
4. **Tập trung Car Hard theo ít cụm drive (Drive Concentration trên Splits-v2):** Nhiều drive trong KITTI không có xe hơi (quay trong khuôn viên/người đi bộ). Kết quả audit v2 cho thấy:
   - Split A: 35 drives, chỉ 18 drives có Car Hard ($N=11,291$, top 1 drive chiếm 16%).
   - Split B: 34 drives, 12 drives có Car Hard ($N=4,776$); top 1 là `drive_0104` chiếm 26.8% (1.282 xe), top 2 là `drive_0059` chiếm 24.8% (1.185 xe); hai drive này chiếm 51.65% Car Hard của B ($n_{\text{eff}} = 5.29$).
   - Split C: 18 drives, 10 drives có Car Hard ($N=1,826$, top 1 là `drive_0096` chiếm 28.7%, $n_{\text{eff}} = 5.28$; giảm mạnh từ 50.8% ở v1).
   - Split T: 29 drives, 10 drives có Car Hard ($N=3,212$, top 1 là `drive_0095` chiếm 23.3%, $n_{\text{eff}} = 5.28$).
   - Split V: 25 drives, chỉ 6 drives có Car Hard ($N=611$, top 1 drive chiếm **41%**!).
   *Hệ quả:* Cluster bootstrap trên T có 10 cụm độc lập (khoảng tin cậy thô, cần cảnh báo khi $<20$ cụm theo quy ước `eval.py`), grouped CV trên B có 12 fold. Đây là căn cứ khoa học vững chắc để chốt **Quyết định D12** (chọn checkpoint detector `last.pt` epoch 100 thay vì dựa vào đỉnh fitness dễ ăn may trên 6 cụm của V).
5. **Nguồn gốc Checkpoint Detector trong train_log.jsonl:** Ba dòng huấn luyện lại D7 (`ddd519a`, `058158d`) có `git_dirty: true` do untracked files lúc chạy nền. Nguồn gốc và tính toàn vẹn của checkpoint được bảo chứng độc lập và bất biến bằng mã băm SHA-256 đối chiếu khớp 100% trong `configs/detector/checkpoints.yaml` và `runs/inference_log.jsonl`.
6. **Neighbor classes trong đối sánh KITTI:** Module matching chỉ coi Car non-Hard và DontCare là đối tượng bỏ qua (ignore); Van và Truck không được coi là neighbor class như devkit chính thức của KITTI. Do đó, một số bbox dự đoán Car trùng với GT Van/Truck bị tính là FP, khiến Precision của detector trong báo cáo có thể thấp hơn thực tế một chút; điều này hoàn toàn không ảnh hưởng đến ước lượng khoảng cách Z vì ranging chỉ thực hiện trên True Positives (TP).
7. **Chuyển dịch Drive D14 (B-v1 sang T-v2 và C-v2):** Tổng cộng 10 drive từ B-v1 sang T-v2 (gồm 4 drive có Car Hard: `0926_0002`, `0926_0017`, `0926_0039`, `0926_0095` và 6 drive rỗng: `0928_0134`, `0928_0136`, `0928_0162`, `0928_0167`, `0928_0168`, `0928_0201`) cùng 3 drive từ B-v1 sang C-v2 (gồm `0926_0096` và 2 drive rỗng: `0928_0153`, `0928_0161`) từng nằm trong B-v1 khi phân tích Day 4 (hình thành quy tắc mask viền ảnh). Đây là ảnh hưởng gián tiếp ở mức thiết kế hình học định tính, không phải tối ưu hóa số học.
   *Hệ quả:* Cluster bootstrap trên T chỉ có $\le 11$ cụm độc lập (khoảng tin cậy thô, cần cảnh báo khi $<20$ cụm theo quy ước `eval.py`), grouped CV trên B tối đa 10 fold, và việc chia lại B∪C xoay quanh ~21 cụm. Đây là căn cứ khoa học vững chắc để chốt **Quyết định D12** (chọn checkpoint detector `last.pt` epoch 100 thay vì dựa vào đỉnh fitness dễ ăn may trên 6 cụm của V).

---

## 5. Tiến độ (theo §8 của kế hoạch)

### Chuẩn bị (§8.0)
- [x] Tạo project và repo GitHub `Distance-Estimation-KITTI-YOLO`
- [ ] Gửi 4 câu hỏi cho thầy
- [x] Đóng băng split và tag git `splits-v1`
- [x] Tạo script kiểm định dữ liệu `scripts/verify_data.py`
- [x] Tạo bộ test `tests/test_splits.py`
- [ ] Cài PyTorch hỗ trợ RTX 5060 (CUDA 12.8 wheel)
- [ ] Chạy thử 1 epoch fine-tune, ghi thời gian/epoch và VRAM: ______
- [x] Dựng thư mục `data/ splits/ configs/ runs/ results/ notebooks/ src/ tests/ docs/ scripts/`
- [x] Tạo log thí nghiệm JSONL (`runs/detector/train_log.jsonl`)

### Tuần 1
- [x] **Ngày 1:** loader viết xong, mapping rand sửa xong, P2 consistency = 1.0000.
- [x] **Ngày 2:** split A/V/B/C/T theo drive đã tạo, KS Car-only đạt D1, đóng băng `splits-v1`.
- [x] **Ngày 2 (tiếp):** chuyển A/V sang định dạng YOLO (`scripts/make_yolo_dataset.py` xong: Train A=3,740 frames, Val V=374 frames, `dataset.yaml`, `SPLIT_HASH.json`).
- [x] **Ngày 3 (sớm):** tính prior W_eff, H_obj, H_cam, y_horizon từ nhãn A (`scripts/compute_priors.py`, `configs/geometry_priors.yaml`).
- [x] **Ngày 3:** Chốt `imgsz = 640` (D5), chạy hoàn tất hàng đợi fine-tune YOLOv8s → YOLO11s → YOLOv5su, chốt ngưỡng conf theo F1 max Car trên V (D6).
- [x] **Ngày 4:** cài 3 cue (a)(b)(c) + hợp nhất log-space (d), kiểm thử đơn vị pass 23/23, đánh giá trên Split B GT bbox vượt điều kiện Tuần 2 (4/5 dải pass, AbsRel 0.0635).
- [x] **Ngày 5:** `eval.py` + test đơn vị (`tests/test_eval.py` 19 tests pass, `tests/test_geometry.py` 27 tests pass, `tests/test_matching.py` pass).
- [x] **Ngày 6:** chạy (a)–(d) trên GT bbox Split B-v2, phân tích chuyên sâu LODO OOF (12 fold), phân rã góc nhìn $\theta$ (chuẩn KITTI), sign count 9/12 drive ($p=0.0730$ tính bằng `scipy.stats.binomtest`), paired cluster bootstrap (`scripts/eval_day6_analysis.py`).
- [x] **Ngày 7:** đệm & hoàn tất tái phân bổ B/C/T (D14, `splits-v2`, `geometry-v2`), chốt D15 (matching 4 trạng thái), D16 (Greedy matching), D22 (`run_inference.py` FP32 `conf=0.05` parquet pipeline).

### Tuần 2
- [x] **Ngày 1:** Chạy suy luận detector trên Split B (1.499 frames) và Split C (766 frames) cho cả 3 detector (`yolov8s`, `yolo11s`, `yolov5su`). Hoàn tất xuất 18 parquet files và báo cáo hiệu năng P/R/F1/Common Support (`detector_eval_b_c.md`).
- [ ] **Ngày 2:** Chạy (a)–(d) trên bbox detector với trọng số refit trên Split B theo từng detector (D17), kèm cột đối chứng dùng trọng số fit trên GT bbox.

---

## 6. Nhật ký theo phiên

### W2-1 — 03/10/2026 (đêm): Hoàn tất Suy luận B & C cho 3 Detector (D15, D16, D22), Báo cáo Hiệu năng Detector (f156546), Track toàn bộ Parquet và Chuẩn bị D17
- **Hoàn tất toàn bộ suy luận Split B và C cho cả 3 Detector (D22):**
  - Chạy `scripts/run_inference.py` ở chế độ FP32, `imgsz=640`, matching IoU 0.5 (Greedy theo điểm số, D16), `conf_min=0.05`, dùng checkpoint `last.pt` (epoch 100, D12) khớp tuyệt đối mã băm SHA-256 trong `configs/detector/checkpoints.yaml`.
  - Cả 6 lượt suy luận chính thức đều đạt `git_dirty: false`, `splits_version: "v2"`, n_frames B=1.499 (hash `0f83c354...`), C=766 (hash `b24db22e...`):
    - Split B: `yolov8s_640` (22:19), `yolo11s_640` (22:31), `yolov5su_640` (22:33).
    - Split C: `yolov8s_640` (22:22), `yolo11s_640` (22:23), `yolov5su_640` (22:23).
  - Ghi nhận 3 dòng log **superseded** trong `runs/inference_log.jsonl`:
    1. `yolov8s_640` B (`2026-10-03T21:45:12`): chạy khi git dirty, thiếu trường `checkpoint_path`.
    2. `yolo11s_640` B (`2026-10-03T22:20:03`): chạy khi git dirty (thay thế bởi run 22:31:22).
    3. `yolov5su_640` B (`2026-10-03T22:21:09`): chạy khi git dirty (thay thế bởi run 22:33:38).
- **Lưu trữ & Track dữ liệu Parquet:**
  - Toàn bộ 18 file parquet (3 file `detections`, `matches`, `gt` $\times$ 3 model $\times$ 2 split B & C) đã được commit và track chính thức vào git repo (`commit b14a83e`). Working tree hoàn toàn sạch (clean).
- **Báo cáo Hiệu năng Detector trên Split B và C (commit `f156546`, `results/tables/detector_eval_b_c.md` & `.json`):**
  - Ngưỡng tối ưu F1 trên Split V: yolov8s (0.790), yolo11s (0.700), yolov5su (0.740).
  - **Split B (4.776 Car Hard, 12 drives):**
    - `yolov8s`: TP = 3.427, FP = 195, Ignored non-Hard = 500, Ignored DontCare = 19; P = 94.62%, R = 71.75%, F1 = 0.8161.
      - Dải: 0–10m: 284/288 (98.61%), 10–20m: 1.089/1.218 (89.41%), 20–30m: 1.204/1.546 (77.88%), 30–50m: 848/1.705 (49.74%), >50m: 2/19 (10.53%).
    - `yolo11s`: TP = 3.523, FP = 200, Ignored non-Hard = 499, Ignored DontCare = 18; P = 94.63%, R = 73.76%, F1 = 0.8290.
      - Dải: 0–10m: 284/288 (98.61%), 10–20m: 1.118/1.218 (91.79%), 20–30m: 1.238/1.546 (80.08%), 30–50m: 881/1.705 (51.67%), >50m: 2/19 (10.53%).
    - `yolov5su`: TP = 3.496, FP = 204, Ignored non-Hard = 512, Ignored DontCare = 20; P = 94.49%, R = 73.20%, F1 = 0.8249.
      - Dải: 0–10m: 285/288 (98.96%), 10–20m: 1.109/1.218 (91.05%), 20–30m: 1.232/1.546 (79.69%), 30–50m: 868/1.705 (50.91%), >50m: 2/19 (10.53%).
    - Tập chung (Common Support - cả 3 detector cùng phát hiện TP): **3.181 / 4.776 xe (66.60%)**. Hợp (ít nhất 1 detector): **3.772 xe (78.98%)**.
  - **Split C (1.826 Car Hard, 10 drives, toàn bộ $\le 50$ m):**
    - `yolov8s`: TP = 1.445, FP = 137, Ignored non-Hard = 172, Ignored DontCare = 9; P = 91.34%, R = 79.13%, F1 = 0.8480.
    - `yolo11s`: TP = 1.489, FP = 165, Ignored non-Hard = 178, Ignored DontCare = 10; P = 90.02%, R = 81.54%, F1 = 0.8557.
    - `yolov5su`: TP = 1.430, FP = 156, Ignored non-Hard = 182, Ignored DontCare = 10; P = 90.16%, R = 78.31%, F1 = 0.8382.
    - Tập chung (Common Support): **1.359 / 1.826 xe (74.42%)**.
- **Chốt nguyên tắc viết Paper cho D14:** Không sử dụng cụm từ "pre-registered" vì commit git script và split diễn ra đồng thời; trong paper trình bày khách quan: *"Protocol được định nghĩa cố định trong docstring của script từ trước, thuật toán chỉ đọc nhãn (số lượng Car Hard và Z depth) mà không nhìn kết quả mô hình"*.
- **Chuẩn bị Quyết định D17:** Sẵn sàng chạy đánh giá (a)–(d) trên bbox detector với trọng số refit trên Split B theo từng detector, đối chứng với cột trọng số fit trên GT bbox.

### Day 6 — 03/10/2026 (tối): Phân tích Chuyên sâu Hình học GT Bbox trên Split B-v2 (D18–D21, LODO OOF, Bootstrap, Góc nhìn $\theta$)
- **Mục tiêu:** Đánh giá toàn diện mô hình hình học trên Split B-v2 mới ($N=4,776$ Car Hard, 12 drive) bằng `scripts/eval_day6_analysis.py`, lưu báo cáo tại `results/tables/day6_gt_bbox_analysis.json`.
- **Số liệu chính:**
  - **In-sample vs OOF (d):**
    - Pooled AbsRel in-sample = **0.0605**; OOF Centered (LODO 12 fold) = **0.0609**; OOF Uncentered = **0.0610**; $Z_h$ đơn lẻ = **0.0688**. Độ lạc quan in-sample rất nhỏ ($\Delta = +0.0004$).
    - Trọng số trung bình qua 12 fold: Centered $[w_w, w_h, w_g] = [0.0701, 0.6680, 0.2619]$ (rất gần với in-sample $[0.0807, 0.6634, 0.2560]$).
  - **Báo cáo đồng thời Macro và Pooled (D18, D20):**
    - Macro AbsRel không trọng số qua 12 drive: (d) OOF = **0.0646** | $Z_h$ = **0.0628** | $Z_g$ = **0.1602**.
    - Trên các drive có $n \ge 30$ ($k=8$ cụm): (d) OOF = **0.0608** | $Z_h$ = **0.0651**.
    - Tuân thủ D20: Ghi nhận cảnh báo "CI thô ($k=12$ cụm)", không khẳng định "có ý nghĩa thống kê" hay "chứng minh không overfit".
  - **Sign Count qua 12 Drive (vs $Z_h$):**
    - (d) thắng trong **9/12** drive, thua 3, hòa 0.
    - $p$-value một phía: **$p = 0.0730$** (tính chính xác bằng `sign_test_one_sided(wins=9, losses=3)` qua phân phối nhị thức $299/4096$).
  - **Paired Cluster Bootstrap (vs $Z_h$ trên Common Support $n=4,676$):**
    - Ước lượng độ chênh lệch $(d - Z_h)$: **$-0.0095$** (Fused tốt hơn $Z_h$).
    - 95% CI thô (12 cụm): **$[-0.0123, -0.0062]$** (loại trừ 0, `excludes_zero: true`).
  - **Phân tầng Góc quan sát $\theta = \min(|\alpha|, \pi - |\alpha|)$ (D19):**
    - Nhìn Ngang (Side, $\theta < 30^\circ$, $N=183$): $Z_w = 0.4241$, $Z_h = 0.0482$, $Z_g = 0.2091$, (d) OOF = **0.0800**. $Z_h$ đơn lẻ thắng Fused vì $w/h$ phóng đại do chiều dài xe nhìn ngang, kéo $Z_w$ vào làm tăng sai số. Đây là động lực tự nhiên cho mô hình residual (f).
    - Nhìn Chéo (Diagonal, $30^\circ \le \theta \le 60^\circ$, $N=1,043$): $Z_w = 0.3080$, $Z_h = 0.0715$, $Z_g = 0.1125$, (d) OOF = **0.0568**.
    - Nhìn Đầu/Đuôi (Front/Rear, $\theta > 60^\circ$, $N=3,550$): $Z_w = 0.2611$, $Z_h = 0.0691$, $Z_g = 0.1554$, (d) OOF = **0.0573**.
  - **Bản chất sai số âm ở gần (D21):**
    - Pattern 111 (đủ 3 cue hợp lệ, không viền) ở 0–10m vẫn có bias âm $-0.1071$, xác nhận bias do khoảng cách $Z_{\text{center}}$ (tâm hộp 3D trong nhãn KITTI) so với $Z_{\text{surface}}$ (mặt gần nhất đo bởi cue), tạo độ lệch vật lý $\approx l / (2Z)$.

### Day 5 — 03/10/2026 (sáng): Xây dựng Khung Đánh giá Chuẩn hóa (eval.py, test_eval.py)
- **Mục tiêu:** Xây dựng module dùng chung `src/evaluation/eval.py` cho toàn bộ các khâu đánh giá từ Ngày 6 đến Tuần 3, bảo đảm tuân thủ v4 §6, D3, D11 và D18.
- **Thành phần cài đặt:**
  - Định nghĩa 5 dải khoảng cách chuẩn theo D3: `0-10`, `10-20`, `20-30`, `30-50`, `>50` m, kèm hàng gộp `>30` m và gắn cờ `low_n` ($n < 100$).
  - Phân tầng theo độ khó KITTI (`difficulty_masks`): Easy, Moderate, Hard.
  - Các hàm tính chỉ số: MAE, RMSE, AbsRel, SqRel, Delta1 ($\delta < 1.25$), RMSE_log.
  - Hàm `evaluate_report`: tổng hợp báo cáo phân nhóm theo dải hoặc theo nhãn nhóm, tự động đếm $n$, $n_{\text{valid}}$, tính `valid_frac` (không bỏ rơi các ca NaN/invalid).
  - Thuật toán cụm: `cluster_bootstrap` và `paired_cluster_bootstrap` lấy mẫu lại nguyên cụm drive độc lập với seed cố định, có cảnh báo khi số cụm $< 20$ (`MIN_CLUSTERS_WARN`).
  - Hàm ghi log tái lập: `make_log_record` và `append_jsonl`.
- **Kiểm thử đơn vị:**
  - Tạo `tests/test_eval.py` bao phủ toàn diện: tính tay các chỉ số, trường hợp hoàn hảo, phân dải, kiểm tra cờ `low_n`, cluster bootstrap và paired bootstrap, guard ghi log.
  - Toàn bộ 19 unit tests ban đầu pass 100% (và nay thêm test `sign_test_one_sided`).

### 03/10/2026 (chiều): Queue 100 epochs hoàn tất, Đánh giá last.pt (D12), Chốt conf thresholds (D6), Phân bổ B/C/T (D14) đóng băng splits-v2 & geometry-v2
- **Hoàn thành hàng đợi huấn luyện 3 detector (D7, 100 epochs, patience=100):**
  - Cả 3 detector đã hoàn thành 100 epochs trên GPU RTX 5060:
    - `yolov8s_640`: 70.56 phút (best.pt tại epoch 16, last.pt tại epoch 100).
    - `yolo11s_640`: 69.31 phút (best.pt tại epoch 15, last.pt tại epoch 100).
    - `yolov5su_640`: 65.22 phút (best.pt tại epoch 30, last.pt tại epoch 100).
- **Đánh giá trên Split V theo Quyết định D12 (chốt last.pt):**
  - Chạy `scripts/eval_detectors_val.py` trên Split V:
    - YOLOv8s (last): Car P = 0.6476, R = 0.8607, mAP50 = 0.8084, mAP50-95 = 0.5778.
    - YOLO11s (last): Car P = 0.6026, R = 0.8463, mAP50 = 0.8057, mAP50-95 = 0.5800.
    - YOLOv5su (last): Car P = 0.5353, R = 0.8511, mAP50 = 0.7727, mAP50-95 = 0.5495.
  - Toàn bộ 3 mô hình đều đạt mAP50 Car > 0.77 và Recall Car > 0.84 trên `last.pt`.
- **Chốt ngưỡng Confidence theo F1 làm trơn moving average 0.05 (D6):**
  - `yolov8s`: conf = 0.790 (smoothed F1 = 0.8355, raw F1 = 0.8368, P = 0.8807, R = 0.7971).
  - `yolo11s`: conf = 0.700 (smoothed F1 = 0.8370, raw F1 = 0.8363, P = 0.8283, R = 0.8445).
  - `yolov5su`: conf = 0.740 (smoothed F1 = 0.8240, raw F1 = 0.8227, P = 0.8254, R = 0.8200).
  - Đã lưu vào `configs/detector/conf_thresholds.yaml`.
- **Thực thi D14: Phân chia lại B/C/T, giữ nguyên 100% A và V:**
  - Chạy `scripts/repartition_bct.py` với 200 seed simulated annealing: 137/200 seed pass toàn bộ tiêu chí khắt khe.
  - Seed 85 được chọn theo đúng luật đăng ký trước (max min n_eff, lowest max KS):
    - B (1,499 frames, 4,776 Car Hard, 12 drives): top1 = 0.268, $n_{\text{eff}} = 5.29$.
    - C (766 frames, 1,826 Car Hard, 10 drives): top1 = 0.287, $n_{\text{eff}} = 5.28$ (giảm ngoạn mục từ 50.8% của drive 0059).
    - T (1,102 frames, 3,212 Car Hard, 10 drives): top1 = 0.233, $n_{\text{eff}} = 5.28$.
    - KS depth: B-C = 0.0460, B-T = 0.0588, C-T = 0.0556 (tất cả đều $\le 0.07$).
  - Ghi đóng băng `splits-v2` (`splits/split_metadata.json`, version v2).
  - Cập nhật frame counts trong `tests/test_splits.py`. Toàn bộ 84/84 test pass 100%.
- **Refit tham số hình học trên Split B mới (geometry-v2):**
  - Chạy lại `scripts/eval_day4_cues.py` trên Split B mới (1,499 frames, 4,776 Car Hard, 12 drives có xe):
    - Trọng số hợp nhất $[w_w, w_h, w_g] = [0.0807, 0.6634, 0.2560]$ (tất cả đều dương, tổng = 1.0).
    - Shrinkage alpha = 0.0008.
    - Gate §8.1: Thắng 3/4 dải độc lập $n \ge 100$ (10–20m: 0.0594 vs 0.0704, 20–30m: 0.0564 vs 0.0645, 30–50m: 0.0553 vs 0.0638; thua 0–10m: 0.1283 vs 0.0619).
    - Đóng băng vào `configs/geometry_params.yaml` với tag `geometry-v2`.
- **[SUPERSEDED (B-v1)] Xác minh 2.1 (Sửa cách tính Gate §8.1 - Quyết định D9):**
  - *(Lưu ý: Các số liệu trong phân tích này được thực hiện trên Split B-v1 cũ làm cơ sở lịch sử [AbsRel 0–10m = 0.1572, n=9 ở >50m]. Bảng trên Split B-v2 mới tương ứng là: AbsRel 0–10m = 0.1283, n=19 ở >50m; xem mục Day 6 hiện hành)*.
  - Gate chỉ tính trên các dải $n \ge 100$ độc lập: 0–10, 10–20, 20–30, 30–50 m (4 dải; không tính dải gộp `>30 m`, dải `>50 m` có $n=9 < 100$).
  - (d) thắng ở 3/4 dải và thua ở dải 0–10 m (AbsRel 0.1572 so với $Z_g$ 0.0651 và $Z_h$ 0.1463).
  - Gate ($\ge 3/4$) vẫn **ĐẠT CHUẨN**, nhưng tỷ lệ chính xác là **3/4 dải**, không phải 4/5. Đã nêu rõ dải thua 0–10 m trong báo cáo.
- **[SUPERSEDED (B-v1)] Xác minh 2.2 (Giải mã dải 0–10 m: So sánh tập chung & tổ hợp pattern):**
  - *(Lưu ý: Phân rã lịch sử trên Split B-v1: 69 xe pattern 100, 145 xe common support; đã SUPERSEDED bởi Split B-v2 và phân tích Day 6 hiện hành)*.
  - Đã chạy phân rã theo pattern: Đúng chính xác **69 xe** chênh lệch ở 0–10 m mang pattern `100` (chỉ còn $Z_w$ hợp lệ do xe gần chạm viền trên/dưới ảnh làm mask $Z_h$ và $Z_g$).
  - Ở nhóm `100`, AbsRel của (d) là **0.2284** (thoái hóa 100% về cue yếu nhất $Z_w$), kéo AbsRel chung từ 0.1238 lên 0.1572.
  - Trên tập chung (Common Support, 145 xe đủ 3 cue): (d) đạt AbsRel = **0.1238**, thắng $Z_w$ (0.1793) và $Z_h$ (0.1460).
  - Kết luận: Sai số 0–10 m là do thoái hóa cue khi bị cắt mép ảnh, không phải do trọng số hợp nhất sai. Sẽ được xử lý bằng validity flags trong mạng residual ở Tuần 2.
- **Xác minh 2.3 (Bản chất $H_{\text{cam}}$: Tham số hiệu dụng vs Chiều cao camera - Quyết định D10):**
  - Đã thực hiện phép thử nghiệm hồi quy Huber trên Split A:
    - Khi fit với khoảng cách tâm xe ($Z_{\text{center}}$): $H_{\text{cam}} = 2.0101$ m (hoặc 2.0422 m), $\delta = -4.985$ px.
    - Khi fit với khoảng cách góc tiếp đất gần nhất ($Z_{\text{closest}}$): $H_{\text{cam}} = \mathbf{1.7217}$ m, $\delta = \mathbf{0.5225}$ px $\approx 0$.
  - Kết luận: Mép đáy 2D ($y_{\text{bottom}}$) là góc tiếp đất gần nhất ($Z_{\text{closest}}$). Vì GT lấy tại tâm xe ($Z_{\text{center}} > Z_{\text{closest}}$), $H_{\text{cam}}$ bị nâng lên ~2.04 m để bù trừ. Đây là **effective ground-plane parameters**, đã quy ước chuẩn hóa trong bài báo.
- **Xác minh 2.4 (Bốn chi tiết kỹ thuật):**
  - (1) Mask biên: Đọc kích thước thực tế $(W, H)$ từng ảnh KITTI qua header PIL, dung sai $\epsilon = 2$ px.
  - (2) In-sample vs OOF: Bảng (d) là in-sample trên Split B (suy luận trên 10 drive có Car Hard của B).
  - (3) Prior: Sử dụng **median** ($W_{\text{eff}} = 2.6184$ m, $H_{\text{obj}} = 1.6797$ m).
  - (4) $n_{\text{gt}}$ trên V: Đã xác nhận $N_{\text{Car, Hard}} = 611$ trên Split V (thay cho con số 833 tổng Car chưa lọc Hard).
- **Đóng băng tham số hình học (D10 - SUPERSEDED bởi geometry-v2):** Tạo file `configs/geometry_params.yaml` bản đầu (`geometry-v1`) chứa toàn bộ tham số, trọng số $[0.0691, 0.6500, 0.2809]$ và hash của Split A (`4402...`) và Split B-v1 (`1242...`). Đóng băng tag `geometry-v1`. Bản hiện hành là `geometry-v2` fit trên Split B-v2 (hash `0f83c354...`) với trọng số $[0.0807, 0.6634, 0.2560]$ theo D14.
- **Chốt Quyết định D11:** Yêu cầu tách triệt để cột `gt_*` khỏi feature extractor và có test chặn.

### 02/10/2026 (chiều muộn): Chốt D7, D8, hoàn thành Day 4 & kiểm thử hình học
- **Cập nhật Quyết định D7 & Chạy lại hàng đợi detector:**
  - Đổi tên các checkpoint cũ thành `runs/detector/prelim_*_640` (superseded).
  - Cấu hình `patience: 100`, `epochs: 100` trong `train_config.yaml` để detector hoàn tất annealing LR và close_mosaic.
  - Chạy `scripts/train_queue.py` ở nền trên GPU RTX 5060.
- **Cập nhật Quyết định D8:**
  - Chuẩn hóa quần thể khớp: GT là Car thỏa Hard filter. Khớp GT Hard trước; detection khớp GT ngoài Hard hoặc DontCare bị bỏ qua, không tính TP/FP và không ranging.
  - Cập nhật `scripts/find_conf_thresholds.py` đọc `imgsz` từ config, ghi nhận commit git và cờ `git_dirty`.
- **Hoàn thành Ngày 4 (Hình học & Hợp nhất log-space trên GT bbox tập B):**
  - Prior Split A: $W_{eff}$ median = 2.6184 m (std rel = 35.5%), $H_{obj}$ median = 1.6797 m (std rel = 10.7%).
  - Hồi quy Huber mặt đất trên A: $\delta = -4.678$ px, $H_{cam} = 2.042$ m (giải thích chênh lệch so với 1.65m do Z lấy tại tâm xe thay vì góc gần nhất).
  - Masking biên ($\epsilon = 2$ px) trên B (4,210 Car Hard): $Z_w$ 2.26%, $Z_h$ 2.45%, $Z_g$ 2.64%.
  - Hợp nhất log-space: Ước lượng $\Sigma$ bằng grouped CV theo drive (10 drive, 4,038 mẫu), co shrinkage Ledoit-Wolf ($\alpha \approx 0.0009$). Trọng số tối ưu: $w_w = 0.0691$, $w_h = 0.6500$, $w_g = 0.2809$ (tất cả dương, tổng bằng 1.0).
  - Kết quả độ chính xác:
    - (a) $Z_w$: AbsRel = 0.2755 (cue yếu nhất, khớp giả thuyết H1)
    - (b) $Z_h$: AbsRel = 0.0719
    - (c*) $Z_g$ (fitted): AbsRel = 0.0975
    - (d*) Fused: AbsRel = **0.0635**, MAE = **1.50 m**, $\delta < 1.25$ = **98.0%**.
  - **Điều kiện sang Tuần 2 (§8.1):** ĐẠT CHUẨN! (d) thắng đơn cue tốt nhất ở 4/5 dải (10–20m, 20–30m, 30–50m, >30m).
  - Bộ kiểm thử `tests/test_geometry.py`: 23/23 tests pass 100%.

### 02/10/2026 (tối): [SUPERSEDED] Huấn luyện 3 detector (patience=15) & Ngưỡng conf thô (D6)
- **Huấn luyện thành công hàng đợi 3 detector (imgsz=640, batch=16, epochs=100, patience=15, seed=42 - SUPERSEDED):**
  - *(LƯU Ý QUAN TRỌNG: Cả checkpoint prelim do dừng sớm ở patience=15 và ngưỡng conf thô 0.430/0.600/0.650 trong phiên này đã chính thức bị thay thế / SUPERSEDED bởi phiên 03/10 chiều: Quyết định D7 huấn luyện 100 epochs tắt early stopping, D12 dùng last.pt, và D6 ngưỡng F1 trơn 0.790 / 0.700 / 0.740)*.
  - **YOLOv8s:** 22.93 phút. Car mAP50 = 0.8224, mAP@0.5:0.95 = **0.5754**, Recall = 0.8679.
  - **YOLO11s:** 22.89 phút. Car mAP50 = 0.8228, mAP@0.5:0.95 = **0.5923**, Precision = 0.7880. (Độ chính xác cao nhất).
  - **YOLOv5su:** 18.44 phút. Car mAP50 = 0.7748, mAP@0.5:0.95 = **0.5453**, Recall = 0.8463.
  - Tổng thời gian huấn luyện cả 3 detector: **64.26 phút** (rất nhanh nhờ tối ưu bộ nhớ VRAM 3.66 GB).
  - Trọng số tốt nhất đã lưu tại `runs/detector/{model}_640/weights/best.pt` (sau đó đổi tên thành `prelim_*_640`).
- **Chốt ngưỡng confidence threshold (Quyết định D6 - SUPERSEDED):**
  - Quét ngưỡng tìm $F_1$ tối đa trên lớp **Car** của tập V (bỏ qua detection khớp với DontCare theo §5.1):
    - [SUPERSEDED] - **YOLOv8s:** `conf = 0.430` $\rightarrow$ Max F1 = **0.8052** (Precision = 0.7921, Recall = 0.8187)
    - [SUPERSEDED] - **YOLO11s:** `conf = 0.600` $\rightarrow$ Max F1 = **0.7957** (Precision = 0.8469, Recall = 0.7503)
    - [SUPERSEDED] - **YOLOv5su:** `conf = 0.650` $\rightarrow$ Max F1 = **0.7798** (Precision = 0.8586, Recall = 0.7143)
  - Lưu cấu hình vào `configs/detector/conf_thresholds.yaml` (sau đó được ghi đè bằng cấu hình v2 chuẩn).

### 02/10/2026 (chiều): Chạy Pilot D5 YOLOv8s (imgsz = 640)
- **Kết quả Pilot YOLOv8s (imgsz=640, 30 epochs, batch=16, seed=42):**
  - Thời gian: 17.02 phút (24 epochs hoàn thành, Early Stopping kích hoạt tại epoch 24 vì đỉnh rơi vào epoch 9 với patience=15).
  - VRAM sử dụng: 3.66 GB / 8.15 GB (~45% VRAM, chạy mượt mà không bị thrashing).
  - **Chỉ số trên tập V (Val):**
    - **Toàn bộ (all):** P = 0.652, R = 0.507, mAP50 = 0.555, mAP@0.5:0.95 = 0.376
    - **Lớp Car (trọng tâm bài toán):**
      - **mAP@0.5:0.95 = 0.526**
      - **mAP@0.5 = 0.760**
      - **Recall = 0.826**
      - **Precision = 0.647**
    - **Van:** mAP50 = 0.354, mAP@0.5:0.95 = 0.256
    - **Truck:** mAP50 = 0.551, mAP@0.5:0.95 = 0.347
  - Tốc độ suy luận: **1.0 ms/ảnh (~1,000 FPS)** trên GPU RTX 5060!
  - Trọng số tốt nhất đã lưu tại `runs/detector/pilot_yolov8s_640/weights/best.pt`.
- **Đánh giá phần cứng imgsz = 960:** Thử nghiệm trước đó cho thấy 960 đẩy VRAM lên 8.02 GB (chạm trần 8GB VRAM của RTX 5060 Laptop GPU), kích hoạt cơ chế shared memory paging của Windows và làm chậm tốc độ huấn luyện xuống ~3 phút 50 giây/epoch (chậm hơn 6.5 lần so với 36s/epoch ở 640).

### 02/10/2026 (trưa): Chốt D1–D6, đóng băng splits-v1, YOLO dataset, priors, guard
- **Chốt D1:** Chạy KS test Car-only (Hard). C vs T có KS stat = 0.0391 ≤ 0.07 → Đạt chuẩn đóng băng split!
- Đã gắn tag git `splits-v1`. Split cũ (hash `fd3c...`, V=6 drive) chính thức superseded.
- **Chốt D2–D6:**
  - D2: Residual/CQR/CI chỉ cho Car; Van chạy mô tả; Truck chỉ ở mức detector.
  - D3: 5 dải khoảng cách thống nhất kèm cờ `*` nếu n<100, thêm hàng gộp >30 m.
  - D4: Tuần 2 không suy luận trên T; T khóa bằng guard.
  - D5: imgsz chốt bằng pilot YOLOv8s (640 vs 960, ~30 epoch).
  - D6: Ngưỡng conf F1 max Car trên V áp riêng từng detector. Đã sửa comment trong `train_config.yaml`.
- **Tạo `scripts/verify_data.py`:** Kiểm tra cấu trúc KITTI, P2 consistency = 1.0000, split hash & drive disjoint. Chạy pass 100%.
- **Tạo `tests/test_splits.py`:** 5 tests pass (kèm regression test H0 vs H1 và test guard Split T).
- **Tạo `scripts/make_yolo_dataset.py`:** Chuyển đổi thành công:
  - Train (A): 3,740 frames, 17,243 xe (Car: 14,902, Van: 1,519, Truck: 822).
  - Val (V): 374 frames, 1,051 xe (Car: 833, Van: 178, Truck: 40).
  - Tạo `data/yolo_kitti/dataset.yaml` và `SPLIT_HASH.json`.
- **Tạo `scripts/compute_priors.py`:** Tính prior hình học chỉ từ A:
  - $H_{cam}$ median = 1.886 m, $y_{horizon}$ cy = 174.8 px.
  - Car: $W_{eff}$ median = 2.618 m, $H_{obj}$ median = 1.680 m. Lưu vào `configs/geometry_priors.yaml`.
- **Cài đặt guard khóa tập T:** `load_split` chặn load T khi `allow_test=False` (bảo vệ exchangeability).
- **Tạo `src/detection/train_detector.py`:** Chuẩn hóa tham số Ultralytics, ghi commit git, split hashes và metrics vào JSONL.

### 02/10/2026 (sáng): Sửa mapping, rebuild split, KS test
- **BUG NGHIÊM TRỌNG:** `read_drive_mapping` đọc dòng `i` thay vì `rand[i]-1`. Xác nhận bằng P2 consistency test: H0=0.81 (sai), H1=1.00 (đúng). Đã sửa.
- Thêm `parse_dontcare()` vào loader.
- Sửa `split_builder.py`: greedy assignment xử lý drive lớn trước (sort by frame count desc) để tránh drives lớn dồn vào split nhỏ.
- Rebuild split với mapping đúng (seed=42):
  - A: 3,740 frames (50.0%), 35 drives
  - V: 374 frames (5.0%), 25 drives
  - B: 1,496 frames (20.0%), 28 drives
  - C: 749 frames (10.0%), 25 drives
  - T: 1,122 frames (15.0%), 28 drives
- ✓ Tỉ lệ gần hoàn hảo. Không rò rỉ drive.
- KS test depth: C vs T p=0.13 (OK). Các cặp khác p < 0.05 nhưng KS stat nhỏ (0.03–0.09), mean/median rất gần.
- Chi-square class: Truck lệch giữa các tập (A=5.2%, T=0.8%). Không sửa được hoàn toàn khi chia theo drive — ghi Limitations.
- Sửa `.gitignore` (`data/*` thay `data/`), `requirements.txt` (CUDA 12.8, ultralytics>=8.3), config detector (3 lớp Car/Van/Truck).

### 02/10/2026 (đêm qua): Loader + Split lần đầu (đã bị thay thế)
- Viết loader và split ban đầu — **mapping sai** (không dùng train_rand.txt).
- Dọn dẹp dữ liệu: xóa ảnh/calib testing, devkit/cpp, devkit/matlab.

### 01/10/2026 (tối): Dựng cấu trúc thư mục
- Tạo cấu trúc thư mục hoàn chỉnh theo §8.0: `data/`, `splits/`, `configs/`, `runs/`, `results/`, `notebooks/`, `src/`, `tests/`, `docs/`, `scripts/`.
- Phân loại 3 file có sẵn vào `docs/`: `KE_HOACH_V4.md`, `NHAT_KY_QUYET_DINH.md`, `ke_hoach_distance_estimation_v3.docx`.
- Tạo `README.md`, `.gitignore`, `requirements.txt`.
- Tạo config templates: `configs/detector/train_config.yaml`, `configs/residual/residual_config.yaml`.
- Tạo `__init__.py` cho tất cả module trong `src/` và `tests/`.
- **Bước tiếp theo:** tải KITTI + devkit, cài dependencies, chạy kiểm tra mapping trên dữ liệu thật.

### 01/10/2026: Khởi tạo project
- Tạo project "Distance Estimation – KITTI – YOLO", gắn repo GitHub vào Context.
- Phát hiện chat mới không thấy kế hoạch trong Context (file `.docx` trong repo có thể chưa đồng bộ hoặc không đọc được) → chuyển kế hoạch sang `KE_HOACH_V4.md` và tạo file nhật ký này.
- Viết `kitti_loader.py` + `test_kitti_loader.py` (6 test pass trên dữ liệu giả lập).
- **Bước tiếp theo:** tải KITTI + devkit, chạy kiểm tra mapping trên dữ liệu thật; sau đó viết script tạo split A/V/B/C/T theo drive kèm hash.

