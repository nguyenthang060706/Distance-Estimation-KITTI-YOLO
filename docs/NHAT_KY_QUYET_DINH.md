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
- ✅ **Ngưỡng conf (D6):** Cùng quy tắc F1 tối đa của Car trên V, nhưng áp riêng cho từng detector (do thang điểm tin cậy khác nhau). Chạy lại sau khi hoàn tất huấn luyện lại D7.
- ✅ **Recipe detector (D7):** epochs=100, patience=100 (tắt early stopping) cho cả 3 detector, giữ best.pt theo fitness mặc định và ghi lại epoch được chọn. Lý do: các checkpoint pilot/cũ dừng sớm ở epoch 14–17 là đỉnh nhiễu (winner's curse trên tập V 374 ảnh), LR chưa annealing (còn ~0.0010) và close_mosaic chưa kích hoạt. Các run cũ đổi tên thành `prelim_*` (superseded), không dùng cho B/C/T.
- ✅ **Quần thể khớp (D8):** GT là Car thỏa mức lọc Hard. Khớp GT Hard trước; nếu không khớp, kiểm tra GT ngoài Hard và DontCare. Detection khớp GT ngoài Hard hoặc DontCare bị bỏ qua, không tính TP/FP và không ranging. Áp dụng cho cả chọn ngưỡng D6 và pipeline ranging.
- ✅ **Đặc trưng của mô hình chỉ dùng thông tin có lúc suy luận.** Không dùng nhãn truncated/occluded/alpha của KITTI làm đặc trưng; chỉ dùng để nhóm phân tích lỗi.
- ✅ **Calibration:** dùng P2 (và R0_rect) riêng từng ảnh; bbox của YOLO phải map về tọa độ ảnh gốc.
- ✅ **T chỉ chạy một lần**, với cấu hình đã đóng băng (git tag). Ablation chạy trên dự đoán out-of-fold của B∪C.
- ✅ **Số liệu tập V:** Chỉ dùng chọn checkpoint và ngưỡng conf; không dùng báo cáo hiệu năng detector trong paper (hiệu năng tính trên B∪C và T). Số FPS từ Ultralytics val không dùng làm latency end-to-end (sẽ đo riêng ở Tier 1).
- ✅ **Gate §8.1 & Đánh giá công bằng (D9):** Gate tính trên các dải $n \ge 100$ không chồng lấn (4 dải: 0–10, 10–20, 20–30, 30–50 m; không tính hàng gộp ">30 m"). Báo cáo thêm so sánh cặp trên tập chung (common support) và bảng phân rã theo tổ hợp cue (pattern breakdown).
- ✅ **Đóng băng hình học & Tham số hiệu dụng (D10):** $\delta = -4.6782$ px và $H_{\text{cam}} = 2.0422$ m là tham số hiệu dụng (effective ground-plane parameters) fit trên A để bù chênh lệch giữa góc tiếp đất gần nhất ($Z_{\text{closest}}$) và tâm xe ($Z_{\text{center}}$). Trọng số hợp nhất $[w_w, w_h, w_g] = [0.0691, 0.6500, 0.2809]$ fit trên B (in-sample trên 10 drive có Car Hard). Đóng băng vào `configs/geometry_params.yaml` kèm hash A và B, gắn tag `geometry-v1`.
- ✅ **Phân tách đặc trưng suy luận khỏi GT (D11):** Pipeline suy luận tuyệt đối tách biệt các cột đặc trưng khỏi các trường nhãn ground truth (`gt_*`, `alpha`, `truncated`, `occluded`, `depth`/`location_z`). Phải có unit test chặn và bảo đảm bộ trích đặc trưng không được đọc `gt_*`.

---

## 3. Quyết định về code

- ✅ Loader **trung lập về split**: nhận danh sách frame ID từ bên ngoài, không chứa logic split.
- ✅ Bỏ `DontCare` khi đọc nhãn xe; **thêm `parse_dontcare()`** trả về bbox DontCare riêng (cần cho §5.1 khớp Hungarian).
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
- [x] Chạy KS test Car-only (Hard-filtered) làm decision gate D1:
  - **C vs T:** KS stat = **0.0391** (≤ 0.07) → **ĐẠT CHUẨN ĐÓNG BĂNG D1!**
  - **B vs C:** KS stat = 0.0527 (≤ 0.07)
  - **B vs T:** KS stat = 0.0723
  - Độ sâu Car Hard: B (mean 25.5, median 24.1), C (mean 25.6, median 25.4), T (mean 26.2, median 26.2).
- [x] Đếm mẫu Car-only dải >50 m (Hard): B=9*, C=8*, T=35* (đều < 100 → đánh cờ `*` và thêm hàng gộp >30 m theo D3).
- [x] Phép thử P2 consistency và split disjointness chuyển thành `scripts/verify_data.py` (chạy tự động trước mỗi khâu).
- [x] Viết `tests/test_splits.py` (5 tests pass: regression mapping H0 vs H1, hash metadata, drive disjoint, frame counts, test split guard).

### 📝 Limitations cần nêu trong paper
1. **Lệch phân bố Truck:** Truck tập trung ở một số drive lớn (A=5.2%, T=0.8%), không thể khắc phục hoàn toàn khi split theo drive với 141 drive.
2. **Split theo drive ≠ theo địa điểm:** Một số drive cùng ngày có thể quay ở cùng khu vực địa lý, nhưng split theo drive là chuẩn cao nhất khả thi trên KITTI raw mapping.
3. **Mẫu dải xa (>50 m) thấp:** Car Hard >50 m chỉ có 8 xe ở C và 35 xe ở T; khoảng tin cậy ở dải này rộng và cần phân tích thận trọng.

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
- [x] **Ngày 5:** `eval.py` + test đơn vị (`tests/test_geometry.py` 23 tests pass, `src/evaluation/metrics.py`).
- [ ] **Ngày 6:** chạy (a)–(d) trên GT bbox, vẽ sai số theo khoảng cách và theo alpha.
- [ ] **Ngày 7:** đệm.

---

## 6. Nhật ký theo phiên

### 03/10/2026 (sáng): Xác minh Day 4 (Mục 2.1–2.4), Chốt D9–D11, Đóng băng geometry-v1
- **Xác minh 2.1 (Sửa cách tính Gate §8.1 - Quyết định D9):**
  - Gate chỉ tính trên các dải $n \ge 100$ độc lập: 0–10, 10–20, 20–30, 30–50 m (4 dải; không tính dải gộp `>30 m`, dải `>50 m` có $n=9 < 100$).
  - (d) thắng ở 3/4 dải và thua ở dải 0–10 m (AbsRel 0.1572 so với $Z_g$ 0.0651 và $Z_h$ 0.1463).
  - Gate ($\ge 3/4$) vẫn **ĐẠT CHUẨN**, nhưng tỷ lệ chính xác là **3/4 dải**, không phải 4/5. Đã nêu rõ dải thua 0–10 m trong báo cáo.
- **Xác minh 2.2 (Giải mã dải 0–10 m: So sánh tập chung & tổ hợp pattern):**
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
- **Đóng băng tham số hình học (D10):** Tạo file `configs/geometry_params.yaml` chứa toàn bộ tham số, trọng số và hash của Split A (`4402...`) và Split B (`1242...`). Đóng băng tag `geometry-v1`.
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

### 02/10/2026 (tối): Hoàn thành hàng đợi 3 detector & Chốt ngưỡng conf (D6)
- **Huấn luyện thành công hàng đợi 3 detector (imgsz=640, batch=16, epochs=100, patience=15, seed=42):**
  - **YOLOv8s:** 22.93 phút. Car mAP50 = 0.8224, mAP@0.5:0.95 = **0.5754**, Recall = 0.8679.
  - **YOLO11s:** 22.89 phút. Car mAP50 = 0.8228, mAP@0.5:0.95 = **0.5923**, Precision = 0.7880. (Độ chính xác cao nhất).
  - **YOLOv5su:** 18.44 phút. Car mAP50 = 0.7748, mAP@0.5:0.95 = **0.5453**, Recall = 0.8463.
  - Tổng thời gian huấn luyện cả 3 detector: **64.26 phút** (rất nhanh nhờ tối ưu bộ nhớ VRAM 3.66 GB).
  - Trọng số tốt nhất đã lưu tại `runs/detector/{model}_640/weights/best.pt`.
- **Chốt ngưỡng confidence threshold (Quyết định D6):**
  - Quét ngưỡng tìm $F_1$ tối đa trên lớp **Car** của tập V (bỏ qua detection khớp với DontCare theo §5.1):
    - **YOLOv8s:** `conf = 0.430` $\rightarrow$ Max F1 = **0.8052** (Precision = 0.7921, Recall = 0.8187)
    - **YOLO11s:** `conf = 0.600` $\rightarrow$ Max F1 = **0.7957** (Precision = 0.8469, Recall = 0.7503)
    - **YOLOv5su:** `conf = 0.650` $\rightarrow$ Max F1 = **0.7798** (Precision = 0.8586, Recall = 0.7143)
  - Lưu cấu hình vào `configs/detector/conf_thresholds.yaml`.

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

