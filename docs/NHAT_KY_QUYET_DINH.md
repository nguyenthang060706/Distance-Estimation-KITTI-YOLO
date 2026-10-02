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
- ✅ **Chia dữ liệu theo drive**, 5 tập: A (50%) / V (5%) / B (20%) / C (10%) / T (15%). Gán drive phân tầng theo Z và class.
- ✅ **Đặc trưng của mô hình chỉ dùng thông tin có lúc suy luận.** Không dùng nhãn truncated/occluded/alpha của KITTI làm đặc trưng; chỉ dùng để nhóm phân tích lỗi.
- ✅ **Calibration:** dùng P2 (và R0_rect) riêng từng ảnh; bbox của YOLO phải map về tọa độ ảnh gốc.
- ✅ **Dải khoảng cách thống nhất:** 0–10, 10–20, 20–30, 30–50, >50 m.
- ✅ **T chỉ chạy một lần**, với cấu hình đã đóng băng (git tag). Ablation chạy trên dự đoán out-of-fold của B∪C.

### Chưa chốt (cần quyết định trước khi chạy thật)

- ⏳ **imgsz** cho detector: thử 640, rồi 960/1280 nếu VRAM cho phép, chốt trên V.
- ⏳ **Quy tắc chọn ngưỡng conf:** ví dụ F1 tối đa trên V, dùng chung cho cả 3 detector.
- ⏳ **Split công bố (Chen et al.):** chỉ dùng nếu kiểm chứng bằng mapping không có drive chung; mặc định là tự chia theo drive.

---

## 3. Quyết định về code

- ✅ Loader **trung lập về split**: nhận danh sách frame ID từ bên ngoài, không chứa logic split.
- ✅ Bỏ `DontCare` khi đọc nhãn xe; **thêm `parse_dontcare()`** trả về bbox DontCare riêng (cần cho §5.1 khớp Hungarian).
- ✅ Hash split = SHA-256 của danh sách ID đã sắp xếp. Metadata ghi `seed`, tên split, `n_frames` và hash vào `split_metadata.json`.
- ✅ Kiểm tra rò rỉ theo drive bằng `assert_split_disjoint_by_drive`.
- ✅ Loader cung cấp cả `depth` (Z) và `distance` (Euclid); **dùng `depth` cho mọi tính toán AbsRel/MAE**.
- ✅ **Mapping fix (02/10):** `read_drive_mapping` giờ dùng `train_rand.txt`. Frame `i` → dòng `rand[i]-1` của `train_mapping.txt`. Xác nhận bằng P2 consistency: H0=0.81, H1=1.00.
- ✅ **3 lớp huấn luyện:** Car/Van/Truck. Car là chính, Van/Truck báo cáo riêng. Pedestrian/Cyclist là nền.
- ✅ **Nhãn huấn luyện:** lấy mọi Car/Van/Truck (không lọc Hard). Hard chỉ áp cho evaluation trên B/C/T.
- ⏳ **Guard khóa tập T:** thêm cơ chế chỉ cho load T khi cờ `frozen=True` (T chạy một lần).
- ⏳ **Log JSONL (seed, hash):** chưa có. Hiện ghi vào `split_metadata.json`.

---

## 4. Việc cần kiểm chứng trên dữ liệu thật

- [x] Tải KITTI Object: `image_2`, `label_2`, `calib`, devkit. Đã xác nhận 7,481 frames, 141 drives.
- [x] Xác minh mapping: **`train_rand.txt` phải dùng.** P2 consistency: H0 (dòng i) = 0.81, H1 (dòng rand[i]-1) = **1.00**. Đã sửa `read_drive_mapping`.
- [x] Chạy KS test Z và chi-square class giữa các tập (02/10). Kết quả:
  - KS depth: stat 0.03–0.09 (nhỏ), p < 0.05 do sample lớn, nhưng mean/median rất gần (26–28m). **C vs T: KS=0.03, p=0.13 — OK** (quan trọng nhất cho CQR).
  - Chi-square class: Truck lệch (A=5.2%, T=0.8%) — hệ quả không tránh được khi chia theo drive với Truck tập trung ở vài drive. **Ghi vào Limitations.**
- [x] ⚠️ C >50m: 77 mẫu, T >50m: 84 mẫu (< 100) → gộp dải hoặc cảnh báo khi phân tích.

---

## 5. Tiến độ (theo §8 của kế hoạch)

### Chuẩn bị (§8.0)
- [x] Tạo project Claude và repo GitHub `Distance-Estimation-KITTI-YOLO`
- [ ] Gửi 4 câu hỏi cho thầy
- [ ] Cài PyTorch hỗ trợ RTX 5060 (thường cần CUDA ≥ 12.8; kiểm tra trang chính thức PyTorch) + ultralytics, xgboost, scikit-learn, onnxruntime
- [ ] Chạy thử 1 epoch fine-tune, ghi thời gian/epoch và VRAM: ______
- [x] Dựng thư mục `data/ splits/ configs/ runs/ results/ notebooks/ src/ tests/ docs/ scripts/`
- [ ] Tạo log thí nghiệm (seed, hash split, phiên bản code, cấu hình detector)

### Tuần 1
- [x] **Ngày 1:** loader viết xong, chạy thành công trên KITTI thật (7,481 frames, 141 drives)
- [x] **Ngày 2:** split A/V/B/C/T theo drive đã tạo, đóng băng (seed=42, hash trong split_metadata.json)
- [ ] **Ngày 2 (tiếp):** chuyển A/V sang định dạng YOLO
- [ ] **Ngày 3:** khởi động hàng đợi fine-tune YOLOv8s → YOLO11s → YOLOv5su; thống kê prior W_eff, H_obj, H_cam, y_horizon từ nhãn A
- [ ] **Ngày 4:** cài 3 cue (a)(b)(c) + hợp nhất log-space (d)
- [ ] **Ngày 5:** `eval.py` + test đơn vị
- [ ] **Ngày 6:** chạy (a)–(d) trên GT bbox, vẽ sai số theo khoảng cách và theo alpha
- [ ] **Ngày 7:** đệm

---

## 6. Nhật ký theo phiên

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
- **Chưa chốt:** đang chờ kết quả KS test cuối để quyết định có cần rebuild.

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

