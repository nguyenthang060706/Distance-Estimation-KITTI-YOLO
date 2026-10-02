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

## 3. Quyết định về code (phiên "Writing KITTI data loader")

- ✅ Loader **trung lập về split**: nhận danh sách frame ID từ bên ngoài, không chứa logic split.
- ✅ Bỏ `DontCare` và các lớp không phải xe khi đọc nhãn.
- ✅ Hash split = SHA-256 của danh sách ID đã sắp xếp. Mỗi lần chạy ghi `seed`, tên split, `n_frames` và hash vào file JSONL.
- ✅ Kiểm tra rò rỉ theo drive bằng `assert_split_disjoint_by_drive` (báo lỗi nếu một drive nằm ở hai tập). Gọi ngay sau khi tạo split.
- ✅ Loader cung cấp cả `depth` (Z) và `distance` (Euclid); **dùng `depth` cho mọi tính toán AbsRel/MAE**.
- ⚠️ **Cần chỉnh cho khớp kế hoạch:** loader đặt mặc định `VEHICLE_CLASSES = Car/Van/Truck`. Theo v4, Car là chính và Van/Truck phải báo cáo riêng. Giữ nguyên danh sách nhưng đảm bảo mọi bảng kết quả tách theo class.
- ⏳ **Guard khóa tập T:** thêm cơ chế chỉ cho load T khi cờ `frozen=True` (T chạy một lần). Đang chờ thêm vào code.

---

## 4. Việc cần kiểm chứng trên dữ liệu thật ⚠️

Loader hiện chỉ được kiểm tra bằng 6 test đơn vị trên **dữ liệu giả lập**, chưa chạy trên KITTI thật.

- [ ] Tải KITTI Object: `image_2`, `label_2`, `calib`, devkit (`devkit_object` có `train_mapping.txt`, `train_rand.txt`).
- [ ] Chạy `read_drive_mapping`, kiểm tra: đủ **7.481 frame**; số drive hợp lý; các frame cùng drive có chỉ số raw liên tiếp.
- [ ] Xác minh cách ghép `train_rand` với `train_mapping` (code hiện viết theo trí nhớ, cần đối chiếu dữ liệu thật).
- [ ] Ghi lại **số drive thực tế** và số xe theo dải khoảng cách, để biết có đủ mẫu chia 5 tập. Nếu một dải có dưới ~100 xe ở C hoặc T thì gộp dải hoặc báo cáo kèm cảnh báo.

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

### 02/10/2026: Loader + Split trên dữ liệu thật
- Viết `src/utils/kitti_loader.py`: loader KITTI đọc image/label/calib/drive mapping, chạy thành công trên 7,481 frames.
- Viết `src/utils/split_builder.py`: tạo split theo drive, phân tầng theo median depth.
- Viết `scripts/create_splits.py`: script tạo + validate splits.
- **Kết quả split (seed=42):**
  - A: 3,679 frames (49.2%), 83 drives — detector train
  - V: 287 frames (3.8%), 6 drives — detector val
  - B: 1,767 frames (23.6%), 20 drives — residual train
  - C: 707 frames (9.5%), 6 drives — CQR calibration
  - T: 1,041 frames (13.9%), 26 drives — final test
- ✓ Không rò rỉ drive giữa các tập.
- ⚠️ V nhỏ hơn target (3.8% vs 5%) do chia theo drive → 287 frames vẫn đủ cho early stopping.
- ⚠️ C >50m: 87 samples (< 100) → cần lưu ý khi phân tích CQR ở dải xa.
- Dọn dẹp dữ liệu: xóa ảnh/calib testing (không có nhãn), xóa devkit/cpp và devkit/matlab.
- **Bước tiếp theo:** chuyển A/V sang định dạng YOLO; thống kê prior; bắt đầu fine-tune.

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

