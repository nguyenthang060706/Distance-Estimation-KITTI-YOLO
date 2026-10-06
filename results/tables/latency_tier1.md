# Bảng Đo Độ Trễ Từng Khâu (Latency Tier 1 Benchmark, T06)

> [!WARNING]
> **PRELIMINARY — KẾT QUẢ SƠ BỘ: KHÔNG TRÍCH DẪN SỐ LIỆU NÀY CHO BÀI BÁO (Quyết định D44)**
> 
> Bảng đo này được thực hiện trong phiên W2-4 để kiểm thử giao diện và độ khả thi của Tier 1, nhưng tồn tại các hạn chế phương pháp luận:
> 1. **Double count ở GPU line:** Hàm `model_pt.predict(source=p)` của Ultralytics đã thực hiện đọc ảnh, letterbox, suy luận và NMS nội bộ. Việc cộng thêm khâu "Preprocess" (đo imread + letterbox riêng) là đếm hai lần thời gian tiền xử lý.
> 2. **Tổng là tổng các trung vị:** Thời gian tổng toàn pipeline hiện được tính bằng tổng các trung vị thành phần ($\sum \text{median}$) và P95 bằng tổng các P95 ($\sum \text{P95}$), thay vì tính trung vị của tổng thời gian thực tế trên từng ảnh ($\text{median}(\sum)$).
> 3. **Đường tắt ở khâu Residual:** Khâu đo residual dùng confidence cố định (0.8) và tính `cx_offset` chia cho chiều rộng ảnh thay vì tiêu cự $f_x$ như trong `feature_extractor.py`.
> 4. **Khác biệt input padding PT vs ORT:** Ultralytics PyTorch ở batch 1 thường dùng dynamic rectangular padding (~640×224), trong khi ONNX export tĩnh là 640×640.
> 5. **Toàn bộ phép đo độ trễ sẽ được viết lại chuẩn hóa và đo chính thức một lần duy nhất cùng khâu CQR ở tác vụ T16.**

> [!IMPORTANT]
> - Tuân thủ đặc tả v4 §5.6 và **Quyết định D40, D44**.
> - **GPU Line (Bắt buộc):** PyTorch `.pt` FP16 trên CUDA GPU (`torch.cuda.synchronize()`).
> - **CPU Line (Bắt buộc):** ONNX Runtime CPU FP32 (cố định số luồng CPU `intra_op_num_threads=4`).
> - **Khâu CQR:** Để trống `—`, sẽ được hoàn thiện ở tác vụ T16.
> - **Cấm:** Không suy diễn kết luận về độ chính xác từ mô hình ONNX.

## 1. Cấu hình Phần cứng & Thư viện Thử nghiệm

- **CPU:** `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD` (Luồng kiểm thử: `4`)
- **GPU:** `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Thư viện:** PyTorch `2.11.0+cu128`, ONNX Runtime `1.30.0`, XGBoost `2.0.3`
- **Tập dữ liệu:** `200` ảnh ngẫu nhiên từ Split B (sau `20` ảnh khởi động warmup).

## 2. Kết quả Kiểm tra Tính Tương đồng (Parity Check: PyTorch .pt vs ONNX)

| Detector | Ảnh kiểm thử | Detections (.pt) | Detections (ONNX) | Tỉ lệ Số lượng (ORT/PT) | Count Parity (0.95–1.05) | Tỉ lệ Khớp IoU ≥ 0.90 | IoU Parity (≥ 0.95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo11s_640` | 50 | 231 | 225 | 0.9740 | ✅ ĐẠT | 0.9437 | ❌ Chưa đạt (0.944 < 0.95) |
| `yolov8s_640` | 50 | 223 | 225 | 1.0090 | ✅ ĐẠT | 0.9417 | ❌ Chưa đạt (0.942 < 0.95) |
| `yolov5su_640` | 50 | 230 | 230 | 1.0000 | ✅ ĐẠT | 0.9043 | ❌ Chưa đạt (0.904 < 0.95) |

*(Ghi chú: Báo cáo trung thực cả hai chỉ số parity theo AGENT_RULES §1.9, không đổi ngưỡng để ép đạt. Nguyên nhân tỉ lệ IoU chưa đạt 0.95 có thể do khác biệt giữa FP16 PyTorch và FP32 ONNX, hoặc khác biệt dynamic rect padding vs static square padding. Sẽ được đối chứng ở T16).*

## 3. Bảng Độ Trễ Từng Khâu (Median / P95 theo ms - Sơ bộ)

| Khâu Pipeline | `yolo11s_640` (GPU / CPU) | `yolov8s_640` (GPU / CPU) | `yolov5su_640` (GPU / CPU) |
| :--- | :---: | :---: | :---: |
| **Preprocess (Resize/Letterbox)** | 32.14 ms / 32.14 ms | 20.55 ms / 20.55 ms | 20.55 ms / 20.55 ms |
| **Detector Inference (FP16 GPU / FP32 CPU)** | 36.76 ms / 106.69 ms | 32.57 ms / 131.25 ms | 34.0 ms / 109.69 ms |
| **Postprocess (Decode/NMS)** | 0.33 ms / 0.47 ms | 0.32 ms / 0.48 ms | 0.32 ms / 0.5 ms |
| **Geometry (Cues + Fusion)** | 0.18 ms / 0.18 ms | 0.18 ms / 0.18 ms | 0.19 ms / 0.19 ms |
| **Residual (Feature + XGBoost)** | 0.7 ms / 0.7 ms | 0.69 ms / 0.69 ms | 0.7 ms / 0.7 ms |
| **CQR Uncertainty** | — / — | — / — | — / — |
| :--- | :---: | :---: | :---: |
| **Tổng Toàn Pipeline (ms) [$\sum \text{median}$]** | **70.11 ms / 140.18 ms** | **54.31 ms / 153.15 ms** | **55.76 ms / 131.63 ms** |
| **Thông lượng Ước tính (FPS)** | **14.3 FPS / 7.1 FPS** | **18.4 FPS / 6.5 FPS** | **17.9 FPS / 7.6 FPS** |

## 4. Nhận xét Phân bố Thời gian Thực thi (Sơ bộ)

1. **Khâu Detector:** Là khâu chiếm phần lớn thời gian trong pipeline (~32–37 ms trên GPU FP16, ~106–131 ms trên CPU ONNX 4 luồng).
2. **Khâu Hình học & Residual:** Cực kỳ gọn nhẹ: hình học chỉ mất ~0.18–0.19 ms, residual XGBoost mất ~0.70 ms mỗi ảnh.
3. **Kế hoạch T16:** Do các hạn chế phương pháp luận ở mục cảnh báo, kết quả trên chỉ mang tính sơ bộ. Khâu đo latency chính thức sẽ được chạy lại ở T16 với pipeline end-to-end hoàn chỉnh (kèm CQR), loại bỏ double count, đo median per-image tổng thể, và kiểm tra input shape chuẩn xác.
