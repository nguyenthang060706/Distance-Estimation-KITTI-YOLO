# Bảng Đo Độ Trễ Từng Khâu (Latency Tier 1 Benchmark, T06)

> [!IMPORTANT]
> - Tuân thủ đúng đặc tả v4 §5.6 và **Quyết định D40**.
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

| Detector | Ảnh kiểm thử | Detections (.pt) | Detections (ONNX) | Tỉ lệ Số lượng (ORT/PT) | Tỉ lệ Khớp IoU ≥ 0.90 | Parity Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo11s_640` | 50 | 231 | 225 | 0.9740 | 0.9437 | ✅ ĐẠT |
| `yolov8s_640` | 50 | 223 | 225 | 1.0090 | 0.9417 | ✅ ĐẠT |
| `yolov5su_640` | 50 | 230 | 230 | 1.0000 | 0.9043 | ✅ ĐẠT |


## 3. Bảng Độ Trễ Từng Khâu (Median / P95 theo ms)

| Khâu Pipeline | `yolo11s_640` (GPU / CPU) | `yolov8s_640` (GPU / CPU) | `yolov5su_640` (GPU / CPU) |
| :--- | :---: | :---: | :---: |
| **Preprocess (Resize/Letterbox)** | 32.14 ms / 32.14 ms | 20.55 ms / 20.55 ms | 20.55 ms / 20.55 ms |
| **Detector Inference (FP16 GPU / FP32 CPU)** | 36.76 ms / 106.69 ms | 32.57 ms / 131.25 ms | 34.0 ms / 109.69 ms |
| **Postprocess (Decode/NMS)** | 0.33 ms / 0.47 ms | 0.32 ms / 0.48 ms | 0.32 ms / 0.5 ms |
| **Geometry (Cues + Fusion)** | 0.18 ms / 0.18 ms | 0.18 ms / 0.18 ms | 0.19 ms / 0.19 ms |
| **Residual (Feature + XGBoost)** | 0.7 ms / 0.7 ms | 0.69 ms / 0.69 ms | 0.7 ms / 0.7 ms |
| **CQR Uncertainty** | — / — | — / — | — / — |
| :--- | :---: | :---: | :---: |
| **Tổng Toàn Pipeline (ms)** | **70.11 ms / 140.18 ms** | **54.31 ms / 153.15 ms** | **55.76 ms / 131.63 ms** |
| **Thông lượng Tương đương (FPS)** | **14.3 FPS / 7.1 FPS** | **18.4 FPS / 6.5 FPS** | **17.9 FPS / 7.6 FPS** |


## 4. Nhận xét Phân bố Thời gian Thực thi

1. **Khâu Detector:** Là khâu chiếm tỉ trọng lớn nhất trong pipeline. Trên GPU NVIDIA RTX 5060 Laptop (PyTorch FP16), thời gian suy luận dao động khoảng ~32–37 ms, trong khi trên CPU (ONNX Runtime 4 luồng) mất khoảng ~106–131 ms.
2. **Khâu Hình học & Residual:** Cực kỳ gọn nhẹ: khâu tính toán hình học (cues + fusion) chỉ mất ~0.18–0.19 ms, và khâu residual (XGBoost) chỉ mất ~0.70 ms cho mỗi ảnh.
3. **Khả năng thời gian thực (Real-time Capability):**
   - Trên GPU, toàn bộ pipeline từ ảnh thô tới ước lượng khoảng cách đạt ~54–70 ms (~14.3–18.4 FPS).
   - Trên CPU, pipeline đạt ~131–153 ms (~6.5–7.6 FPS).
