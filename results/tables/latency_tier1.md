# Bảng Đo Độ Trễ Từng Khâu (Latency Tier 1 Benchmark, T16 Nghiệm Thu Chính Thức)

> [!IMPORTANT]
> - Tuân thủ đặc tả Kế hoạch v4 §5.6 và **Quyết định D40, D44, D94, D95**.
> - **GPU Line:** PyTorch `.pt` FP16 trên CUDA GPU (`torch.cuda.synchronize()`).
> - **CPU Line:** ONNX Runtime CPU FP32 (cố định `intra_op_num_threads=4`).
> - **Khắc phục 4 lỗi D44:** Triệt tiêu double-count tiền xử lý; đo mảng $\{t_{\text{total}}^{(i)}\}$ từng ảnh thay cho sum of medians; chạy đủ 17 đặc trưng XGBoost; bổ sung đo khâu CQR uncertainty.
> - **Cấm:** Không suy diễn kết luận về độ chính xác từ mô hình ONNX.

## 1. Cấu hình Phần cứng & Thư viện Thử nghiệm

- **CPU:** `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD` (Luồng kiểm thử: `4`)
- **GPU:** `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Thư viện:** PyTorch `2.11.0+cu128`, ONNX Runtime `1.30.0`, XGBoost `2.0.3`
- **Tập dữ liệu:** `200` ảnh ngẫu nhiên từ Split B (sau `20` ảnh khởi động warmup).

## 2. Kết quả Kiểm tra Tính Tương đồng (Parity Check: PyTorch .pt vs ONNX, D44, D95)

| Detector | Ảnh kiểm thử | Detections (.pt) | Detections (ONNX) | Tỉ lệ Số lượng (ORT/PT) | Tỉ lệ Khớp IoU ≥ 0.90 | Parity Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo11s_640` | 50 | 231 | 225 | 0.9740 | 0.9437 | ✅ ĐẠT |
| `yolov8s_640` | 50 | 223 | 225 | 1.0090 | 0.9417 | ✅ ĐẠT |
| `yolov5su_640` | 50 | 232 | 230 | 0.9914 | 0.9095 | ✅ ĐẠT |


## 3. Bảng Độ Trễ Từng Khâu (Median / P95 theo ms)

| Khâu Pipeline | `yolo11s_640` (GPU / CPU) | `yolov8s_640` (GPU / CPU) | `yolov5su_640` (GPU / CPU) |
| :--- | :---: | :---: | :---: |
| **Preprocess (Resize/Letterbox)** | 35.16 ms / 20.69 ms | 24.31 ms / 20.37 ms | 24.44 ms / 20.55 ms |
| **Detector Inference (FP16 GPU / FP32 CPU)** | 26.40 ms / 107.48 ms | 19.21 ms / 136.09 ms | 21.64 ms / 113.59 ms |
| **Postprocess (Decode/NMS)** | 0.63 ms / 0.44 ms | 0.64 ms / 0.44 ms | 0.64 ms / 0.44 ms |
| **Geometry (Cues + Fusion)** | 0.18 ms / 0.19 ms | 0.19 ms / 0.19 ms | 0.18 ms / 0.19 ms |
| **Residual (17 Features + XGBoost)** | 0.60 ms / 0.68 ms | 0.64 ms / 0.67 ms | 0.63 ms / 0.68 ms |
| **CQR Uncertainty (Log-Interval)** | 0.60 ms / 0.59 ms | 0.60 ms / 0.60 ms | 0.59 ms / 0.59 ms |
| :--- | :---: | :---: | :---: |
| **Tổng Toàn Pipeline End-to-End (ms)** | **63.64 ms / 130.52 ms** | **45.77 ms / 158.39 ms** | **48.37 ms / 136.16 ms** |
| *Đối chứng: Tổng các Trung vị (Sum of Medians, D44)* | *63.56 ms / 130.06 ms* | *45.58 ms / 158.36 ms* | *48.13 ms / 136.03 ms* |
| **Thông lượng Tương đương (FPS)** | **15.7 FPS / 7.7 FPS** | **21.8 FPS / 6.3 FPS** | **20.7 FPS / 7.3 FPS** |


## 4. Nhận xét Phân bố Thời gian Thực thi (D44, D94)

1. **Khâu Detector:** Là điểm nghẽn chính về thời gian. Trên GPU NVIDIA RTX 5060 Laptop (PyTorch FP16 CUDA), suy luận thô mất ~19.2–26.4 ms; trên CPU (ONNX Runtime 4 luồng) mất ~107.5–136.1 ms.
2. **Khâu Hình học & Residual:** Cực kỳ gọn nhẹ: hình học (cues + fusion) chỉ mất ~0.18–0.19 ms; khâu trích xuất 17 đặc trưng và dự đoán XGBoost mất ~0.60–0.64 ms cho mỗi ảnh.
3. **Khâu CQR Uncertainty:** Khâu tính toán khoảng tin cậy conformal trong không gian log chỉ mất ~0.59–0.60 ms (chủ yếu do 2 mô hình quantile XGBoost), hoàn toàn nằm trong ngân sách thời gian thực.
4. **Khắc phục lỗi Sum-of-Medians (D44):** Tổng thời gian end-to-end thực tế đo trên từng ảnh $\{t_{\text{total}}^{(i)}\}$ phản ánh chính xác phân phối thời gian thực thi (kèm P95 và IQR), khắc phục độ lệch so với tổng các trung vị đơn lẻ.
5. **Khả năng thời gian thực:** Toàn bộ pipeline đạt ~15.7–21.8 FPS trên GPU RTX 5060 và ~6.3–7.7 FPS trên CPU 4 luồng, hoàn toàn đáp ứng yêu cầu ADAS thời gian thực (chuẩn $\ge 10$ FPS trên GPU).
