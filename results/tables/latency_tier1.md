# Bảng Đo Độ Trễ Từng Khâu (Latency Tier 1 Benchmark, PRELIMINARY-v2, D98)

> [!IMPORTANT]
> - Tuân thủ đặc tả Kế hoạch v4 §5.6 và **Quyết định D40, D44, D94, D95, D98, D99**.
> - **Nhãn trạng thái (D98):** `PRELIMINARY-v2` (kết quả một lượt đo trên laptop cá nhân; không dùng để xếp hạng throughput giữa các detector).
> - **GPU Line:** PyTorch `.pt` FP16 trên CUDA GPU (`torch.cuda.synchronize()`).
> - **CPU Line:** ONNX Runtime CPU FP32 (cố định `intra_op_num_threads=4`).
> - **Khắc phục 4 lỗi D44:** Loại bỏ Disk I/O khỏi vòng lặp đo (In-memory benchmark); đo mảng $\{t_{\text{total}}^{(i)}\}$ từng ảnh thay cho sum of medians; chạy đủ 17 đặc trưng XGBoost; bổ sung đo khâu CQR uncertainty.
> - **Cấm:** Không suy diễn kết luận về độ chính xác từ mô hình ONNX.

## 1. Cấu hình Phần cứng & Thư viện Thử nghiệm

- **CPU:** `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD` (Luồng kiểm thử: `4`)
- **GPU:** `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Thư viện:** PyTorch `2.11.0+cu128`, ONNX Runtime `1.30.0`, XGBoost `2.0.3`
- **Tập dữ liệu:** `200` ảnh ngẫu nhiên từ Split B (sau `20` ảnh khởi động warmup).

## 2. Kết quả Kiểm tra Tính Tương đồng (Parity Check: PyTorch .pt vs ONNX, D44, D95, D99)

| Detector | Ảnh kiểm thử | Detections (.pt) | Detections (ONNX) | Tỉ lệ Số lượng (ORT/PT) | Tỉ lệ Khớp IoU ≥ 0.90 | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo11s_640` | 50 | 231 | 225 | 0.9740 | 0.9437 | Count ĐẠT, IoU KHÔNG ĐẠT |
| `yolov8s_640` | 50 | 223 | 225 | 1.0090 | 0.9417 | Count ĐẠT, IoU KHÔNG ĐẠT |
| `yolov5su_640` | 50 | 232 | 230 | 0.9914 | 0.9095 | Count ĐẠT, IoU KHÔNG ĐẠT |

> [!NOTE]
> **Giải trình Parity (D95, D99):** Count Parity đạt chuẩn [0.95, 1.05]. IoU Match Rate đạt 0.9095–0.9437 (< 0.95, KHÔNG ĐẠT theo ngưỡng đăng ký). Các nguyên nhân có thể gồm: khác biệt dynamic letterbox PyTorch vs fixed square 640 ONNX, model.half() FP16 vs ORT FP32, và sự khác biệt giữa NMS numpy vs NMS Ultralytics torch (đây là các giả thuyết kỹ thuật chưa kiểm chứng độc lập). Tuyệt đối không làm mềm kết quả không đạt.

## 3. Bảng Độ Trễ Từng Khâu (Median / P95 theo ms)

| Khâu Pipeline | `yolo11s_640` (GPU / CPU) | `yolov8s_640` (GPU / CPU) | `yolov5su_640` (GPU / CPU) |
| :--- | :---: | :---: | :---: |
| **Preprocess (In-memory Letterbox/H2D)** | 9.59 ms / 6.12 ms | 9.88 ms / 6.17 ms | 9.80 ms / 6.13 ms |
| **Detector Inference (FP16 GPU / FP32 CPU)** | 26.11 ms / 108.47 ms | 19.17 ms / 134.28 ms | 21.08 ms / 111.56 ms |
| **Postprocess (Decode/NMS)** | 0.61 ms / 0.45 ms | 0.64 ms / 0.44 ms | 0.60 ms / 0.43 ms |
| **Geometry (Cues + Fusion)** | 0.18 ms / 0.19 ms | 0.18 ms / 0.19 ms | 0.18 ms / 0.19 ms |
| **Residual (17 Features + XGBoost)** | 0.60 ms / 0.68 ms | 0.59 ms / 0.67 ms | 0.57 ms / 0.67 ms |
| **CQR Uncertainty (Log-Interval)** | 0.59 ms / 0.58 ms | 0.60 ms / 0.59 ms | 0.58 ms / 0.58 ms |
| :--- | :---: | :---: | :---: |
| **Tổng Toàn Pipeline End-to-End (ms)** | **38.01 ms / 116.73 ms** | **31.70 ms / 142.54 ms** | **33.12 ms / 119.87 ms** |
| *Đối chứng: Tổng các Trung vị (Sum of Medians, D44)* | *37.69 ms / 116.50 ms* | *31.07 ms / 142.33 ms* | *32.81 ms / 119.57 ms* |
| **Thông lượng Tương đương (FPS)** | **26.3 FPS / 8.6 FPS** | **31.5 FPS / 7.0 FPS** | **30.2 FPS / 8.3 FPS** |

## 4. Nhận xét Phân bố Thời gian Thực thi (D44, D94, D98)

1. **Khâu Detector:** Là điểm nghẽn chính về thời gian. Trên GPU NVIDIA RTX 5060 Laptop (PyTorch FP16 CUDA), suy luận thô mất ~19.2–26.1 ms; trên CPU (ONNX Runtime 4 luồng) mất ~108.5–134.3 ms.
2. **Khâu Hình học & Residual:** Cực kỳ gọn nhẹ: hình học (cues + fusion) chỉ mất ~0.18–0.18 ms; khâu trích xuất 17 đặc trưng và dự đoán XGBoost mất ~0.57–0.60 ms cho mỗi ảnh.
3. **Khâu CQR Uncertainty:** Khâu tính toán khoảng tin cậy conformal trong không gian log chỉ mất ~0.58–0.60 ms (chủ yếu do 2 mô hình quantile XGBoost), hoàn toàn nằm trong ngân sách thời gian thực.
4. **So sánh Sum-of-Medians vs End-to-End per-image (D44, D94):** Chênh lệch giữa tổng các trung vị đơn lẻ và trung vị chuỗi tổng $\{t_{\text{total}}^{(i)}\}$ trên dữ liệu thực tế là 0.20–0.63 ms (≤ 2.0%). Việc đo trực tiếp thời gian end-to-end trên từng ảnh là chuẩn mực phương pháp luận thống kê nhằm phản ánh đúng phân phối tổng thể và theo dõi chính xác các phân vị đuôi (P95, IQR).
5. **Thông lượng hệ thống (D98):** Toàn bộ pipeline đạt ~26.3–31.5 FPS trên GPU RTX 5060 Laptop và ~7.0–8.6 FPS trên CPU 4 luồng trong lượt đo này. Kết quả thuộc diện PRELIMINARY-v2, không dùng để xếp hạng throughput giữa các detector. [CẦN TRÍCH DẪN tiêu chuẩn thời gian thực cụ thể nếu đưa vào bài báo].
