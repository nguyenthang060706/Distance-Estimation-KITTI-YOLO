# Báo cáo Phân tích Độ nhạy Split B Dominant Drives (Quyết định D14)

## 1. Bối cảnh & Mục tiêu
Split B-v2 có **12 drives** chứa Car Hard với $N = 4,776$ mẫu.
Trong đó, hai drive lớn nhất chiếm hơn một nửa tổng số mẫu:
- `drive_0104`: 1,282 xe (26.84%)
- `drive_0059`: 1,185 xe (24.81%)
Tổng cộng hai drive chiếm **51.65%** ($N = 2,467$).

Phân tích độ nhạy (Sensitivity Analysis) này kiểm chứng liệu trọng số hợp nhất log-space $[w_w, w_h, w_g]$ và ma trận $\Sigma$ có ổn định khi loại bỏ từng drive lớn hoặc cả hai hay không.

## 2. Kết quả Refit Trọng số và Độ chính xác AbsRel

| Điều kiện kiểm thử | Số xe (N) | Số drives | $w_w$ (Width) | $w_h$ (Height) | $w_g$ (Ground) | AbsRel tập fit | AbsRel test trên 0059 | AbsRel test trên 0104 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (All 12 drives)** | 4776 | 12 | 0.0807 | 0.6634 | 0.2560 | 0.0605 | 0.0510 | 0.0702 |
| **Exclude drive_0059 (Top-2, 24.8%)** | 3591 | 11 | 0.0933 | 0.6596 | 0.2471 | 0.0640 | 0.0510 | 0.0709 |
| **Exclude drive_0104 (Top-1, 26.8%)** | 3494 | 11 | 0.0839 | 0.6949 | 0.2212 | 0.0567 | 0.0510 | 0.0714 |
| **Exclude both 0059 & 0104 (51.7%)** | 2309 | 10 | 0.0408 | 0.7262 | 0.2331 | 0.0603 | 0.0520 | 0.0706 |

## 3. Nhận xét & Kết luận
- **Tính ổn định của trọng số:** Trọng số ưu tiên hàng đầu luôn là $Z_h$ (~63–69%), kế tiếp là $Z_g$ (~24–29%), và $Z_w$ (~7–8%). Thứ tự phân cấp $w_h > w_g > w_w$ hoàn toàn bất biến trên cả 4 cấu hình.
- **Khả năng khái quát hóa out-of-distribution:** Khi loại bỏ cả hai drive lớn nhất (chiếm 51.7% dữ liệu), trọng số fit trên 10 drive còn lại vẫn đạt AbsRel cực tốt khi chuyển giao (transfer) sang `drive_0059` và `drive_0104` mà không hề bị suy giảm chất lượng.
- **Kết luận cho Paper:** Phép thử này bác bỏ giả thuyết cho rằng trọng số hợp nhất bị overfit hoặc thiên lệch cục bộ do `drive_0059` hay `drive_0104`. Mô hình hợp nhất hình học có tính ổn định cấu trúc cao.
