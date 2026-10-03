# Báo cáo Chuyển dịch Phân chia Drive: Splits-v1 sang Splits-v2 (Quyết định D14)

## 1. Bối cảnh & Lý do thực hiện D14
- Ở `splits-v1`, Split C bị tập trung thái quá: `drive_0059` chiếm tới 1.185 / 2.331 xe (**50.8%**) khiến $n_{\text{eff}}(C) = 3.2$.
- D14 thực hiện hoán đổi nội bộ các drive thuộc cụm $(B \cup C \cup T)$, **bảo toàn tuyệt đối 100% Split A và Split V** (hash A `4402edf8...`, V `a093a974...` không đổi).
- Simulated Annealing với 200 seed, chọn seed 85 theo đúng protocol đăng ký trước (tối đa hóa $\min(n_{\text{eff}})$, kiểm soát chặt KS depth $\le 0.07$).

## 2. Thống kê tổng hợp v1 vs v2

| Thuộc tính | v1 B | v1 C | v1 T | v2 B | v2 C | v2 T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Số frame** | 1.496 | 749 | 1.122 | **1.499** | **766** | **1.102** |
| **Tổng drive** | 28 | 25 | 28 | **34** | **18** | **29** |
| **Drive có Car Hard** | 10 | 11 | 11 | **12** | **10** | **10** |
| **Số Car Hard** | 4.210 | 2.331 | 3.273 | **4.776** | **1.826** | **3.212** |
| **Số xe >50 m** | 9 | 8 | 35 | **19** | **0** | **33** |
| **$n_{\text{eff}}$** | 5.4 | 3.2 | 5.6 | **5.29** | **5.28** | **5.28** |
| **Top-1 drive share** | 26.8% | 50.8% | 27.5% | **26.8%** | **28.7%** | **23.3%** |

*(Ghi chú: Tổng $B \cup C \cup T$ bảo toàn bất biến: 3.367 frame, 81 drive, 9.814 Car Hard, 52 xe >50 m).*

## 3. Kiểm định Kolmogorov-Smirnov (KS) về phân phối độ sâu Z trên Splits-v2 (Car Hard)

- **KS(B, C):** statistic = **0.0460** ($p = 7.1319e-03$) $\le 0.0700$ (ĐẠT)
- **KS(B, T):** statistic = **0.0588** ($p = 3.1749e-06$) $\le 0.0700$ (ĐẠT)
- **KS(C, T):** statistic = **0.0556** ($p = 1.4080e-03$) $\le 0.0700$ (ĐẠT)

## 4. Chi tiết các Drive chuyển dịch giữa v1 và v2 (có chứa Car Hard)

| Drive | v1 Split | v2 Split | Car Hard | Frames | Ghi chú tác động thiết kế |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `2011_09_26_drive_0002_sync` | B | T | 2 | 12 | Rất ít mẫu |
| `2011_09_26_drive_0005_sync` | T | C | 60 | 48 | Chuyển sang C |
| `2011_09_26_drive_0017_sync` | B | T | 41 | 114 | Từng ở B-v1 |
| `2011_09_26_drive_0019_sync` | C | T | 253 | 280 | Giảm tải cho C |
| `2011_09_26_drive_0027_sync` | T | C | 25 | 188 | |
| `2011_09_26_drive_0028_sync` | T | B | 92 | 430 | Bổ sung cho B |
| `2011_09_26_drive_0039_sync` | B | T | 628 | 396 | **Drive lớn ở B-v1 chuyển sang T-v2** (Limitations) |
| `2011_09_26_drive_0052_sync` | C | T | 12 | 78 | |
| `2011_09_26_drive_0059_sync` | C | B | 1.185 | 370 | **Drive lớn nhất chuyển từ C sang B** (B-v2 top-1 = 24.8% Car Hard) |
| `2011_09_26_drive_0060_sync` | C | B | 25 | 72 | |
| `2011_09_26_drive_0084_sync` | T | B | 845 | 383 | Bổ sung cho B |
| `2011_09_26_drive_0095_sync` | B | T | 747 | 267 | **Drive lớn ở B-v1 chuyển sang T-v2** (Limitations) |
| `2011_09_26_drive_0096_sync` | B | C | 524 | 165 | **Drive lớn ở B-v1 chuyển sang C-v2** (Limitations) |
| `2011_09_26_drive_0106_sync` | T | B | 346 | 134 | Bổ sung cho B |
| `2011_09_26_drive_0113_sync` | C | B | 9 | 158 | |
| `2011_09_28_drive_0034_sync` | C | B | 6 | 45 | |
| `2011_09_28_drive_0047_sync` | T | C | 12 | 26 | |
| `2011_09_29_drive_0004_sync` | T | C | 364 | 159 | Bổ sung cho C |

## 5. Đánh giá Tác động & Ghi chú vào Báo cáo/Paper (Limitations)
1. **Không có rò rỉ vào Detector:** A và V hoàn toàn không đổi giữa v1 và v2.
2. **Không có rò rỉ tham số:** Tham số hình học `geometry-v2` chỉ fit trên B-v2 mới, hoàn toàn chưa nhìn thấy C và T.
3. **Ảnh hưởng gián tiếp mức thiết kế (Design-level Indirect Influence):**
   - Các drive lớn như `drive_0039` (628 xe), `drive_0095` (747 xe), `drive_0096` (524 xe) nằm trong B-v1 khi tiến hành phân tích Day 4 (hình thành quy tắc mask viền ảnh, cổng kiểm định D9, chẩn đoán dải 0–10 m).
   - Khi sang v2, các drive này chuyển sang T và C. Tuy nhiên, quy tắc mask là tính chất hình học vật lý của camera/sensor, không phải siêu tham số tối ưu hóa số học. Cần ghi rõ điểm này trong mục Limitations của Paper.
4. **Vấn đề tập trung của `drive_0059` trên Split B:**
   - `drive_0059` có 1.185 Car Hard (~24.8% lượng xe của B). Cần thực hiện kiểm tra độ nhạy (Sensitivity Analysis): fit lại trọng số hợp nhất và mô hình residual khi loại bỏ `drive_0059` để xác nhận tính vững chắc (robustness).
5. **Đặc điểm dải >50 m trên Split C:**
   - Split C v2 không có mẫu Car Hard nào ở dải >50 m ($n=0$). Đánh giá detector trên Split C chỉ phản ánh cự ly $\le 50$ m.
