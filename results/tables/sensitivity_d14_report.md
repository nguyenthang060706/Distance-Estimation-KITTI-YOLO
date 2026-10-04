# Báo cáo Phân tích Độ nhạy Split B Dominant Drives (Quyết định D14)

## 1. Bối cảnh & Mục tiêu
Split B-v2 có **12 drives** chứa Car Hard với $N = 4,776$ mẫu.
Trong đó, hai drive lớn nhất chiếm hơn một nửa tổng số mẫu:
- `drive_0104`: 1,282 xe (26.84%)
- `drive_0059`: 1,185 xe (24.81%)
Tổng cộng hai drive chiếm **51.65%** ($N = 2,467$).

Mục tiêu là mô tả sự biến thiên của trọng số hợp nhất $[w_w, w_h, w_g]$ và độ lệch AbsRel khi loại bỏ từng drive lớn hoặc cả hai khỏi tập khớp trọng số.
Toàn bộ metrics được tính bằng `src.evaluation.eval.depth_metrics` theo quy ước D18/D20.

## 2. Kết quả Refit Trọng số và Độ chính xác AbsRel

| Điều kiện kiểm thử | $N_{\text{fit}}$ ($n_{\text{valid}}$) | Drives | $w_w$ (Width) | $w_h$ (Height) | $w_g$ (Ground) | Fit AbsRel | Test `drive_0059` ($N=1,185$) | Test `drive_0104` ($N=1,282$) | Macro AbsRel (12 drives) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (All 12 drives)** | 4776 (4729) | 12 | 0.0807 | 0.6634 | 0.2560 | 0.0605 | 0.0510 [in-sample] | 0.0702 [in-sample] | 0.0631 |
| **Exclude drive_0059 (Top-2, 24.8%)** | 3591 (3564) | 11 | 0.0933 | 0.6596 | 0.2471 | 0.0640 | **0.0510** [OOS] | 0.0709 [in-sample] | 0.0621 |
| **Exclude drive_0104 (Top-1, 26.8%)** | 3494 (3448) | 11 | 0.0839 | 0.6949 | 0.2212 | 0.0567 | 0.0510 [in-sample] | **0.0714** [OOS] | 0.0610 |
| **Exclude both 0059 & 0104 (51.7%)** | 2309 (2283) | 10 | 0.0408 | 0.7262 | 0.2331 | 0.0603 | **0.0520** [OOS] | **0.0706** [OOS] | 0.0644 |

> [!NOTE]
> Các ô in đậm kèm ký hiệu `[OOS]` là kết quả kiểm thử ngoài mẫu (Out-Of-Sample) thực sự (drive mục tiêu bị loại hoàn toàn khỏi tập fit trọng số). Các ô `[in-sample]` được cung cấp để đối chiếu đường cơ sở.

## 3. Nhận xét định lượng (Mô tả dữ liệu theo D18/D20)
1. **Biên độ biến thiên của trọng số:**
   - $w_h$ (Height): dao động trong khoảng **66.0% – 72.6%** (0.6596 đến 0.7262).
   - $w_g$ (Ground): dao động trong khoảng **22.1% – 25.6%** (0.2212 đến 0.2560).
   - $w_w$ (Width): dao động trong khoảng **4.1% – 9.3%** (0.0408 đến 0.0933), tức biên độ thay đổi gấp **2.3 lần** khi loại bỏ đồng thời hai drive lớn.
   - Thứ tự phân cấp tương đối $w_h > w_g > w_w$ giữ nguyên trên tất cả 4 tập con.

2. **Độ lệch AbsRel trên các ô ngoài mẫu (Out-of-sample $\Delta$):**
   - Khi loại `drive_0059`: AbsRel OOS trên `drive_0059` là **0.0510** (chênh lệch so với baseline: $\Delta = +0.0000$).
   - Khi loại `drive_0104`: AbsRel OOS trên `drive_0104` là **0.0714** (chênh lệch so với baseline: $\Delta = +0.0012$, tương đương tăng tương đối +1.7%).
   - Khi loại đồng thời cả hai drive (mất 51.7% dữ liệu fit):
     - OOS trên `drive_0059`: **0.0520** ($\Delta = +0.0010$, tăng tương đối +2.0%).
     - OOS trên `drive_0104`: **0.0706** ($\Delta = +0.0004$, tăng tương đối +0.6%).
   - Macro AbsRel qua toàn bộ 12 drives: dao động trong biên hẹp từ **0.0610 đến 0.0644**.

3. **Lưu ý phương pháp luận cho Paper:**
   - Do Split B chỉ có 12 cụm drive ($N=4,776$), khoảng tin cậy của ước lượng hiệp phương sai là thô.
   - Việc loại bỏ các drive lớn dẫn đến sự tái phân bổ đáng kể tỷ trọng $w_w$ (từ 8.1% xuống 4.1%), cho thấy ước lượng $w_w$ nhạy hơn với thành phần drive so với $w_h$ và $w_g$.
   - Tuy nhiên, độ lệch AbsRel ngoài mẫu trên hai drive lớn bị loại vẫn nằm trong phạm vi nhỏ ($\le +0.0012$, tức $\le 2.0\%$ tương đối), phản ánh cấu trúc tương quan giữa 3 cue hình học có tính ổn định tương đối giữa các cụm.
