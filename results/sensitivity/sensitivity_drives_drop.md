# Sensitivity Analysis: Split B Dominant Drives Drop (Task T09.1)

> **Context (Decisions D14, D31, D36, D85):** Evaluation of fusion weight stability and geometric ranging performance on detector `yolo11s_640` when the two largest drives (`drive_0059` and `drive_0104`) are excluded.
> **Mẫu số phân tích:** Tập True Positives vượt ngưỡng hoạt động `pass_thr` của `yolo11s_640` trên Split B ($N=3,523$, trong đó `drive_0059` chiếm 860 mẫu = 24.41%, `drive_0104` chiếm 760 mẫu = 21.57%, tổng hai drive chiếm 1,620 mẫu = 45.98%). Khác với mẫu số $N=4,776$ Ground Truth Car Hard ở D14 (hai drive chiếm 51.65%).
> **Lưu ý trọng số:** Trọng số $w_w = 0.0000$ là kết quả refit trên bounding box detector (NNLS active theo D31), khác với trọng số $[0.0807, 0.6634, 0.2560]$ fit trên Ground Truth bbox ở Day 4/D14.

## Fusion Weights and Out-Of-Sample Error Stability

| Test Condition | $N_{fit}$ ($n_{complete}$) | Cụm Drives $k$ | $w_w$ (Width) | $w_h$ (Height) | $w_g$ (Ground) | Fit AbsRel | Test `drive_0059` ($N=860$) | Test `drive_0104` ($N=760$) | Macro AbsRel (12 drives, CI thô) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (All 12 drives)** | 3523 (3361) | 12 | 0.0000 | 0.6984 | 0.3016 | 0.0604 | 0.0636 [in-sample] | 0.0665 [in-sample] | 0.0755 |
| **Exclude drive_0059 (Top-1, 24.4%)** | 2663 (2552) | 11 | 0.0000 | 0.7142 | 0.2858 | 0.0593 | 0.0637 [OOS] | 0.0668 [in-sample] | 0.0750 |
| **Exclude drive_0104 (Top-2, 21.6%)** | 2763 (2611) | 11 | 0.0000 | 0.7151 | 0.2849 | 0.0586 | 0.0637 [in-sample] | 0.0668 [OOS] | 0.0749 |
| **Exclude both 0059 & 0104 (46.0%)** | 1903 (1802) | 10 | 0.0000 | 0.7577 | 0.2423 | 0.0560 | 0.0640 [OOS] | 0.0679 [OOS] | 0.0739 |

## Quantitative Findings & Synthesis
1. **Độ ổn định của trọng số hợp nhất:**
   - Trọng số chiều cao $w_h$ duy trì vai trò chủ đạo ổn định: dao động trong khoảng **69.8% – 75.8%** (0.6984 đến 0.7577).
   - Trọng số mặt đất $w_g$ dao động trong khoảng **24.2% – 30.2%** (0.2423 đến 0.3016).
   - Trọng số bề rộng $w_w$ được giải về **0.0000** (do cơ chế ràng buộc NNLS loại bỏ trọng số âm khi cue bề rộng có độ biến thiên lớn trên bounding box detector).
   - Thứ tự phân cấp tương đối $w_h > w_g > w_w$ được bảo toàn nghiêm ngặt trên cả 4 tập điều kiện.
2. **Độ lệch sai số ngoài mẫu (Out-Of-Sample AbsRel):**
   - Khi loại bỏ hoàn toàn `drive_0059` (mất 24.4% dữ liệu fit), sai số OOS trên chính drive này là **0.0637** (so với in-sample 0.0636, chênh lệch $\Delta = +0.0001$).
   - Khi loại bỏ hoàn toàn `drive_0104` (mất 21.6% dữ liệu fit), sai số OOS trên chính drive này là **0.0668** (so với in-sample 0.0665, chênh lệch $\Delta = +0.0003$).
   - Khi loại bỏ đồng thời cả 2 drive (mất 46.0% dữ liệu fit), sai số OOS trên 0059 là **0.0640** ($\Delta = +0.0004$) và trên 0104 là **0.0679** ($\Delta = +0.0014$).
   - Macro AbsRel qua toàn bộ 12 drives chỉ biến thiên trong dải hẹp từ **0.0739 đến 0.0755** (CI thô, 10–12 cụm).
3. **Kết luận:** Trọng số hợp nhất hình học và sai số suy luận trên cụm không bị phụ thuộc quá mức vào bất kỳ drive đơn lẻ nào trong Split B.