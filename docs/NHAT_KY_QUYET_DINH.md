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
- ✅ **Chia dữ liệu theo drive**, 5 tập: A (50%) / V (5%) / B (20%) / C (10%) / T (15%). Gán drive greedy theo kích thước để khớp tỉ lệ frame, kiểm tra bằng KS stat Z và chi-square class.
- ✅ **Đóng băng split (D1):** Đã gắn tag git `splits-v1` (seed=42). Split cũ (hash `fd3c...`, V = 6 drive) là **superseded** do bug mapping.
- ✅ **Phạm vi lớp (D2):** Residual, CQR và CI chỉ làm cho **Car**. Van chỉ chạy (a)–(d) với prior riêng, báo cáo mô tả. Truck chỉ báo cáo ở mức detector (P/R) và số lượng.
- ✅ **Dải khoảng cách (D3):** 5 dải thống nhất (0–10, 10–20, 20–30, 30–50, >50 m) kèm n và cờ `*` nếu n < 100, thêm một hàng gộp ">30 m" cho phân tích coverage có điều kiện.
- ✅ **Quy ước chạy T (D4):** Tuần 2 chỉ suy luận trên B, C (và A để chẩn đoán). T chỉ chạy một lần duy nhất trong script cuối tuần 3.
- ✅ **imgsz (D5, sửa lời):** Chốt 640 do giới hạn phần cứng VRAM 8 GB (RTX 5060 Laptop GPU). 960 không được so sánh bằng mAP do chạm trần VRAM ở batch 16 gây thrashing bộ nhớ chia sẻ. Dùng 640 chung cho cả 3 detector. Thử 960 (batch 8) chỉ là ablation tùy chọn trên YOLOv8s, không chặn tiến độ. Ghi Limitations về khả năng phát hiện xe ở xa.
- ✅ **Ngưỡng conf (D6, bổ sung chi tiết):** Cùng quy tắc F1 tối đa của Car Hard trên V, áp riêng cho từng detector. Quần thể V gồm 611 Car Hard trên 6 drive ($n_{\text{eff}} = 3.1$). Để tránh bẫy gai nhiễu cục bộ, đường F1 được làm trơn bằng moving average với cửa sổ 0.05 ($\pm 0.025$). Ngưỡng tối ưu thu được trên `last.pt`: `yolov8s` = 0.790 (smoothed F1 = 0.8355, raw F1 = 0.8368, raw argmax trùng tại 0.790); `yolo11s` = 0.700 (smoothed F1 = 0.8370, raw F1 = 0.8363, raw argmax trùng tại 0.700); `yolov5su` = 0.740 (smoothed F1 = 0.8240, raw F1 = 0.8227, raw argmax trùng tại 0.740). Cả 3 ngưỡng đều nằm sâu bên trong lưới quét [0.05, 0.90], không chạm biên.
- ✅ **Recipe detector (D7):** epochs=100, patience=100 (tắt early stopping) cho cả 3 detector, giữ best.pt theo fitness mặc định và ghi lại epoch được chọn. Lý do: các checkpoint pilot/cũ dừng sớm ở epoch 14–17 là đỉnh nhiễu (winner's curse trên tập V 374 ảnh), LR chưa annealing (còn ~0.0010) và close_mosaic chưa kích hoạt. Các run cũ đổi tên thành `prelim_*` (superseded), không dùng cho B/C/T.
- ✅ **Quần thể khớp (D8):** GT là Car thỏa mức lọc Hard. Khớp GT Hard trước; nếu không khớp, kiểm tra GT ngoài Hard và DontCare. Detection khớp GT ngoài Hard hoặc DontCare bị bỏ qua, không tính TP/FP và không ranging. Áp dụng cho cả chọn ngưỡng D6 và pipeline ranging.
- ✅ **Đặc trưng của mô hình chỉ dùng thông tin có lúc suy luận.** Không dùng nhãn truncated/occluded/alpha của KITTI làm đặc trưng; chỉ dùng để nhóm phân tích lỗi.
- ✅ **Calibration:** dùng P2 (và R0_rect) riêng từng ảnh; bbox của YOLO phải map về tọa độ ảnh gốc.
- ✅ **T chỉ chạy một lần**, với cấu hình đã đóng băng (git tag). Ablation chạy trên dự đoán out-of-fold của B∪C.
- ✅ **Số liệu tập V:** Chỉ dùng chọn checkpoint và ngưỡng conf; không dùng báo cáo hiệu năng detector trong paper (hiệu năng tính trên B∪C và T). Số FPS từ Ultralytics val không dùng làm latency end-to-end (sẽ đo riêng ở Tier 1).
- ✅ **Gate §8.1 & Đánh giá công bằng (D9):** Gate tính trên các dải $n \ge 100$ không chồng lấn (4 dải: 0–10, 10–20, 20–30, 30–50 m; không tính hàng gộp ">30 m"). Báo cáo thêm so sánh cặp trên tập chung (common support) và bảng phân rã theo tổ hợp cue (pattern breakdown).
- ✅ **Đóng băng hình học & Tham số hiệu dụng (D10):** $\delta = -4.6782$ px và $H_{\text{cam}} = 2.0422$ m là tham số hiệu dụng (effective ground-plane parameters) fit trên A (hash A `4402edf8...`) để bù chênh lệch giữa góc tiếp đất gần nhất ($Z_{\text{closest}}$) và tâm xe ($Z_{\text{center}}$). Trọng số hợp nhất hiện hành fit trên Split B-v2 mới (hash B `0f83c354...`, 12 drive có Car Hard, $N=4,776$): $[w_w, w_h, w_g] = [0.0807, 0.6634, 0.2560]$ (tổng = 1.0, shrinkage $\alpha = 0.0008$). Đóng băng vào `configs/geometry_params.yaml` gắn tag `geometry-v2`. *(Ghi chú: Trọng số cũ $[0.0691, 0.6500, 0.2809]$ fit trên Split B-v1 cũ hash `1242...` tag `geometry-v1` đã chính thức bị thay thế / superseded do D14)*.
- ✅ **Phân tách đặc trưng suy luận khỏi GT (D11):** Pipeline suy luận tuyệt đối tách biệt các cột đặc trưng khỏi các trường nhãn ground truth (`gt_*`, `alpha`, `truncated`, `occluded`, `depth`/`location_z`). Phải có unit test chặn và bảo đảm bộ trích đặc trưng không được đọc `gt_*` (đã có trong `tests/test_feature_guard.py`).
- ✅ **Chốt checkpoint detector bằng `last.pt` (D12):** Dùng `last.pt` (epoch 100) cho cả 3 detector, chốt trước khi xem kết quả huấn luyện lại, và ghi epoch của `best.pt` vào log để đối chứng. Lý do: Split V chỉ có 6 drive chứa Car Hard (trong đó 1 drive chiếm 41%), việc chọn `best.pt` theo đỉnh fitness trên V dễ rơi vào winner's curse / đỉnh nhiễu của vài cụm; `last.pt` tại epoch 100 bảo đảm LR annealing hoàn tất và close_mosaic đã kích hoạt đầy đủ trên toàn bộ Split A.
- ✅ **Xử lý xe suy giảm cue & pattern 000 (D13):** Toàn bộ xe thỏa Car Hard (kể cả 16.3% mang pattern `000` ở 0–10m trên Split B-v2 [47/288 xe] không có cue hình học nào do hiệu ứng cắt viền; con số cũ 13.5% trên B-v1 đã superseded) đều thuộc quần thể đánh giá ranging và được dự đoán bằng mô hình direct ranging (e) làm fallback kèm cờ `fallback_flag=True` (metadata, không nằm trong feature whitelist theo D28). C và T đều đi qua đúng pipeline fallback này để bảo toàn tính exchangeability, không loại bỏ ca khó khỏi bảng tổng hợp. $Z_{\text{base}} = Z_d$ nếu $\ge 1$ cue hợp lệ, ngược lại là $Z_e$. Mô hình (e) fit trên toàn bộ B. Target $r = \ln Z_{gt} - \ln Z_{\text{base}}$. Conformalize chung trên C; coverage theo cờ dùng làm chẩn đoán.
- ✅ **Phân bổ lại drive cho B/C/T, giữ nguyên A và V (D14):** Audit concentration phát hiện Split C cũ bị `drive_0059` chi phối 50.8% ($n_{\text{eff}} = 3.2$), gây rủi ro sập coverage CQR cho RQ3. Quyết định D14: phân bổ lại drive giữa B, C, T qua tìm kiếm tối đa 200 seed simulated annealing, protocol định sẵn trong docstring của script, chỉ đọc nhãn (Car Hard count và Z depth), tuyệt đối không dùng kết quả mô hình (không dùng từ "pre-registered"). Giữ nguyên 100% Split A và V. Tiêu chí: $\text{top1\_share} \le 0.35$; $\ge 10$ drive Car Hard mỗi tập; KS $\le 0.07$ cả 3 cặp; frame ratio lệch $\le 3$ điểm % so với 20/10/15. Seed 85 được chọn tối ưu: $n_{\text{eff}}$ cân bằng ở mức 5.28 cho cả 3 tập, top1 C giảm xuống 28.7%, KS B-C=0.0460, B-T=0.0588, C-T=0.0556 (tất cả đều $\le 0.0588$). Đóng băng `splits-v2` và refit trọng số hình học fit trên B mới thành `geometry-v2`. Xuất bảng chuyển dịch drive chi tiết vào `results/tables/split_migration_v1_to_v2.md`. Thêm phân tích độ nhạy (Sensitivity Analysis): fit lại trọng số hợp nhất và mô hình residual khi loại bỏ lần lượt `drive_0059` (top-2, 1.185 xe) và `drive_0104` (top-1, 1.282 xe) khỏi Split B (hai drive chiếm 51.65% Car Hard). Ghi chú Limitations: 10 drive từ B-v1 sang T-v2 và 3 drive từ B-v1 sang C-v2 có ảnh hưởng gián tiếp ở mức thiết kế định tính trong Day 4 (quy tắc mask viền).
- ✅ **Ngưỡng IoU khớp = 0.5 và Trạng thái đánh giá 4 lớp (D15):** Xây dựng module dùng chung `src/detection/matching.py` trả về 4 trạng thái rõ ràng: `TP`, `FP`, `IGNORED_NONHARD`, `IGNORED_DONTCARE`. Ưu tiên GT Hard (unmatched) trước; nếu trùng GT Hard đã ghép thì gán `FP` (duplicate, kể cả khi nằm trong vùng DontCare); kiểm tra GT non-Hard rồi DontCare để bỏ qua. Ngưỡng IoU khớp mặc định là **0.5** theo đúng kế hoạch v4 §5.1 và ngưỡng F1 của D6 (mAP@0.7 báo cáo riêng). Tiêu chí DontCare mặc định là `iou` (IoU $\ge 0.5$) tái lập 100% kết quả ngưỡng D6; hỗ trợ `area_pred` theo devkit chính thức.
- ✅ **Khớp Greedy theo điểm số giảm dần thay cho Hungarian (D16):** Chốt thuật toán matching là Greedy theo confidence giảm dần (cập nhật v4 §5.1). Lý do: trong Greedy matching, trạng thái gán của một detection chỉ phụ thuộc vào các detection có điểm số cao hơn nó; do đó khớp một lần ở sàn `conf >= 0.05` rồi lọc cờ `pass_thr` tương đương hoàn toàn về mặt toán học với việc lọc ngưỡng trước rồi mới khớp. Thuật toán Hungarian phụ thuộc vào tập ứng viên tổng thể nên không có tính chất bảo toàn phân cấp (nested filtering) này.
- ✅ **Quy tắc Out-of-Fold cho mô hình direct (e) trên Split B (D16b / Out-of-fold rule):** Khi tạo target khoảng cách $r = \ln Z_{gt} - \ln Z_{\text{base}}$ trên Split B, nếu vật thể mang pattern `000` cần fallback sang $Z_e$, thì $Z_e$ phải được tính dạng out-of-fold theo drive (Grouped CV). Mô hình $Z_e$ fit trên toàn bộ B chỉ dùng để suy luận trên C và T.
- ✅ **Quy tắc Out-of-Fold LODO cho $Z_d$ trên Split B khi tạo Target Residual (D16c):** Trọng số hợp nhất $[w_w, w_h, w_g]$ fit trên Split B có độ lệch lạc quan nhỏ in-sample (AbsRel 0.0605 so với 0.0610 LODO OOF). Nhằm bảo đảm target residual $r = \ln Z_{gt} - \ln Z_{\text{base}}$ trên tập huấn luyện B phản ánh đúng sai số thực tế khi suy luận trên C và T (out-of-sample), $Z_d$ trên Split B được tính bằng trọng số LODO OOF qua hàm `fuse_depths_lodo()` trong `src/geometry/fusion.py` (mỗi drive $d$ dùng trọng số $\mathbf{w}_{-d}$ ước lượng từ 11 drive còn lại; kiểm thử bất biến test pass 100%). Trên C và T, $Z_d$ dùng trọng số toàn cục fit trên toàn bộ B (`geometry-v2`). Phân tích độ nhạy D14 (`scripts/eval_sensitivity_d14.py`, báo cáo tại `results/tables/sensitivity_d14_report.md`) mô tả dữ liệu thực nghiệm: thứ tự $w_h$ (66.0–72.6%) $> w_g$ (22.1–25.6%) $> w_w$ (4.1–9.3%) giữ nguyên qua 4 cấu hình; $w_w$ dao động gấp 2.3 lần khi mất 51.7% mẫu; độ lệch AbsRel ngoài mẫu trên `drive_0059` là $+0.0000$ đến $+0.0010$ và trên `drive_0104` là $+0.0004$ đến $+0.0012$ (tức $\le +2.0\%$ tương đối; 12 cụm, CI thô).
- ✅ **Refit trọng số hình học trên detector bboxes (D17):** Trọng số hợp nhất $[w_w, w_h, w_g]$ trong `geometry-v2` là cận dưới lý thuyết trên GT bbox. Khi chuyển sang bbox detector ở Tuần 2, refit trọng số và ma trận hiệp phương sai $\Sigma$ trên Split B riêng cho từng detector (dùng LODO CV để đánh giá). Đồng thời giữ thêm cột đối chứng dùng trọng số fit trên GT bbox để phân rã riêng biệt giữa sai số hình học và sai số detector bbox. Báo cáo tỷ lệ ước lượng hợp lệ (`valid_frac`) và số GT bị bỏ sót (FN) riêng rẽ, không loại bỏ ca khó khỏi bảng tổng hợp.
- ✅ **Báo cáo đồng thời Macro và Pooled, hạ giọng kết luận (D18 & D20):** Báo cáo kết quả phải trình bày song song cả Pooled AbsRel và Macro AbsRel theo drive (unweighted trung bình qua 12 drive). Không sử dụng cụm từ "chứng minh không overfit" hoặc "có ý nghĩa thống kê" khi số lượng drive $\le 12$; ghi chú rõ "CI thô (12 cụm)". Centered covariance giữ nguyên vì kết quả in-sample và OOF tương đương (0.0609 vs 0.0610).
- ✅ **Sửa nhãn góc quan sát $\theta$ theo quy ước KITTI (D19):** Định nghĩa góc nhìn $\theta = \min(|\alpha|, \pi - |\alpha|)$. Theo quy ước KITTI, $|\alpha| \approx \pi/2$ ($\theta > 60^\circ$) là nhìn **Đầu/Đuôi (Front/Rear)** (N=3550, median $w \cdot Z / f_x = 2.279$ m, $Z_w$ AbsRel = 0.2611). Còn $\alpha \approx 0$ hoặc $\pi$ ($\theta < 30^\circ$) là nhìn **Ngang (Side)** (N=183, median $w \cdot Z / f_x = 4.487$ m - tương ứng chiều dài xe, $Z_w$ AbsRel = 0.4241). Ở bin nhìn ngang, $Z_h$ riêng lẻ (0.0482) thắng Fused (0.0800) vì trọng số toàn cục kéo $Z_w$ vào $\to$ Đây là động lực tự nhiên then chốt cho mô hình residual (f) với tỷ số $w/h$ làm proxy cho góc nhìn $\theta$.
- ✅ **Giả thuyết sai số âm dải 0–10m (D21):** Bias của (d) giảm đơn điệu theo khoảng cách (0–10m: $-0.128$, 10–20m: $-0.035$, 20–30m: $+0.014$, 30–50m: $+0.003$). Phép kiểm trên Split B v2 cho thấy Pattern 111 (cả 3 cue hợp lệ, không hề chạm biên) vẫn có bias $-0.1071$ (median $-0.0984$). Do đó, sai số âm ở gần không phải do hiệu ứng biên (border cut) mà chủ yếu do Ground Truth KITTI đo đến tâm hộp 3D ($Z_{\text{center}}$) trong khi các cue thị giác đo đến mặt tiếp xúc gần nhất ($Z_{\text{surface}}$), tạo độ lệch tương đối $\approx l / (2Z)$. Đây là phần residual học được và đưa vào mục Discussion.
- ✅ **Đặc tả hạ tầng suy luận detector `run_inference.py` (D22):**
  - Chạy ở chế độ FP32, `conf=0.05`, `iou_nms=0.7`, `imgsz=640`, matching IoU `--iou-match 0.5`.
  - Cờ `--split` chỉ nhận `A`, `B`, `C`; Split T bị khóa cứng bằng `PermissionError`.
  - Tự động kiểm tra hash split đối chiếu với `split_metadata.json` và kiểm tra mã băm SHA-256 của checkpoint đối chiếu với `configs/detector/checkpoints.yaml` trước khi thực thi.
  - Tách thành **3 artifact** lưu trữ:
    1. `{model}_{split}_detections.parquet`: Chỉ gồm dữ liệu test-time (tọa độ ảnh gốc, `confidence`, `class_id: 0`, cờ `pass_thr`, intrinsics P2, kích thước ảnh) - chuẩn hóa khớp tuyệt đối với `feature_extractor.py`.
    2. `{model}_{split}_matches.parquet`: Phục vụ đánh giá (trạng thái TP/FP/IGNORED, `matched_gt_idx`, `matched_iou`).
    3. `{model}_{split}_gt.parquet`: Toàn bộ GT Car Hard phục vụ tính Recall, False Negative và target residual $Z_{\text{gt}}$.
  - Tự động ghi nhật ký JSONL vào `runs/inference_log.jsonl` kèm phiên bản `ultralytics`, `torch`, `seed: 42`, `iou_match`, `splits_version`.
- ✅ **Quần thể ranging và thời điểm join nhãn GT (D23):** Quần thể ranging đánh giá hiệu năng khoảng cách định nghĩa nghiêm ngặt là các detection True Positive ($TP$) đạt ngưỡng `pass_thr = True` (D6) và khớp với Car Hard qua Greedy matching IoU 0.5 (D15, D16). Ma trận đặc trưng `features` được trích xuất hoàn toàn trước; các nhãn `z_gt`, `matched_iou`, `difficulty` chỉ được join sau vào DataFrame `eval` riêng biệt (D11, D22). Các phân tích ở sàn `conf >= 0.05` (`floor`) và các mức `eps` khác chỉ là bài toán ablation, không làm thay đổi cấu hình đóng băng.
- ✅ **Quy tắc chọn cấu hình và tuning Out-of-Fold trên B (D24):** Mọi quyết định lựa chọn siêu tham số, ngưỡng và cấu hình mô hình residual/conformal chỉ được tối ưu hóa dựa trên dự đoán Out-of-Fold (OOF) của Split B (thông qua nested Grouped CV theo drive). Dữ liệu Split C tuyệt đối không được dùng để tuning, chỉ dùng để hiệu chuẩn conformal và báo cáo độ phủ mô tả; OOF trên $B \cup C$ chỉ dùng để trình bày sau khi đã đóng băng pipeline.
- ✅ **Không gian tìm kiếm XGBoost nhỏ và Pre-registration (D25):** Grid tìm kiếm cho mô hình residual XGBoost được giới hạn chặt chẽ $\le 24$ cấu hình, bắt buộc `max_depth <= 4`, `min_child_weight` đủ lớn để tránh overfit theo cụm xe, `learning_rate = 0.05`, `n_estimators` cố định trong từng fold (không dùng early stopping trên fold đánh giá). Bắt buộc có baseline tuyến tính tối thiểu (f0) và direct depth (e) để đối chứng. Cấu hình cũ trong `configs/residual/residual_config.yaml` (depth=6, 500 cây) chính thức bị thay thế bởi `residual_prereg_v1.yaml`.
- ✅ **Giao thức độ phủ Conformal và tính khả hoán B vs C (D26):** Trong giai đoạn phát triển (dev), sử dụng LODO calibration trên Split C (hiệu chỉnh trên $C \setminus d$, kiểm tra trên $d$). Để kiểm tra độ ổn định độ phủ, thực hiện 20 lần chia lại $B \cup C$ theo drive (seeds 0–19, tỷ lệ 50% fit / 25% calib / 25% eval theo số xe Car Hard). Phân nhóm Mondrian bin theo độ sâu dự đoán $\hat{Z}$, gộp bin $\ge 30$ m; mọi khoảng tin cậy từ cluster bootstrap đều ghi chú rõ "CI thô ($k$ cụm)".
- ✅ **Quy trình chạy nghiệm thu duy nhất trên Split T (D27):** Split T chỉ được phép thực thi đúng một lần duy nhất qua script `scripts/run_final_T.py` do con người chạy trực tiếp (T12). Script được bảo vệ bằng 4 lớp guard: (1) kiểm tra tag git `final-config-v1` trùng với HEAD; (2) kiểm tra cây làm việc sạch; (3) kiểm tra mã băm split và checkpoint; (4) lock file `runs/final_T.lock` (chặn tuyệt đối lần chạy thứ hai). Guard công khai `PermissionError` trên `run_inference.py` giữ nguyên.
- ✅ **Phân loại metadata cho cờ fallback (D28):** Cờ `fallback_flag` chỉ là nhãn siêu dữ liệu (metadata) phục vụ phân tích nhóm và phân tầng độ phủ chẩn đoán; tuyệt đối **không** được đưa vào `DEFAULT_FEATURE_WHITELIST` của bộ trích đặc trưng.
- ✅ **Nguồn trọng số hình học cho target residual Z_base (D29):** Khi tạo target residual $r = \ln Z_{gt} - \ln Z_{\text{base}}$ trên Split B, $Z_d$ bắt buộc sử dụng trọng số LODO OOF theo drive (`fuse_depths_lodo`) để tránh lạc quan in-sample (D16c). Trên Split C và T, $Z_d$ sử dụng vector trọng số toàn cục và ma trận $\Sigma$ được fit trên toàn bộ Split B.
- ✅ **Đặc trưng dẫn xuất ln_z_base cho mô hình residual (D30):** Đại lượng `ln_z_base` là đặc trưng dẫn xuất (derived feature, suy ra được lúc suy luận từ các cue hợp lệ hoặc $Z_e$), được cho phép sử dụng trong mô hình hồi quy tuyến tính (f0) và XGBoost (f). `ln_z_base` nằm trong tập `DERIVED_FEATURES` riêng biệt với guard test độc lập, không xâm phạm whitelist gốc D11, và sẽ có bài kiểm tra ablation bỏ `ln_z_base` ở T05.
- ✅ **Cơ chế kẹp trọng số $w_w \to 0$ khi refit trên bbox detector (D31):** Khi refit ma trận hiệp phương sai sai số $\Sigma_{\text{detector}}$ trên bbox detector (D17, T02), tương quan sai số giữa chiều rộng và chiều cao đổi dấu từ âm sang dương ($\Sigma_{wh}$ từ $-0.0014$ trên GT thành $+0.0094$ trên YOLOv8s; $\Sigma_{wg}$ từ $-0.0053$ thành $+0.0008$). Do phương sai của chiều rộng $\Sigma_{ww} = 0.0768$ lớn gấp 10 lần chiều cao $\Sigma_{hh} = 0.0071$ và mất hiệu ứng bù trừ (hedging), nghiệm tối ưu không ràng buộc đẩy trọng số chiều rộng về giá trị âm; thuật toán NNLS kẹp chặt $w_w \to 0.0000$, dồn trọng số sang $[w_h, w_g] \approx [0.70, 0.30]$. Hiện tượng này phản ánh sự thoái hóa thông tin của cue $Z_w$ dưới tác động của jitter detector và góc nhìn xe, hoàn toàn phù hợp với cơ sở lý thuyết của D19.
- ✅ **Phân rã sai số theo cue $\times$ dải, Survivorship Bias và Nguyên tắc RQ2 (D32):**
  1. *Phân rã đa chiều:* Hiệu năng hình học (a)–(d) phải luôn được phân rã đồng thời theo từng cue, từng dải khoảng cách và từng detector trên cùng tập TP chung.
  2. *Survivorship Bias (Thiên lệch kẻ sống sót):* Kết quả so sánh sai số giữa detector bbox và GT bbox có $\Delta \approx 0.0000$ (CI chứa 0) chỉ đúng trên tập **True Positives ($TP$)** đã qua ngưỡng tin cậy. Ở cự ly xa ($30\text{--}50$ m), detector bỏ sót gần một nửa số xe (Recall $\sim 50\text{--}54\%$). Do đó, độ phủ và sai số của mô hình ranging phải luôn được ghi rõ là có điều kiện trên tập đối tượng phát hiện thành công, và số lượng $FN$ phải được báo cáo song hành.
  3. *Nguyên tắc RQ2 (So sánh các detector):* Khi đánh giá ảnh hưởng của các detector khác nhau đến sai số khoảng cách, nếu 95% khoảng tin cậy từ cluster bootstrap chồng lấn nhau hoặc chứa 0, báo cáo trung thực kết quả là *"không phân biệt được sự khác biệt có ý nghĩa"* (indistinguishable), tuyệt đối không gượng ép xếp hạng thứ bậc detector.
- ✅ **Pre-registration mô hình residual (D33):** Đóng băng cấu hình tại `configs/residual/residual_prereg_v1.yaml` (thay thế file nháp cũ): grid XGBoost gồm 12 cấu hình nhỏ gọn (`max_depth <= 4`, `min_child_weight >= 10`, `n_estimators = 200`, `learning_rate = 0.05`), tập đặc trưng chuẩn hóa (f: 17, f0: 5, e: 10), loại bỏ hoàn toàn `class_id` (hằng số 0) và cờ `fallback_flag` (metadata).
- ✅ **Dựng Z_base qua inner-LODO trong tập huấn luyện (D34):** Để ngăn chặn rò rỉ target khi các đối tượng pattern 000 fallback sang $Z_e$ trong các fold train của Split B, $Z_e$ trong tập huấn luyện được dự đoán qua inner-LODO 11 fold giữa 11 drive train. Tập test fold được dự đoán bằng $Z_e$ fit trên toàn bộ 11 drive train.
- ✅ **Quy tắc Code freeze và thực thi đa detector (D35):** Hoàn thiện và kiểm chứng toàn diện pipeline trên detector chính `yolo11s_640`, sau đó đóng băng mã nguồn và chạy lại cùng một logic cho cả 3 detector (`yolo11s_640`, `yolov8s_640`, `yolov5su_640`). Bất kỳ thay đổi logic nào đều bắt buộc chạy lại đồng bộ trên cả 3 detector.
- ✅ **Thứ tự detector thực nghiệm (D36):** Ưu tiên phát triển và kiểm chứng trên `yolo11s_640` trước (detector mới nhất), sau đó mở rộng sang `yolov8s_640` và `yolov5su_640` với cùng mã nguồn và cấu hình đóng băng.
- ✅ **Phạm vi nghiên cứu thành phần Ablation T05 (D37):** Toàn bộ phân tích ablation (bỏ nhóm đặc trưng, bỏ từng cue, MLP vs XGBoost) chỉ thực hiện trên dự đoán OOF của Split B (12 folds LODO), tuyệt đối không chạm Split C (bảo toàn Split C nguyên vẹn cho Conformalization).
- ✅ **Quy chuẩn Drop Single Cue (D38):** Khi loại bỏ cue $Z_k$, loại bỏ $Z_k$ khỏi hợp nhất hình học (refit trọng số và $\Sigma$ trên 2 cue còn lại), bỏ đồng thời $\ln z_k$ và $\text{valid}_k$ khỏi Model (f); pattern 000 đi fallback (e); hàm `fit_fusion_weights` hỗ trợ `cue_names` tùy chọn.
- ✅ **Thứ tự cắt giảm khi trễ tiến độ (D39):** Nếu trễ lịch, cắt Jitter (J1) đầu tiên; nếu trễ T06 dời về W3-4 (T16) cùng khâu CQR, tuyệt đối không dời sang W2-5 (đường găng của T07 CQR).
- ✅ **Chuẩn đo đạc Latency Tier 1 (D40):** GPU chính dùng PyTorch FP16 trên CUDA với `torch.cuda.synchronize()`; CPU chính dùng ONNX Runtime CPU FP32 (cố định 4 luồng).
- ✅ **Tiêu chí Gate T04 cho mô hình Residual (D41):** Đánh giá OOF trên Split B báo cáo song song cả Pooled và Macro theo 12 drive. Gate T04 yêu cầu AbsRel OOF của (f) < (d) ở cả Pooled và Macro; nếu không đạt báo cáo trung thực. (Đổi từ D31 tasks cũ để tránh trùng D31 nhật ký).
- ✅ **Cơ chế Canary xáo nhãn trong train (D42):** Kiểm tra rò rỉ target trong pipeline nested LODO B. Khi xáo ngẫu nhiên nhãn $Z_{\text{gt}}$ trong tập huấn luyện, OOF của mô hình residual (f) không được phép tốt hơn baseline $Z_{\text{base}}$. (Đổi từ D32 tasks cũ để tránh trùng D32 nhật ký).
- ✅ **Protocol fit toàn B cho Model cuối & Serialization full_fw (D43):** Sửa lỗi giao thức fit trên nền in-sample: Khi fit mô hình cuối (f)/(f0) trên toàn bộ Split B để triển khai sang C và T, target $r$ và đặc trưng dẫn xuất $\ln z_{\text{base}}$ bắt buộc sử dụng $Z_{\text{base}}$ OOF từ `B_oof.parquet` (LODO $Z_d$ và OOF $Z_e$), không dùng $Z_{\text{base}}$ in-sample. Trọng số toàn cục `full_fw` (fit trên toàn bộ B) được serialize thành `full_fw.json` cùng mã băm SHA-256 lưu trong `manifest.json`.
- ✅ **Latency Tier 1 sơ bộ & Quy chuẩn Parity Check (D44):** Kết quả T06 được gắn nhãn `PRELIMINARY` (không trích số vào bài báo) do các hạn chế phương pháp luận: double count tiền xử lý ở GPU line, tổng tính theo sum of medians, đường tắt residual, và khác biệt dynamic padding của PyTorch vs static padding của ONNX. Bảng Parity báo cáo độc lập cả hai tiêu chí: Count Parity (đạt) và IoU Parity (chưa đạt ngưỡng 0.95), tuân thủ AGENT_RULES §1.9 không đổi ngưỡng. Phép đo chính thức sẽ thực hiện ở T16 kèm CQR.
- ✅ **Mức phân vị hiệu chỉnh conformal dùng order statistic chính xác (D46):** `np.quantile(..., method='higher')` sử dụng nội suy virtual index $q(n-1)$ gây lỗi off-by-one, lấy phần tử thứ $k+1$ thay vì $k$, tạo ra kết quả bảo thủ sai lệch với lý thuyết conformal prediction hữu hạn mẫu. Mức conformal $\hat{Q}$ được tính trực tiếp từ thống kê thứ tự: $k = \lceil (n + 1)(1 - \alpha) - 10^{-12} \rceil$; nếu $k > n$, $\hat{Q} = +\infty$; nếu $k \le n$, $\hat{Q} = s_{(k)} = \text{sorted\_scores}[k - 1]$.
- ✅ **Công thức khoảng tin cậy CQR, xử lý crossing và Winkler score (D47):** Khoảng tin cậy trong log-space là $[r_{lo}, r_{hi}] = [q_{lo} - \hat{Q}, q_{hi} + \hat{Q}]$. Cả khi tính score và khi dự báo, quantile thô đều được chuẩn hóa qua `sort_quantiles`. Trường hợp crossing ($r_{lo} > r_{hi}$) được đếm và báo cáo công khai số lượng, tuyệt đối không silently clip. Độ sắc nét và phạt miscoverage được đo bằng Winkler score (Gneiting-Raftery 2007) tính trong không gian log ($r$).
- ✅ **Tiêu chí nghiệm thu Gate T07 và quy chuẩn báo cáo cỡ mẫu nhỏ (D48):** Khoảng chấp nhận cho pooled coverage LODO trên Split C là $[85\%, 95\%]$ cho $\alpha = 0.1$ (danh nghĩa $90\%$). Báo cáo macro coverage trên toàn bộ 10 drive song song với macro trên các drive có $n \ge 30$. Phân nhóm `fallback_flag = True` chỉ báo cáo số lượng $n$ và coverage mô tả, không đưa ra kết luận diễn giải do cỡ mẫu quá nhỏ ($n \le 10$).
- ✅ **Chuẩn hóa module suy luận out-of-sample dùng chung giữa Split C và Split T (D49):** Logic suy luận từ cues thô $\to Z_d \to Z_e \to Z_{\text{base}} \to$ features được đặt tại `src/pipeline/apply_frozen.py`, dùng chung cho cả T07, T10 và T12 để bảo đảm tính nhất quán tuyệt đối giữa Split C (calibration) và Split T (test cuối cùng). Có assertion kiểm tra $Z_d$ tính lại từ `full_fw.json` khớp chính xác với $Z_d$ trong cues parquet.
- ✅ **Drive-level Heterogeneity và Cảnh báo Độ phủ (D50):** Báo cáo độ phủ CQR đi kèm cảnh báo: Pooled coverage thấp hơn mức danh nghĩa $2\text{--}3$ điểm phần trăm chủ yếu do hai cụm drive lớn `0057` ($n \approx 250\text{--}260$) và `0004` ($n \approx 309\text{--}325$) chiếm tới $\approx 39.3\%$ mẫu bị under-cover (67%–79%), trong khi 8 cụm còn lại over-cover (96%–100%). Đây là biểu hiện của tính không đồng nhất giữa các cụm (drive-level heterogeneity), phá vỡ một phần giả định exchangeability giữa các drive. Tác vụ T08 (Mondrian CQR, 20 lần chia lại) có nhiệm vụ chẩn đoán sâu hiện tượng này.
- ✅ **Không claim bảo đảm độ phủ cho nhóm Fallback (D51):** Nhóm `fallback_flag = True` (pattern 000, $n=7\text{--}10$) có sai số thực tế $|r| \approx 2.56$ (do Model (e) ước lượng $Z_e \approx 0.45\text{--}0.65$m cho các ca cắt mép cận cảnh $Z_{gt} \approx 5.8\text{--}8.6$m), vượt xa biên độ hiệu chuẩn CQR ($[r_{lo}, r_{hi}] \approx [-0.2, +0.4]$). Khoảng tin cậy CQR do đó không bảo đảm độ phủ cho nhóm fallback (độ phủ thực nghiệm $0.0000$ ở cả 3 detector). Khi báo cáo hoặc triển khai, nhóm fallback chỉ được coi là chỉ báo cảnh báo, không được claim bảo đảm độ phủ conformal.
- ✅ **Báo cáo phân bố độ phủ 20 resplits trên B∪C (D52):** Coverage dev trên $B \cup C$ báo cáo đầy đủ cả pooled, macro theo drive và toàn bộ 20 giá trị per-seed. Độ lệch chuẩn ($\sigma$) qua 20 seed phản ánh độ nhạy với cách phân hoạch (partition sensitivity), không phải khoảng tin cậy (CI). Không xếp hạng giữa Split Conformal, Standard CQR và Mondrian CQR khi chênh lệch giữa các phương án ($\le 3$ điểm phần trăm) nhỏ hơn nhiều so với độ lệch chuẩn ($7\text{--}9$ điểm phần trăm).
- ✅ **Under-coverage hệ thống trên held-out drive (D53):** Trên các cụm drive giữ lại (held-out drive), độ phủ thực tế ở mọi phương án và detector đều thấp hơn mức danh nghĩa $90\%$ (pooled $83.8\text{--}87.4\%$, macro $77.9\text{--}84.5\%$, khoảng dao động $63.6\text{--}99.6\%$), được báo cáo như kết quả thực nghiệm chính cho RQ3/H3. Resplit 20 lần mang tính bi quan do calib chỉ có $\approx 5\text{--}6$ drive (so với 10 drive của Split C ở T07 đạt $87.1\text{--}88.0\%$). Mondrian CQR mang lại cải thiện mô tả ở cự ly gần $0\text{--}10$ m ($+6.8\text{..}+9.3$ điểm phần trăm) nhưng vẫn chưa đạt mức danh nghĩa $90\%$ (đạt $\approx 73.6\text{--}74.9\%$).
- ✅ **Quy chuẩn báo cáo độ phủ có điều kiện (D54):** Mọi bảng độ phủ có điều kiện bắt buộc phải kèm theo số mẫu duy nhất ($n$ unique), số cụm ($k$) và gắn cờ `*` cảnh báo khi $n < 100$. Tuyệt đối không cộng gộp số mẫu lặp lại qua các seed mà không nêu rõ.
- ✅ **Phương án conformal chính cho Split T (D55):** Chọn **Standard CQR** làm phương án conformal chính cho Split T theo đúng thiết kế tiên nghiệm tại Kế hoạch v4 §5.4 (không dựa trên kết quả phát triển trên C). Hai phương án Split Conformal và Mondrian CQR được báo cáo song song dưới dạng đối chứng mô tả (descriptive baseline), tuyệt đối không xếp hạng khi sự khác biệt không có ý nghĩa thống kê (D52).
- ✅ **Calibration và Serialization đầy đủ 3 phương án trên Split C (D56):** Trước khi đóng băng cấu hình (T11), thực hiện calibrate cả 3 phương án (Standard CQR: $\hat{Q}$; Split Conformal: $\hat{Q}$; Mondrian CQR: 4 bin cự ly $[0, 10), [10, 20), [20, 30), [30, \infty)$ với 5 mốc, tuân thủ đúng quy tắc gộp bin $n_{\text{bin}} < 50$ của `coverage_prereg_v1.yaml`) trên toàn bộ 10 drive của Split C và lưu vào artifact `runs/residual/{model}/conformal_calib_C.json`. Tập C đóng vai trò calibration set cố định, hoàn toàn hợp lệ để lưu trước code freeze.
- ✅ **Phạm vi ablation eps mask T09.3 (D57):** Thí nghiệm ablation eps mask $\in \{2, 4, 6\}$ ở tác vụ T09 chỉ đánh giá độ nhạy của tỷ lệ cue hình học hợp lệ (`valid_w, valid_h, valid_g`) và sai số của mô hình hình học thuần túy (d). Đặc trưng `touch_*` trong `src/residual/feature_extractor.py` giữ nguyên hằng số `BORDER_EPS = 2` cố định (tuân thủ Luật 1.4 không sửa code feature extractor).
- ✅ **Chẩn đoán và Hạn chế cho nhóm Fallback (D58):** Chẩn đoán trên Split B OOF ($n \approx 37\text{--}40$) và Split C ($n \approx 7\text{--}10$) cho thấy: mô hình điểm (f) vẫn xử lý được nhóm fallback (AbsRel $\approx 0.12$ trên B OOF), nhưng mô hình quantile $q_{05}/q_{95}$ và CQR không nắm được nhóm này (độ phủ thực nghiệm $0.0000$ theo D51). Nguyên nhân là khoảng CQR được dựng quanh $Z_{\text{base}}$ với biên độ hẹp $[-0.2, +0.4]$ trong log-space, không thể bao phủ được khoảng cách thực tế khi $Z_{\text{base}} \approx 0.5$ m. Quyết định không chèn các quy tắc sửa chữa ad-hoc (như ép chặn $Z_e \ge \min Z$ của B) nhằm bảo toàn tính trung thực và nguyên bản của pipeline đóng băng (D13, D23); đưa toàn bộ hạn chế này vào mục **Limitations** của bài báo.
- ✅ **Quy chuẩn đối soát Dry-run trên Split C (D59):** Trước khi refactor code, chụp "Golden File" (`runs/dryrun_golden_C.json`) từ pipeline hiện hành trên Split C. Sau refactor, chế độ dry-run của `run_final_T.py` chạy qua đúng nhánh `run_inference_core(split="C", allow_test=True)` và so sánh với Golden File theo dung sai quy định rõ ràng: số lượng đối tượng phát hiện khớp 100%, IoU bounding box $\ge 0.999$, mức phân vị $|\Delta \hat{Q}| \le 10^{-6}$, độ phủ $|\Delta \text{Cov}| \le 10^{-4}$, sai số tương đối của Mean Width và Winkler $\le 10^{-4}$; không đòi hỏi bằng bit-by-bit do tính bất định của tính toán GPU.
- ✅ **Lưu trữ per-object đầy đủ cho Split T (D60):** Runner `run_final_T.py` tự tạo features, cues và $Z_d$ từ detections, xuất toàn bộ per-object artifacts vào `results/final/{model}_T_predictions.parquet` (gồm $Z_{\text{base}}$, $\hat{Z}_f$, khoảng tin cậy của cả 3 phương án, `fallback_flag`, các cột GT đánh giá và danh sách False Negatives). Tuyệt đối không ghi đè vào `results/datasets/` để bảo đảm tính toàn vẹn và độc lập của dữ liệu nghiệm thu cho T14/T15.
- ✅ **Bổ sung toàn diện cấu hình đóng băng pipeline_frozen_v1.yaml (D61):** File cấu hình đóng băng bao gồm đầy đủ: ngưỡng tin cậy `pass_thr` từng detector (D6), `best_params_f`, `best_alpha_f0`, `dontcare_mode`, `seed_quantile`, mã SHA-256 của tất cả các file cấu hình nguồn và model weights/calibrations, kèm file khóa môi trường `configs/requirements_frozen.txt` (`pip freeze`). Toàn bộ model weights và file calibration trên C được track chính thức vào Git repo.
- ✅ **Bảo vệ tính cô lập cho Thí nghiệm Độ nhạy T09 (D62):** Thí nghiệm T09.1 bổ sung bài kiểm tra đối chứng bỏ `drive_0104` (bên cạnh bỏ `drive_0059`) do việc bỏ 0059 đẩy tỷ trọng của 0104 lên $\approx 35.7\% > 35\%$ ($n_{\text{eff}} = 4.14$). Toàn bộ kết quả T09 được lưu vào thư mục riêng biệt `results/sensitivity/`, tuyệt đối không ghi đè vào `runs/residual/` hay các file OOF của Split B.
- ✅ **Kiểm tra toàn diện SHA của models trong Guard 3 (D63):** `check_guard_3_hashes` trong `scripts/run_final_T.py` bắt buộc duyệt và kiểm tra mã băm SHA-256 của toàn bộ các file mô hình trong `det_info["models"]` (`model_f`, `model_f0`, `model_e`, `full_fw`, `model_q05`, `model_q95`) đối chiếu với `configs/pipeline_frozen_v1.yaml`. Điều này ngăn chặn triệt để rủi ro sửa đổi file mô hình sau khi gắn tag `final-config-v1`, khắc phục lỗ hổng do Guard 2 bỏ qua thư mục runtime `runs/`.
- ✅ **Bản chất của Dry-run trên Split C là Regression Check (D64):** Kết quả độ phủ $\approx 90.0\%$ trên Split C trong chế độ dry-run là in-sample (do $\hat{Q}$ được hiệu chuẩn trên chính C). Kết quả này chỉ phục vụ mục đích kiểm chứng hồi quy kỹ thuật (regression check) đối soát dung sai với Golden File (D59), tuyệt đối không được trích dẫn hay diễn giải như độ phủ ngoài mẫu (out-of-sample coverage) của pipeline trong báo cáo/bài báo.
- ✅ **Ghi nhận nhật ký T12 qua final_T_log.jsonl và hỗ trợ đầy đủ cột per-object cho T13 (D65):** Runner `scripts/run_final_T.py` bắt buộc tự động ghi nhận nhật ký vào `runs/final_T_log.jsonl` (sự kiện START sau khi tạo lock và COMPLETED sau khi kết thúc với metrics, commit, hashes) theo AGENT_RULES §5 và checklist T12. Bảng per-object `results/final/{model}_T_predictions.parquet` phải chứa đầy đủ dự đoán điểm của cả Model (e) (`z_hat_e`) và Model (f0) (`z_hat_f0`, `r_hat_f0`) cho toàn bộ các mẫu (thay vì chỉ tính (e) khi có fallback), bảo đảm cung cấp đủ dữ liệu per-object cho bài toán so sánh ablation T13 mà không phá vỡ nguyên tắc chạy Split T một lần duy nhất. Tác vụ T09 (độ nhạy phụ trợ) được dời sang W3-2 theo cơ chế D39.
- ✅ **Test an toàn zero-touch cho Split T (D66):** Test `test_run_inference_core_allow_test_passes_load_split` trong `tests/test_permission_guard_t.py` bắt buộc dùng Sentinel Exception ngắt thực thi ngay trong `fake_load_split`. Bản nháp test trước đây gọi qua `load_split` thật cho phép T dù mock detector, dẫn tới việc đọc nhãn và header ảnh của 1102 frames Split T (không thực hiện suy luận, không fit mô hình, không rò rỉ log). Thay thế hoàn toàn bằng sentinel test độc lập, cam kết nguyên tắc "Split T tuyệt đối không bị chạm vào trước T12" (AGENT_RULES §1.1).
- ✅ **Chuẩn hóa cặp sự kiện final_T_log, Preflight Guard và tách Mondrian khỏi dữ liệu T (D67):**
  - Nhật ký `runs/final_T_log.jsonl` ghi đúng một cặp sự kiện `START` và `COMPLETED` (hoặc `FAILED` nếu gặp sự cố hạ tầng/OOM), sửa đổi tài liệu T12 và T18 từ "đúng một dòng" thành "đúng một cặp sự kiện" để tránh sai lệch audit. Bọc vòng lặp thực thi Split T trong khối `try...except` ghi nhận sự kiện `FAILED` kèm traceback đầy đủ trước khi re-raise.
  - Bổ sung kiểm tra Preflight (`check_guard_preflight`) trước khi tạo file lock `runs/final_T.lock`: kiểm tra quyền ghi thư mục output, kiểm tra dung lượng đĩa trống $\ge 1\text{ GB}$, và kiểm tra sự hiện diện của GPU CUDA.
  - Cố định khởi tạo `MondrianBinning` trong runner đọc trực tiếp từ `conformal_calib_C` (`bin_edges`, `bin_labels`, `n_bins`), loại bỏ hoàn toàn phụ thuộc vào tham số `z_hat_f` của Split T trong constructor.
  - Ghi chú: `configs/requirements_frozen.txt` là bản chụp pip freeze của môi trường toàn cục (bao gồm cả thư viện agent phụ trợ), không phải môi trường runtime tối thiểu của pipeline.
- ✅ **Tiền đăng ký quy tắc diễn giải kết quả Split T (D68):** Toàn bộ kết quả thực nghiệm trên Split T sau khi chạy T12 vào ngày 10/10 phải được báo cáo trung thực và nguyên vẹn:
  - Tuyệt đối cấm refit mô hình, cấm chỉnh sửa $\alpha = 0.1$, cấm điều chỉnh ngưỡng hay binning sau khi quan sát dữ liệu T.
  - Dựa trên phát hiện thực nghiệm tại T08 (độ phủ held-out drive thực tế dao động 83.8%–87.4% trên $B \cup C$ do chỉ có 10 cụm drive), độ phủ thực tế trên Split T rất có thể sẽ dưới mức danh nghĩa 90%. Đây là phát hiện khoa học trung thực và khách quan cho RQ3/H3 về giới hạn conformal trong điều kiện cụm nhỏ hữu hạn, không được xem là lỗi và không được tìm cách che giấu hay điều chỉnh tham số.
  - Không xếp hạng hơn kém giữa các detector nếu khoảng tin cậy Bootstrap (10 cụm) chồng lấn hoặc chứa 0. Mọi độ phủ báo cáo đều là độ phủ có điều kiện trên tập True Positives vượt ngưỡng tin cậy.
- ✅ **Ghi nhận số liệu mốc gốc Split T từ T12 (D69):** Chạy nghiệm thu T12 hoàn thành thành công trong 147.8s (tag `final-config-v1`, commit `e3ead56`). Split T gồm 1,102 frames, 10 drive có xe, tổng số 3,212 Ground Truth Car objects (khớp 100% giữa 3 detector). Số lượng True Positives vượt ngưỡng tin cậy D6 và False Negatives:
  - `yolo11s_640` (conf 0.70): $n_{\text{TP}} = 2,712$, $n_{\text{FN}} = 500$ (Recall 84.43%), $n_{\text{fallback}} = 36$ (1.33%). SHA parquet: `4119e4b636c6d685...`
  - `yolov8s_640` (conf 0.79): $n_{\text{TP}} = 2,660$, $n_{\text{FN}} = 552$ (Recall 82.81%), $n_{\text{fallback}} = 36$ (1.35%). SHA parquet: `8105f1d843050c37...`
  - `yolov5su_640` (conf 0.74): $n_{\text{TP}} = 2,674$, $n_{\text{FN}} = 538$ (Recall 83.25%), $n_{\text{fallback}} = 36$ (1.35%). SHA parquet: `be5cc9e2354f98ee...`
  Khóa vĩnh viễn `runs/final_T.lock` đã được tạo thành công. Toàn bộ số liệu trên là mốc gốc bất biến tuyệt đối.
- ✅ **Phân định phạm vi tác vụ T13 và các tác vụ hậu T12 (D70):** Tất cả các phân tích từ T13 đến T18 chỉ được phép đọc các tệp tĩnh đã sinh trong `results/final/*` và `results/tables/final_eval_T.json`. Tuyệt đối cấm code mới nhập `load_split("splits", "T")` hoặc gọi lại suy luận trên T dưới mọi hình thức, tuân thủ nguyên tắc AGENT_RULES §1.1. Mọi bảng kết quả so sánh đều phải kèm cảnh báo hiệu ứng lựa chọn mẫu (survivorship bias) qua việc công bố song song $n_{\text{GT}} = \text{TP} + \text{FN}$.
- ✅ **Khắc phục lỗi triển khai serialization base_score và chuẩn hóa hậu T12 (D71):**
  * **Phát hiện**: Quá trình kiểm toán độc lập sau T12 phát hiện lỗi cú pháp trong các tệp serialize `model_f.json` và `model_e.json` (tồn tại từ commit `76f4fd8`): trường `learner.learner_model_param.base_score` bị bao trong chuỗi mảng `"[val]"` thay vì chuỗi số vô hướng. Parser C++ của XGBoost khi nạp gặp lỗi parse cú pháp nên silently fallback về giá trị mặc định của thư viện là `0.5`, gây độ lệch hằng số $+0.5$ trong log-residuals $\hat{r}$, đẩy AbsRel của Model (f) lên ~0.62 và Model (e) lên ~0.92, đồng thời làm méo mó độ rộng Split Conformal ($3.37\times$) và Mondrian CQR ($34\times$ ở dải 0–10m).
  * **Quy trình chuẩn hóa bảo vệ Split T**:
    1. Giữ nguyên 100% lockfile `runs/final_T.lock`, không chạy lại detector inference trên Split T, không nạp lại raw data Split T (tuân thủ D27, D70). Toàn bộ detections, matches, và ground truth trên Split T được bảo toàn bất biến trong `results/final/*.parquet`.
    2. Chuẩn hóa chuỗi `base_score` trong 6 file JSON (xóa dấu ngoặc vuông `[]`), khôi phục 100% hành vi nguyên bản của các cây quyết định đã học trên Split B, tuyệt đối không retrain, không tuning.
    3. Hiệu chuẩn lại Split Conformal và Mondrian CQR trên Split C (`scripts/calibrate_conformal_c.py`).
    4. Cập nhật mã băm SHA-256 mới trong `runs/residual/*/manifest.json` và `configs/pipeline_frozen_v1.yaml` (schema version 1.1, chuẩn bị tag `final-config-v1.1`).
    5. Sao lưu dữ liệu dự đoán ban đầu thành `results/final/*_T_predictions_v1_buggy.parquet`.
    6. Thực hiện suy luận hậu kiểm (post-hoc recomputation) trên các artifact tĩnh qua `scripts/recompute_final_T_posthoc.py`.
    7. Cam kết liêm chính học thuật: Báo cáo song song cả kết quả gốc v1 (có lỗi triển khai) và v1.1 (đã chuẩn hóa), giải trình chi tiết trong mục Limitations của bài báo.
- ✅ **Bổ sung Sanity Gate vào Runner Dry-run (D72):**
  * Mở rộng `verify_dryrun_against_golden` trong `scripts/run_final_T.py` với 3 cổng kiểm tra mức độ hợp lý (Sanity Gates):
    1. $\text{AbsRel}(f) \le 0.10$ trên Split C (đảm bảo cùng quy mô với OOF trên Split B).
    2. Độ rộng Split Conformal trung bình nằm trong khoảng $[1.0, 2.0\times]$ (không vượt quá xa CQR).
    3. Độ rộng Mondrian CQR trung bình nằm trong khoảng $[1.0, 2.5\times]$.
  * Cập nhật `runs/dryrun_golden_C.json` với số liệu sạch của Split C ($\text{AbsRel}(f) \approx 0.070\text{--}0.072$, CQR mean width $\approx 1.33\text{--}1.35\times$, Winkler $\approx 0.33\text{--}0.41$).
- ✅ **Chuẩn hóa Phương pháp luận Đánh giá T13 và Diễn giải Thống kê (D73):**
  * **Sửa lỗi nhãn Bootstrap**: Quy ước hiển thị $\Delta = \text{AbsRel}(f) - \text{AbsRel}(d)$, trong đó giá trị âm phản ánh Model (f) cải thiện so với Model (d). Nhãn ghi rõ `diff_absrel (f - d)`. Tương tự, so sánh giữa 2 detector $d_1$ vs $d_2$ ghi rõ `diff_absrel (d1 - d2)`.
  * **Phân tích tương quan RQ2**: Thừa nhận hiện tượng lặp giả (pseudo-replication) khi tính $p$-value tổng gộp (pooled) trên 2,712 quan sát chỉ thuộc 10 cụm drive. Bổ sung khoảng tin cậy Cluster Bootstrap 95% (resample theo drive) cho cả hệ số tương quan Pearson $r$ và Spearman $\rho$. Báo cáo RQ2 ở mức độ mô tả thận trọng, chỉ ra tương quan yếu ($|r| \le 0.15$).
  * **Đánh giá độ phủ CQR**: Báo cáo độ phủ thực nghiệm (96.4%–97.1%) kèm điều kiện chặt chẽ: có điều kiện trên True Positives (Recall 83.2%–84.4%), và phân tầng theo khoảng cách (dải gần 0–10m đạt ~80.5%).
  * **Chuẩn hóa văn phong**: Loại bỏ toàn bộ ngôn ngữ mang tính cảm tính ("rực rỡ / xuất sắc / khẳng định thực nghiệm") theo chuẩn mực D68.



---


## 3. Quyết định về code

- ✅ Loader **trung lập về split**: nhận danh sách frame ID từ bên ngoài, không chứa logic split.
- ✅ Bỏ `DontCare` khi đọc nhãn xe; **thêm `parse_dontcare()`** trả về bbox DontCare riêng (cần cho §5.1 khớp Greedy D16).
- ✅ Hash split = SHA-256 của danh sách ID đã sắp xếp. Metadata ghi `seed`, tên split, `n_frames` và hash vào `split_metadata.json`.
- ✅ Kiểm tra rò rỉ theo drive bằng `assert_split_disjoint_by_drive` và script `scripts/verify_data.py`.
- ✅ Loader cung cấp cả `depth` (Z) và `distance` (Euclid); **dùng `depth` cho mọi tính toán AbsRel/MAE**.
- ✅ **Mapping fix (02/10):** `read_drive_mapping` dùng `train_rand.txt`. Frame `i` → dòng `rand[i]-1` của `train_mapping.txt`. Xác nhận bằng P2 consistency: H0=0.81, H1=1.0000.
- ✅ **3 lớp huấn luyện YOLO:** Car (0), Van (1), Truck (2). Không áp lọc Hard cho nhãn train.
- ✅ **Guard khóa tập T:** Hàm `load_split(splits_dir, 'T', allow_test=False)` trong `src/utils/split_builder.py` chặn load T nếu chưa bật `allow_test=True` hoặc `ALLOW_TEST_SPLIT=1`. Đã có unit test.
- ✅ **Log JSONL:** `runs/detector/train_log.jsonl` tự động ghi seed, split hashes, git commit, thời gian chạy và metrics.
- ✅ **Prior hình học:** `scripts/compute_priors.py` tính W_eff, H_obj, H_cam, y_horizon chỉ từ nhãn Split A, lưu vào `configs/geometry_priors.yaml`.

---

## 4. Việc cần kiểm chứng trên dữ liệu thật

- [x] Tải KITTI Object: `image_2`, `label_2`, `calib`, devkit. Đã xác nhận 7,481 frames, 141 drives.
- [x] Xác minh mapping: `train_rand.txt` kiểm chứng bằng P2 consistency = **1.0000** (141/141 drives).
- [x] Chạy KS test Car-only (Hard-filtered) trên Splits-v2 (Decision D14, Seed 85):
  - **B vs C:** KS stat = **0.0460** ($p = 7.13e-3 \le 0.0700$) → **ĐẠT!**
  - **B vs T:** KS stat = **0.0588** ($p = 3.17e-6 \le 0.0700$) → **ĐẠT!**
  - **C vs T:** KS stat = **0.0556** ($p = 1.41e-3 \le 0.0700$) → **ĐẠT!**
  *(Ghi chú: Các số KS v1 cũ: B-C=0.0527, B-T=0.0723, C-T=0.0391 đã superseded bởi splits-v2)*.
  - Độ sâu Car Hard: B (mean 25.5, median 24.1), C (mean 25.6, median 25.4), T (mean 26.2, median 26.2).
- [x] Đếm mẫu Car-only dải >50 m (Hard) trên Splits-v2: B=19*, C=0*, T=33* (tổng B∪C∪T = 52 xe; B-v1 cũ là 9*, C-v1 là 8*, T-v1 là 35*). Đánh cờ `*` khi $n < 100$ và có hàng gộp >30 m theo D3.
- [x] Phép thử P2 consistency và split disjointness chuyển thành `scripts/verify_data.py` (chạy tự động trước mỗi khâu).
- [x] Viết `tests/test_splits.py` (5 tests pass: regression mapping H0 vs H1, hash metadata, drive disjoint, frame counts, test split guard).

### 📝 Limitations cần nêu trong paper
1. **Lệch phân bố Truck:** Truck tập trung ở một số drive lớn (A=5.2%, T=0.8%), không thể khắc phục hoàn toàn khi split theo drive với 141 drive.
2. **Split theo drive ≠ theo địa điểm:** Một số drive cùng ngày có thể quay ở cùng khu vực địa lý, nhưng split theo drive là chuẩn cao nhất khả thi trên KITTI raw mapping.
3. **Mẫu dải xa (>50 m) thấp:** Car Hard >50 m chỉ có 19 xe ở B, 0 xe ở C và 33 xe ở T (tổng cộng 52 xe trên B∪C∪T). Khoảng tin cậy ở dải này rộng và đánh giá detector trên C chỉ phản ánh cự ly $\le 50$ m.
4. **Tập trung Car Hard theo ít cụm drive (Drive Concentration trên Splits-v2):** Nhiều drive trong KITTI không có xe hơi (quay trong khuôn viên/người đi bộ). Kết quả audit v2 cho thấy:
   - Split A: 35 drives, chỉ 18 drives có Car Hard ($N=11,291$, top 1 drive chiếm 16%).
   - Split B: 34 drives, 12 drives có Car Hard ($N=4,776$); top 1 là `drive_0104` chiếm 26.8% (1.282 xe), top 2 là `drive_0059` chiếm 24.8% (1.185 xe); hai drive này chiếm 51.65% Car Hard của B ($n_{\text{eff}} = 5.29$).
   - Split C: 18 drives, 10 drives có Car Hard ($N=1,826$, top 1 là `drive_0096` chiếm 28.7%, $n_{\text{eff}} = 5.28$; giảm mạnh từ 50.8% ở v1).
   - Split T: 29 drives, 10 drives có Car Hard ($N=3,212$, top 1 là `drive_0095` chiếm 23.3%, $n_{\text{eff}} = 5.28$).
   - Split V: 25 drives, chỉ 6 drives có Car Hard ($N=611$, top 1 drive chiếm **41%**!).
   *Hệ quả:* Cluster bootstrap trên T có 10 cụm độc lập (khoảng tin cậy thô, cần cảnh báo khi $<20$ cụm theo quy ước `eval.py`), grouped CV trên B có 12 fold. Đây là căn cứ khoa học vững chắc để chốt **Quyết định D12** (chọn checkpoint detector `last.pt` epoch 100 thay vì dựa vào đỉnh fitness dễ ăn may trên 6 cụm của V).
5. **Nguồn gốc Checkpoint Detector trong train_log.jsonl:** Ba dòng huấn luyện lại D7 (`ddd519a`, `058158d`) có `git_dirty: true` do untracked files lúc chạy nền. Nguồn gốc và tính toàn vẹn của checkpoint được bảo chứng độc lập và bất biến bằng mã băm SHA-256 đối chiếu khớp 100% trong `configs/detector/checkpoints.yaml` và `runs/inference_log.jsonl`.
6. **Neighbor classes trong đối sánh KITTI:** Module matching chỉ coi Car non-Hard và DontCare là đối tượng bỏ qua (ignore); Van và Truck không được coi là neighbor class như devkit chính thức của KITTI. Do đó, một số bbox dự đoán Car trùng với GT Van/Truck bị tính là FP, khiến Precision của detector trong báo cáo có thể thấp hơn thực tế một chút; điều này hoàn toàn không ảnh hưởng đến ước lượng khoảng cách Z vì ranging chỉ thực hiện trên True Positives (TP).
7. **Chuyển dịch Drive D14 (B-v1 sang T-v2 và C-v2):** Tổng cộng 10 drive từ B-v1 sang T-v2 (gồm 4 drive có Car Hard: `0926_0002`, `0926_0017`, `0926_0039`, `0926_0095` và 6 drive rỗng: `0928_0134`, `0928_0136`, `0928_0162`, `0928_0167`, `0928_0168`, `0928_0201`) cùng 3 drive từ B-v1 sang C-v2 (gồm `0926_0096` và 2 drive rỗng: `0928_0153`, `0928_0161`) từng nằm trong B-v1 khi phân tích Day 4 (hình thành quy tắc mask viền ảnh). Đây là ảnh hưởng gián tiếp ở mức thiết kế hình học định tính, không phải tối ưu hóa số học.
   *Hệ quả:* Cluster bootstrap trên T chỉ có $\le 11$ cụm độc lập (khoảng tin cậy thô, cần cảnh báo khi $<20$ cụm theo quy ước `eval.py`), grouped CV trên B tối đa 10 fold, và việc chia lại B∪C xoay quanh ~21 cụm. Đây là căn cứ khoa học vững chắc để chốt **Quyết định D12** (chọn checkpoint detector `last.pt` epoch 100 thay vì dựa vào đỉnh fitness dễ ăn may trên 6 cụm của V).
8. **Hiện tượng (f) ≈ (e) và vai trò của hình học (T04, T05):** Trên cả 3 detector, sai số của mô hình residual (f) và mô hình hồi quy trực tiếp (e) từ bbox xấp xỉ tương đương nhau (Pooled AbsRel: 0.0462 vs 0.0462 trên yolo11s; 0.0467 vs 0.0468 trên v8s; 0.0484 vs 0.0482 trên v5su; CI của f − e chứa 0). Thí nghiệm ablation (T05) xác nhận bỏ toàn bộ $\ln z_*$ chỉ làm sai số tăng $\Delta \le +0.0003$. Nguyên nhân có thể do $Z_h$ và $Z_g$ là các biến đổi đơn điệu của $h$ và $y_{\text{bottom}} - c_y$, mà cây quyết định tự học được tương đương. Do đó, bài báo **không khẳng định hình học là bắt buộc để đạt độ chính xác cao nhất**, mà định vị giá trị cốt lõi của hình học ở 3 khía cạnh: (1) cung cấp nền vật lý có thể diễn giải (d); (2) làm điểm neo để phân rã tách bạch sai số hình học thuần túy so với sai số do jitter detector ($\Delta \text{AbsRel} \approx 0$); và (3) đóng vai trò fallback an toàn khi mất dữ liệu.

---

## 5. Tiến độ (theo §8 của kế hoạch)

### Chuẩn bị (§8.0)
- [x] Tạo project và repo GitHub `Distance-Estimation-KITTI-YOLO`
- [ ] Gửi 4 câu hỏi cho thầy
- [x] Đóng băng split và tag git `splits-v1`
- [x] Tạo script kiểm định dữ liệu `scripts/verify_data.py`
- [x] Tạo bộ test `tests/test_splits.py`
- [ ] Cài PyTorch hỗ trợ RTX 5060 (CUDA 12.8 wheel)
- [ ] Chạy thử 1 epoch fine-tune, ghi thời gian/epoch và VRAM: ______
- [x] Dựng thư mục `data/ splits/ configs/ runs/ results/ notebooks/ src/ tests/ docs/ scripts/`
- [x] Tạo log thí nghiệm JSONL (`runs/detector/train_log.jsonl`)

### Tuần 1
- [x] **Ngày 1:** loader viết xong, mapping rand sửa xong, P2 consistency = 1.0000.
- [x] **Ngày 2:** split A/V/B/C/T theo drive đã tạo, KS Car-only đạt D1, đóng băng `splits-v1`.
- [x] **Ngày 2 (tiếp):** chuyển A/V sang định dạng YOLO (`scripts/make_yolo_dataset.py` xong: Train A=3,740 frames, Val V=374 frames, `dataset.yaml`, `SPLIT_HASH.json`).
- [x] **Ngày 3 (sớm):** tính prior W_eff, H_obj, H_cam, y_horizon từ nhãn A (`scripts/compute_priors.py`, `configs/geometry_priors.yaml`).
- [x] **Ngày 3:** Chốt `imgsz = 640` (D5), chạy hoàn tất hàng đợi fine-tune YOLOv8s → YOLO11s → YOLOv5su, chốt ngưỡng conf theo F1 max Car trên V (D6).
- [x] **Ngày 4:** cài 3 cue (a)(b)(c) + hợp nhất log-space (d), kiểm thử đơn vị pass 23/23, đánh giá trên Split B GT bbox vượt điều kiện Tuần 2 (trên B-v2: 3/4 dải độc lập n>=100 pass, AbsRel 0.0605 in-sample, 0.0610 OOF; con số cũ 4/5 dải, AbsRel 0.0635 trên B-v1 đã superseded).
- [x] **Ngày 5:** `eval.py` + test đơn vị (`tests/test_eval.py` 19 tests pass, `tests/test_geometry.py` 27 tests pass, `tests/test_matching.py` pass).
- [x] **Ngày 6:** chạy (a)–(d) trên GT bbox Split B-v2, phân tích chuyên sâu LODO OOF (12 fold), phân rã góc nhìn $\theta$ (chuẩn KITTI), sign count 9/12 drive ($p=0.0730$ tính bằng `scipy.stats.binomtest`), paired cluster bootstrap (`scripts/eval_day6_analysis.py`).
- [x] **Ngày 7:** đệm & hoàn tất tái phân bổ B/C/T (D14, `splits-v2`, `geometry-v2`), chốt D15 (matching 4 trạng thái), D16 (Greedy matching), D22 (`run_inference.py` FP32 `conf=0.05` parquet pipeline).

### Tuần 2
- [x] **Ngày 1 (W2-1):** Chạy suy luận detector trên Split B (1.499 frames) và Split C (766 frames) cho cả 3 detector (`yolov8s`, `yolo11s`, `yolov5su`). Hoàn tất xuất 18 parquet files và báo cáo hiệu năng P/R/F1/Common Support (`detector_eval_b_c.md`).
- [x] **Ngày 2 (W2-2, T00–T03):** Hoàn thành chuẩn bị dữ liệu `build_dataset` (T01), đánh giá hình học trên bbox detector kèm refit trọng số và phân rã sai số (T02, D17/D29/D31), chẩn đoán độ lệch phân bố bbox A vs B vs C (T03, D32).
- [x] **Ngày 3 (W2-3, T04):** Residual (f0), (f), (e) + pre-register (`prereg-residual-v1`), nested LODO 12 fold trên Split B, Gate T04 vượt chuẩn (AbsRel giảm từ 0.059–0.061 xuống 0.046–0.048 ở cả 3 detector).
- [x] **Ngày 4 (W2-4, T05, T06):** Ablation 10 cấu hình trên Split B OOF (T05); Latency Tier 1 benchmark trên GPU FP16 và CPU ORT FP32 (T06, Preliminary D44).
- [x] **Sửa lỗi giao thức & Refit Full B (06/10, D43):** Refit toàn bộ model cuối trên nền $Z_{\text{base}}$ OOF từ `B_oof.parquet`, serialize `full_fw.json` và cập nhật manifest cho cả 3 detector.
- [x] **Ngày 5 (W2-5, T07):** Conformal Quantile Regression (CQR) trên r, conformalize trên Split C (D46–D49).
- [x] **Ngày 6 (W2-6, T08):** Đánh giá độ ổn định độ phủ qua 20 resplits trên held-out drives ($B \cup C$), Mondrian CQR và Split Conformal đối chứng (D50–D54).
- [x] **Ngày 7 (W2-7, T10, T11):** Đóng băng cấu hình toàn diện, runner nghiệm thu Split T với 4 guard, dry-run C đạt dung sai (D55–D65).

---

## 6. Nhật ký theo phiên

### W3-1 — 10/10/2026: T12 Nghiệm Thu Split T Thành Công và Khóa Bất Biến Kết Quả (D69–D70)
- **Thực thi chính thức T12 (Human-in-the-loop, D27, D69):**
  - Người nghiên cứu trực tiếp chạy `python scripts/run_final_T.py --confirm FINAL_T_RUN` trên terminal.
  - Vượt qua 4 Safety Guards và Preflight: Guard 1 (tag `final-config-v1` trùng HEAD commit `e3ead56`), Guard 2 (cây sạch), Guard 3 (toàn bộ SHA khớp 100%), Preflight Check (GPU RTX 5060, đĩa trống 419.6 GB, thư mục ghi được), Guard 4 (tạo atomic lock `runs/final_T.lock`).
  - Hoàn tất suy luận trên 1,102 frames của Split T trong 147.8s (21.0 - 25.4 FPS). Ghi nhận cặp sự kiện `START` và `COMPLETED` (`status: "SUCCESS"`) vào `runs/final_T_log.jsonl`.
  - Toàn bộ 15 tệp artifacts (detections, matches, gt, predictions, fn) được xuất an toàn vào `results/final/`. Tệp tóm tắt `results/tables/final_eval_T.json` được tạo lập.
- **Số liệu mốc gốc bất biến trên Split T (D69):**
  - Tổng số đối tượng Ground Truth Car Hard: **3,212** xe trên 10 drive có xe (khớp tuyệt đối giữa 3 detector).
  - True Positives và False Negatives:
    + `yolo11s_640` (conf 0.70): $n_{\text{TP}} = 2,712$, $n_{\text{FN}} = 500$ (Recall 84.43%), $n_{\text{fallback}} = 36$ (1.33%). CQR Pooled Coverage: **95.58%**, Macro: **92.29%**, Mean Width Ratio: 1.327, Mean Winkler: 0.8607.
    + `yolov8s_640` (conf 0.79): $n_{\text{TP}} = 2,660$, $n_{\text{FN}} = 552$ (Recall 82.81%), $n_{\text{fallback}} = 36$ (1.35%). CQR Pooled Coverage: **96.32%**, Macro: **96.41%**, Mean Width Ratio: 1.362, Mean Winkler: 0.8868.
    + `yolov5su_640` (conf 0.74): $n_{\text{TP}} = 2,674$, $n_{\text{FN}} = 538$ (Recall 83.25%), $n_{\text{fallback}} = 36$ (1.35%). CQR Pooled Coverage: **95.74%**, Macro: **97.57%**, Mean Width Ratio: 1.341, Mean Winkler: 0.8807.
  - Cả 3 detector đều đạt độ phủ danh nghĩa $\ge 90\%$ trên Split T với $n_{\text{crossings}} = 0$.
- **Khởi động T13 (D70):** Toàn bộ phân tích bảng chính, đối sánh phương án (a)–(g), paired cluster bootstrap và tương quan RQ2 được thực thi tự động từ `results/final/*` mà không chạm vào dữ liệu thô Split T.

### W2-7 — 09/10/2026: T10 & T11 Đóng Băng Cấu Hình Pipeline Toàn Diện, Runner Nghiệm Thu Split T và Chuẩn Bị Tag final-config-v1 (D55–D68)
- **Chuỗi tác vụ đường găng đóng Tuần 2 (T10, T11):**
  - **Khâu 1 — Chụp Golden File trên Split C (D59):** Viết `scripts/capture_golden_c.py`, chụp baseline trên code hiện hành lưu vào `runs/dryrun_golden_C.json`.
  - **Khâu 2 — Refactor Additive `allow_test` và vá lỗi `load_split`:**
    - Tách `run_inference_core(..., allow_test=False)` trong `scripts/run_inference.py`, giữ nguyên chữ ký công khai `run_inference_for_model` ném `PermissionError` với T.
    - Sửa lỗi truyền `allow_test=allow_test` vào `load_split` để ngăn chặn sập pipeline sau khi tạo lock trên Split T.
    - Bổ sung `allow_test=False` cho `load_artifacts` (`build_dataset.py`) và `apply_frozen_pipeline` (`apply_frozen.py`).
    - Bảo đảm an toàn zero-touch cho Split T (D66): test `test_run_inference_core_allow_test_passes_load_split` sử dụng Sentinel Exception dừng ngay trong `fake_load_split`, bảo đảm không đọc frame, nhãn hay ảnh của Split T trong test suite.
  - **Khâu 3 — Calibrate 3 Phương án trên toàn Split C (D56):**
    - Viết `scripts/calibrate_conformal_c.py`, lưu `conformal_calib_C.json` cho cả 3 detector gồm Standard CQR ($\hat{Q}$), Split Conformal ($\hat{Q}$) và Mondrian CQR (4 bin theo `binning.edges`).
  - **Khâu 4 & 5 — Runner Nghiệm Thu Split T và Dry-run C (D27, D59, D60, D63, D64, D65, D67, D68):**
    - Xây dựng `scripts/run_final_T.py` với 4 Safety Guards và Preflight: Guard 1 (tag HEAD), Guard 2 (clean tree), Guard 3 (kiểm tra toàn diện SHA của splits, checkpoints, tất cả trained models và configs theo D63), Preflight Check (đĩa trống $\ge 1\text{ GB}$, quyền ghi thư mục, GPU CUDA theo D67), Guard 4 (atomic lock `runs/final_T.lock` với cờ `FINAL_T_RUN`).
    - Runner hỗ trợ ghi nhật ký `runs/final_T_log.jsonl` (cặp sự kiện START và COMPLETED, bọc `try...except` ghi FAILED kèm traceback theo D67) và tính toán đầy đủ các trường per-object (`z_hat_e`, `z_hat_f0`, `r_hat_f0`) lưu vào `results/final/{model}_T_predictions.parquet` phục vụ T13.
    - Khởi tạo `MondrianBinning` trực tiếp từ `conformal_calib_C.json`, không phụ thuộc vào dữ liệu Split T (D67).
    - Tiền đăng ký quy tắc diễn giải kết quả Split T (D68): báo cáo trung thực, không refit, không đổi $\alpha$, chấp nhận khả năng under-coverage trên held-out drive theo phát hiện thực nghiệm T08/RQ3.
    - Chạy `python scripts/run_final_T.py --dry-run C`: kích hoạt Guard 3 và đạt 100% PASS khớp Golden File trên cả 3 detector trong dung sai nghiêm ngặt ($|\Delta \hat{Q}| = 0.00$, $|\Delta \text{Cov}| = 0.000\%$, exact TP count). Ghi nhận dry-run là regression check (D64).
  - **Khâu 6 — Đóng băng Cấu hình và Môi trường (D61):**
    - Hoàn tất `configs/pipeline_frozen_v1.yaml` với đầy đủ mã băm SHA-256 của splits, checkpoints, toàn bộ mô hình và configs; đồng bộ tham số `iou_match: 0.5` (D15) và `dontcare_mode: "iou"` (D8); chú thích rõ `seed_quantile: 42 (q05=42, q95=43)`.
    - Tạo `configs/requirements_frozen.txt` (176 packages, UTF-8, môi trường toàn cục).
    - Mở rộng `tests/test_permission_guard_t.py` lên 8 tests (kiểm tra PermissionError, `load_split` bypass khi allow_test bằng sentinel, tamper detection Guard 3, Guard 4 lock). Test suite đạt **173 passed, 1 warning (MLP)**.
    - Chạy `python scripts/verify_data.py`: **ALL DATA VERIFICATIONS PASSED**.
    - Xác nhận `runs/final_T.lock` chưa tồn tại, Split T hoàn toàn nguyên vẹn (zero touch).
    - Tác vụ T09 (độ nhạy phụ trợ) chính thức dời sang W3-2 theo cơ chế D39 để bảo đảm an toàn đường găng đóng băng trước 17:30.

### W2-6 — 08/10/2026: T08 Đánh giá Độ ổn định Độ phủ Conformal qua 20 Resplits trên Held-out Drives (D50–D54)
- **Pha A — Tiền đăng ký (Pre-registration, commit `e1f9e5b`, tag `prereg-coverage-v1`):**
  - Tạo `configs/residual/coverage_prereg_v1.yaml` đóng băng đầy đủ: tỷ lệ phân hoạch $B \cup C$ (fit 50% / calib 25% / eval 25%), ràng buộc mỗi phân hoạch $\ge 4$ cụm và $\text{top1\_share} \le 0.50$, 20 seed (0–19), 3 phương án đối sánh (Split Conformal, Standard CQR, Mondrian CQR), 4 khoảng Mondrian theo $\hat{Z}$ ($[0, 10), [10, 20), [20, 30), [30, \infty)$) kèm cơ chế fallback gộp bin khi $n_{\text{bin}} < 50$.
- **Pha B — Thực thi 20 Resplits và Đánh giá Thực nghiệm (D50–D54):**
  - Xây dựng module `src/uncertainty/resplit.py` sinh phân hoạch ngẫu nhiên thỏa mãn đầy đủ các ràng buộc cụm.
  - Mở rộng module `src/uncertainty/cqr.py`: `MondrianBinning`, `conformalize_mondrian`, `predict_interval_mondrian`, `compute_split_conformal_scores`, `predict_split_interval`.
  - Viết `tests/test_resplit.py` (6 tests) và bổ sung 4 test cases trong `tests/test_cqr.py` (kiểm tra bất biến phân vị đối với nhãn $Z_{\text{gt}}$). Toàn bộ 165 tests pass 100%.
  - Tạo script runner `scripts/run_coverage_stability.py` chạy toàn diện 20 seeds $\times$ 3 phương án $\times$ 3 detector.
  - **Phát hiện chính — Under-coverage hệ thống trên held-out drives (D53):**
    - Pooled coverage trên cả 3 detector đều thấp hơn mức danh nghĩa 90%: yolo11s (85.22%–87.41%), yolov8s (84.45%–86.68%), yolov5su (83.82%–86.29%).
    - Macro coverage theo drive tụt xuống 77.94%–84.49% (thấp hơn pooled 3–7 điểm %), cho thấy các drive nhỏ bị under-cover nghiêm trọng hơn. Khoảng dao động per-seed rất rộng (63.6% – 99.6%).
    - Đây là kết quả thực nghiệm trung thực cho RQ3/H3: Độ phủ conformal hữu hạn mẫu bị suy giảm khi ngoại suy sang các cụm hoàn toàn chưa quan sát (held-out drives), và resplit 20 lần mang tính bi quan do tập calib chỉ có 5–6 cụm.
  - **Phân tích vị trí cụm gây lệch (Drive-level Heterogeneity, D50):**
    - Hai cụm lớn `0057` và `0004` chi phối trực tiếp độ phủ của seed eval: Khi cả 2 cụm nằm trong eval, coverage tụt xuống **72.38%**; khi 1 cụm trong eval đạt **87.58%**; khi 0 cụm trong eval đạt **94.44%**.
  - **Hiệu quả Mondrian CQR và Không xếp hạng (D52, D54):**
    - Mondrian CQR cải thiện độ phủ ở cự ly gần 0–10m (+6.8%..+9.3%, đạt ~74.9% so với ~68.0% của Standard CQR), nhưng độ phủ tổng thể giữa 3 phương án chênh lệch không đáng kể ($\le 3\%$) và nhỏ hơn nhiều so với độ biến thiên giữa các seed ($\sigma \approx 7\text{--}9\%$). Tuân thủ D52: không xếp hạng hơn kém giữa 3 phương án.
    - Tần suất gộp bin Mondrian: 0/20 seed (0%) do $n_{\text{calib}} \ge 1000$ đủ mẫu cho cả 4 bin.
  - Hoàn tất xuất `results/tables/coverage_stability_20resplits.{json,md}`, `results/tables/coverage_conditional_dev.{json,md}`, và cập nhật `runs/pipeline_log.jsonl`.

### W2-5 — 07/10/2026: T07 Conformal Quantile Regression (CQR) lõi trên Split C (D46–D49)
- **T07 — Conformal Quantile Regression (CQR lõi trên đường găng):**
  - Hoàn thiện module `src/uncertainty/cqr.py` (6 hàm lõi) và module suy luận dùng chung `src/pipeline/apply_frozen.py` (D49).
  - Viết `tests/test_cqr.py` gồm 10 test cases (order statistic hand-crafted, LODO invariance, exchangeable coverage 200 trials, heteroscedasticity, crossing counter, disjoint guard, OOF Z_base invariance, Winkler properties). Toàn bộ test suite đạt **154 passed, 1 warning (MLP)**.
  - Viết `scripts/run_cqr.py` thực thi end-to-end trên cả 3 detector: fit XGBoost quantile ($q=0.05, 0.95$) trên Split B với `best_params_f` cố định (D24), LODO calibrate trên 10 drive Car Hard của Split C, và calibrate toàn bộ Split C xuất `cqr_calib_C.json`.
  - **Số liệu nghiệm thu và Đánh giá Gate T07 (D48):**
    - `yolo11s_640`: Pooled Cov = **0.8717** (Gate [0.85, 0.95]: **PASS**), Macro Cov (10 cụm) = **0.8976**, Macro Cov ($n \ge 30$, 7 cụm) = **0.9014**, Mean Width ($Z_{hi}/Z_{lo}$) = **1.330**, Winkler log-space = **0.5785**, Crossings = **0 (0.0%)**, $\hat{Q}_{\text{full\_C}} = +0.05107$.
    - `yolov8s_640`: Pooled Cov = **0.8713** (Gate [0.85, 0.95]: **PASS**), Macro Cov (10 cụm) = **0.9158**, Macro Cov ($n \ge 30$, 7 cụm) = **0.9001**, Mean Width ($Z_{hi}/Z_{lo}$) = **1.353**, Winkler log-space = **0.6503**, Crossings = **0 (0.0%)**, $\hat{Q}_{\text{full\_C}} = +0.06390$.
    - `yolov5su_640`: Pooled Cov = **0.8797** (Gate [0.85, 0.95]: **PASS**), Macro Cov (10 cụm) = **0.9013**, Macro Cov ($n \ge 30$, 7 cụm) = **0.9004**, Mean Width ($Z_{hi}/Z_{lo}$) = **1.344**, Winkler log-space = **0.6642**, Crossings = **0 (0.0%)**, $\hat{Q}_{\text{full\_C}} = +0.05246$.
  - **Phân tích độ lệch phân phối & Fallback (D47, D48):**
    - $\hat{Q}$ mang dấu dương (+0.051 đến +0.064) và mean $r$ trên C lệch so với OOF B ($\Delta r = -0.008\text{--}-0.014$), giải thích trực tiếp bởi hiện tượng dịch chuyển đáy bbox $\Delta y_2$ phát hiện từ T03 ($KS(B, C) = 0.20\text{--}0.24$, median $\Delta y_2$ đổi dấu).
    - Cụm drive $n \ge 30$ đạt độ phủ macro gần như hoàn hảo 90.0% (90.01%–90.14%), trong khi các drive rất nhỏ (`0079`: 4–7 mẫu, `0047`: 4–7 mẫu, `0027`: 24–25 mẫu) tạo độ nhấp nhô cục bộ.
    - Nhóm fallback (pattern 000) chỉ có 7–10 mẫu ($0.5\%\text{--}0.7\%$), báo cáo mô tả không suy diễn (D48).
    - Không có bất kỳ trường hợp quantile crossing nào ($0/4364$ ca).
  - Hoàn tất xuất `results/tables/cqr_coverage_dev.{json,md}`, `runs/residual/{model}/model_q05.json`, `model_q95.json`, `cqr_calib_C.json`, cập nhật `manifest.json` và `pipeline_log.jsonl`.


### W2-4 — 06/10/2026: T05 Ablation, T06 Latency Tier 1 (Preliminary D44), và Sửa lỗi Giao thức Refit Full B (D43)
- **T05 — Nghiên cứu Thành phần Mô hình (Ablation trên Split B OOF, D25, D37, D38, D45):**
  - Chạy `scripts/run_ablation_oof.py` đánh giá 10 cấu hình ablation trên cả 3 detector:
    - Nhóm Drop Group (A1–A6): Drop `bbox_geometry`, `edge_flags`, `confidence`, `cues_ln_z`, `validity_flags`, `ln_z_base`.
    - Nhóm Drop Single Cue (C1–C3): Drop $Z_w$, $Z_h$, $Z_g$ (tái hợp nhất hình học và loại đặc trưng tương ứng).
    - Nhóm Alternative Model (M1): MLP (128, 64) un-tuned vs XGBoost.
  - **Số liệu chính và diễn giải thực nghiệm (D45):**
    - **Drop $Z_h$:** Là cue duy nhất khiến sai số tăng rõ rệt ở cả 3 detector ($\Delta \text{Pooled} +0.0070\text{--}+0.0091$, 95% CI thô loại 0: yolo11s $[+0.0024, +0.0119]$, v8s $[+0.0023, +0.0153]$, v5su $[+0.0029, +0.0163]$). Đây là hiệu ứng trực tiếp lên $Z_{\text{base}}$ vì $Z_h$ chiếm ~70% trọng số hợp nhất.
    - **Drop $Z_w$ và Drop $Z_g$:** $\Delta \text{Pooled} \approx 0$ (-0.0007 đến +0.0005, CI chứa 0). Macro AbsRel thậm chí giảm nhẹ khi bỏ $Z_g$ (-0.0018 đến -0.0050). Kết quả này hoàn toàn phù hợp với Quyết định D31 (trọng số $w_w \to 0$ khi refit trên bbox detector).
    - **Drop `cues_ln_z`:** Bỏ toàn bộ $\ln z_*$ chỉ làm $\Delta \text{Pooled} \le +0.0003$ (CI chứa 0), nhất quán với việc hồi quy trực tiếp từ bbox (e) đạt sai số gần tương đương (f).
    - **Drop `validity_flags`:** Không thấy đóng góp đo lường được ($\Delta \le +0.0003$, CI chứa 0), dư thừa theo cấu trúc vì khi cue invalid thì $\ln z_k = 0$ và đã có cờ `touch_*` đi kèm.
    - **Drop `ln_z_base` (D30):** Đóng góp nhỏ ($\Delta \le 0.0010$); chỉ tách được ở 1/3 detector (`yolo11s_640`: +0.0010 $[+0.0006, +0.0016]$), 2 detector còn lại CI chứa 0.
    - **Drop `bbox_geometry` và `confidence`:** 95% CI thô chứa 0 ở cả 3 detector.
    - **MLP un-tuned vs XGBoost:** Pooled AbsRel của MLP kém hơn +0.0014..+0.0026 (CI chứa 0), nhưng Macro AbsRel của MLP lại thấp hơn ở 2/3 detector (yolo11s: -0.0020, yolov8s: -0.0037). Hai mô hình không phân biệt được sự khác biệt có ý nghĩa thống kê (D32).
  - Hoàn tất xuất `results/tables/ablation_oof_b.json` và `results/tables/ablation_oof_b.md`.
- **T06 — Đo độ trễ Latency Tier 1 (v4 §5.6, D40, D44):**
  - Chạy `scripts/bench_latency.py` đo 200 ảnh Split B trên GPU NVIDIA RTX 5060 Laptop (PyTorch FP16) và CPU (ONNX Runtime FP32, 4 luồng).
  - Gắn nhãn `PRELIMINARY` (D44) do các hạn chế phương pháp luận: double-count tiền xử lý ở GPU line, tổng tính theo sum of medians, đường tắt residual, và khác biệt dynamic padding của PyTorch vs static padding của ONNX.
  - Parity check: Count Parity đạt chuẩn (0.974–1.009, trong [0.95, 1.05]), nhưng IoU match rate ($\ge 0.90$) đạt 0.904–0.944 (chưa đạt ngưỡng 0.95). Báo cáo trung thực cả 2 chỉ số theo AGENT_RULES §1.9.
  - Hoàn thiện và đo lại chính thức ở tác vụ T16 cùng khâu CQR.
- **Sửa lỗi giao thức và Refit Model cuối (D43):**
  - Khắc phục lỗi protocol ở `fit_full_b_models`: trước đây $Z_{\text{base}}$ dùng để fit (f)/(f0) toàn B được tính in-sample, vi phạm D16b và D29.
  - Sửa `fit_full_b_models` nhận $Z_{\text{base}}$ OOF từ `B_oof.parquet` làm nền target $r$ và đặc trưng dẫn xuất $\ln z_{\text{base}}$.
  - Bổ sung serialization cho `FusionWeights` (`save_fusion_weights`, `load_fusion_weights`).
  - Viết `scripts/refit_full_b.py`, refit thành công mô hình cuối trên cả 3 detector, lưu `full_fw.json`, cập nhật `manifest.json` đầy đủ mã SHA-256 và ghi log `runs/pipeline_log.jsonl`.
  - Bộ test suite tăng lên 144 tests pass 100%.

### W2-3 — 05/10/2026: T04 Mô hình Residual OOF trên Split B, Pre-registration prereg-residual-v1 (D13, D16b, D24, D25, D28–D30, D33, D34, D41, D42)
- **Pha A — Pre-registration (D25, D33):**
  - Tạo `configs/residual/residual_prereg_v1.yaml` đóng băng không gian tìm kiếm XGBoost 12 cấu hình (`max_depth <= 4`, `min_child_weight >= 10`, `n_estimators = 200`, `lr = 0.05`), tập đặc trưng chuẩn hóa (f: 17, f0: 5, e: 10), loại `class_id` và `fallback_flag`. Đánh dấu superseded cho `residual_config.yaml`. Gắn tag git `prereg-residual-v1`.
- **Pha B — Thực thi Nested LODO và Đánh giá Gate T04:**
  - Cài đặt `src/residual/models.py`: `fit_f0` (Ridge), `fit_f` (XGBoost), `fit_e` (direct log depth).
  - Cài đặt `src/pipeline/oof.py`: `run_nested_lodo_b` thực hiện nested Leave-One-Drive-Out CV qua 12 drive của Split B. Ở mỗi fold: Z_d tính theo trọng số LODO $w_{-d}$; Z_e cho pattern 000 trong train dùng inner-LODO 11 fold (D34); Z_base = Z_d nếu $\ge 1$ cue hợp lệ, fallback Z_e nếu pattern 000. Target $r = \ln Z_{\text{gt}} - \ln Z_{\text{base}}$.
  - Chạy trên `yolo11s_640` trước (D36), sau đó chạy `yolov8s_640` và `yolov5su_640` (D35). Xuất 3 file `*_B_oof.parquet` tại `results/datasets/`.
- **Số liệu chính & Đánh giá Gate T04:**
  - `yolo11s_640`: Baseline (d) Pooled AbsRel = 0.0607, Macro = 0.0759 $\to$ (f) Pooled = **0.0462**, Macro = **0.0541** ($\Delta = -0.0144$, CI $[-0.0205, -0.0090]$, sign test 12 thắng / 0 thua, $p = 0.0002$). Gate T04 **ĐẠT**!
  - `yolov8s_640`: Baseline (d) Pooled = 0.0592, Macro = 0.0674 $\to$ (f) Pooled = **0.0467**, Macro = **0.0516** ($\Delta = -0.0125$, CI $[-0.0208, -0.0042]$, sign test 11 thắng / 1 thua, $p = 0.0032$). Gate T04 **ĐẠT**!
  - `yolov5su_640`: Baseline (d) Pooled = 0.0613, Macro = 0.0742 $\to$ (f) Pooled = **0.0484**, Macro = **0.0519** ($\Delta = -0.0129$, CI $[-0.0215, -0.0047]$, sign test 11 thắng / 1 thua, $p = 0.0032$). Gate T04 **ĐẠT**!
  - So sánh với baseline tuyến tính (f0): (f0) đạt Pooled 0.0543–0.0581; $\Delta(f - f0) \approx -0.0076\text{--}-0.0097$ (CI loại trừ 0), xác nhận mô hình phi tuyến XGBoost vượt trội có ý nghĩa so với Ridge tuyến tính.
  - So sánh với direct model (e): (e) đạt Pooled 0.0462 (11s), 0.0468 (v8s), 0.0482 (v5su); $\Delta(f - e) \approx 0.0000$ (CI chứa 0). Mô hình hybrid (f) và mô hình direct (e) đạt độ chính xác tương đương nhau.
  - Canary test (D42): Hoán vị nhãn $Z_{\text{gt}}$ trong train folds khiến RMSE OOF không vượt được $Z_{\text{base}}$ (pass canary guard).
  - Xuất báo cáo `results/tables/residual_oof_b.{json,md}`.

### W2-2 — 04/10/2026: Hoàn tất Chuẩn bị Pipeline (T01), Hình học trên Bbox Detector (T02, D17/D29/D31), và Chẩn đoán Lệch Bbox A vs B vs C (T03, D32)
- **T00 — Dọn Nhật ký, README và Thống kê Đơn vị:**
  - Đồng bộ các quyết định D10 (superseded `geometry-v1`), D12 (`last.pt`), D14 (`splits-v2`, `geometry-v2`).
  - Chuẩn hóa vai trò tập V trong `README.md` (chọn checkpoint và ngưỡng conf, D7: không early stopping).
  - Thêm hàm `sign_test_one_sided(wins, losses)` vào `src/evaluation/eval.py`, loại bỏ giá trị $p$-value hardcode trong Day 6; bổ sung test đơn vị kiểm tra $9/3 \to 299/4096 = 0.0730$.
- **T01 — Xây dựng Dataset Ranging từ Parquet (D11, D22, D23):**
  - Tạo module `src/pipeline/build_dataset.py` và script `scripts/build_datasets.py`, thực hiện phân tách sạch giữa `features` (chỉ thông tin test-time) và `eval` (nhãn GT, matching status), bảo vệ bằng `check_no_gt_leakage` và `FORBIDDEN_FEATURE_SUBSTRINGS`.
  - Khớp tuyệt đối 100% số lượng TP và tổng GT trên Split B và C với `detector_eval_b_c.json` (Split B: v8s 3427, 11s 3523, v5su 3496; Split C: 1445, 1489, 1430). Tạo đủ 18 file parquet và 6 file summary JSON tại `results/datasets/`.
  - Toàn bộ 8 unit tests mới pass; tổng test suite đạt 113/113 passed.
- **T02 — Hình học trên Bbox Detector và Phân rã Sai số (D17, D29, D31, D32):**
  - Tạo `src/pipeline/geometry_stage.py` và script `scripts/eval_geometry_detector.py`.
  - **Kiểm tra hồi quy (Regression check):** Chạy lại trên 4.776 GT bbox của Split B tái lập chính xác AbsRel in-sample = 0.0605 (lệch đúng 0.0000) và LODO OOF = 0.0610 (lệch 0.0001 <= 0.002) so với Day 4 và Day 6.
  - **Hiện tượng kẹp $w_w \to 0$ (D31):** Khi refit trọng số trên bbox detector của Split B, tương quan sai số $(\Sigma_{wh}, \Sigma_{wg})$ đảo dấu từ âm sang dương, làm mất hiệu ứng phân tán rủi ro; NNLS kẹp $w_w \to 0$, dồn trọng số sang $[w_h, w_g] \approx [0.70, 0.30]$. Mô hình refit (d) đạt Pooled AbsRel 0.0592–0.0613, tốt hơn dùng cứng trọng số GT (0.0635–0.0659).
  - **Phân rã sai số detector vs GT bbox:** Paired cluster bootstrap giữa (d)-Detector và (d)-GT bbox trên tập TP chung cho thấy $\Delta \approx 0.0000$ và 95% CI đều chứa 0 (YOLOv8s: $\Delta = -0.0030$, CI $[-0.0114, +0.0060]$; YOLO11s: $\Delta = +0.0000$, CI $[-0.0081, +0.0092]$). Ghi nhận cảnh báo Survivorship bias (D32): kết quả này đo trên tập TP, trong khi ở dải xa 30–50 m detector trượt $\sim 50\%$ đối tượng (Recall $49.7\%\text{--}54.6\%$).
  - Bổ sung hàm `macro_by_cluster` vào `eval.py`. Xuất 6 file parquet cues và bảng báo cáo tại `results/tables/geometry_on_detector_bbox.{json,md}`.
- **T03 — Chẩn đoán Dịch chuyển Phân bố Bbox A vs B vs C (D32):**
  - Hoàn tất suy luận Split A trên GPU FP32 cho 3 detector, tạo đủ 9 file parquet trong `results/predictions/` và ghi nhật ký `runs/inference_log.jsonl` (`git_dirty: false`).
  - Xây dựng `src/evaluation/bbox_diag.py` và `scripts/diag_bbox_shift.py`, xuất báo cáo `results/tables/bbox_shift_A_vs_B.md` và biểu đồ trực quan `results/figures/bbox_shift_A_vs_B.png`.
  - **Kết quả chẩn đoán:** Phân bố bbox trên Split A (seen) chặt hơn rõ rệt so với Split B (unseen) (Recall tụt từ ~95% xuống ~72–74%, trung vị IoU giảm ~0.07, độ tản mạn $\Delta y_2$ tăng hơn gấp đôi, KS test $D > 0.53, p < 10^{-4}$). Trong khi đó, B và C (cùng unseen) có phân bố IoU rất tương đồng ($D_{\text{KS}} \le 0.048$ trên YOLOv8s/YOLO11s).
  - **Kết luận:** Khẳng định tính đúng đắn và bắt buộc của thiết kế phân tách Split A (học detector) và Split B (học residual/conformal) trong Kế hoạch v4 §0.1 và §5.3.

### W2-1 — 03/10/2026 (đêm): Hoàn tất Suy luận B & C cho 3 Detector (D15, D16, D22), Báo cáo Hiệu năng Detector (f156546), Track toàn bộ Parquet và Chuẩn bị D17
- **Hoàn tất toàn bộ suy luận Split B và C cho cả 3 Detector (D22):**
  - Chạy `scripts/run_inference.py` ở chế độ FP32, `imgsz=640`, matching IoU 0.5 (Greedy theo điểm số, D16), `conf_min=0.05`, dùng checkpoint `last.pt` (epoch 100, D12) khớp tuyệt đối mã băm SHA-256 trong `configs/detector/checkpoints.yaml`.
  - Cả 6 lượt suy luận chính thức đều đạt `git_dirty: false`, `splits_version: "v2"`, n_frames B=1.499 (hash `0f83c354...`), C=766 (hash `b24db22e...`):
    - Split B: `yolov8s_640` (22:19), `yolo11s_640` (22:31), `yolov5su_640` (22:33).
    - Split C: `yolov8s_640` (22:22), `yolo11s_640` (22:23), `yolov5su_640` (22:23).
  - Ghi nhận 3 dòng log **superseded** trong `runs/inference_log.jsonl`:
    1. `yolov8s_640` B (`2026-10-03T21:45:12`): chạy khi git dirty, thiếu trường `checkpoint_path`.
    2. `yolo11s_640` B (`2026-10-03T22:20:03`): chạy khi git dirty (thay thế bởi run 22:31:22).
    3. `yolov5su_640` B (`2026-10-03T22:21:09`): chạy khi git dirty (thay thế bởi run 22:33:38).
- **Lưu trữ & Track dữ liệu Parquet:**
  - Toàn bộ 18 file parquet (3 file `detections`, `matches`, `gt` $\times$ 3 model $\times$ 2 split B & C) đã được commit và track chính thức vào git repo (`commit b14a83e`). Working tree hoàn toàn sạch (clean).
- **Báo cáo Hiệu năng Detector trên Split B và C (commit `f156546`, `results/tables/detector_eval_b_c.md` & `.json`):**
  - Ngưỡng tối ưu F1 trên Split V: yolov8s (0.790), yolo11s (0.700), yolov5su (0.740).
  - **Split B (4.776 Car Hard, 12 drives):**
    - `yolov8s`: TP = 3.427, FP = 195, Ignored non-Hard = 500, Ignored DontCare = 19; P = 94.62%, R = 71.75%, F1 = 0.8161.
      - Dải: 0–10m: 284/288 (98.61%), 10–20m: 1.089/1.218 (89.41%), 20–30m: 1.204/1.546 (77.88%), 30–50m: 848/1.705 (49.74%), >50m: 2/19 (10.53%).
    - `yolo11s`: TP = 3.523, FP = 200, Ignored non-Hard = 499, Ignored DontCare = 18; P = 94.63%, R = 73.76%, F1 = 0.8290.
      - Dải: 0–10m: 284/288 (98.61%), 10–20m: 1.118/1.218 (91.79%), 20–30m: 1.238/1.546 (80.08%), 30–50m: 881/1.705 (51.67%), >50m: 2/19 (10.53%).
    - `yolov5su`: TP = 3.496, FP = 204, Ignored non-Hard = 512, Ignored DontCare = 20; P = 94.49%, R = 73.20%, F1 = 0.8249.
      - Dải: 0–10m: 285/288 (98.96%), 10–20m: 1.109/1.218 (91.05%), 20–30m: 1.232/1.546 (79.69%), 30–50m: 868/1.705 (50.91%), >50m: 2/19 (10.53%).
    - Tập chung (Common Support - cả 3 detector cùng phát hiện TP): **3.181 / 4.776 xe (66.60%)**. Hợp (ít nhất 1 detector): **3.772 xe (78.98%)**.
  - **Split C (1.826 Car Hard, 10 drives, toàn bộ $\le 50$ m):**
    - `yolov8s`: TP = 1.445, FP = 137, Ignored non-Hard = 172, Ignored DontCare = 9; P = 91.34%, R = 79.13%, F1 = 0.8480.
    - `yolo11s`: TP = 1.489, FP = 165, Ignored non-Hard = 178, Ignored DontCare = 10; P = 90.02%, R = 81.54%, F1 = 0.8557.
    - `yolov5su`: TP = 1.430, FP = 156, Ignored non-Hard = 182, Ignored DontCare = 10; P = 90.16%, R = 78.31%, F1 = 0.8382.
    - Tập chung (Common Support): **1.359 / 1.826 xe (74.42%)**.
- **Chốt nguyên tắc viết Paper cho D14:** Không sử dụng cụm từ "pre-registered" vì commit git script và split diễn ra đồng thời; trong paper trình bày khách quan: *"Protocol được định nghĩa cố định trong docstring của script từ trước, thuật toán chỉ đọc nhãn (số lượng Car Hard và Z depth) mà không nhìn kết quả mô hình"*.
- **Chuẩn bị Quyết định D17:** Sẵn sàng chạy đánh giá (a)–(d) trên bbox detector với trọng số refit trên Split B theo từng detector, đối chứng với cột trọng số fit trên GT bbox.

### Day 6 — 03/10/2026 (tối): Phân tích Chuyên sâu Hình học GT Bbox trên Split B-v2 (D18–D21, LODO OOF, Bootstrap, Góc nhìn $\theta$)
- **Mục tiêu:** Đánh giá toàn diện mô hình hình học trên Split B-v2 mới ($N=4,776$ Car Hard, 12 drive) bằng `scripts/eval_day6_analysis.py`, lưu báo cáo tại `results/tables/day6_gt_bbox_analysis.json`.
- **Số liệu chính:**
  - **In-sample vs OOF (d):**
    - Pooled AbsRel in-sample = **0.0605**; OOF Centered (LODO 12 fold) = **0.0609**; OOF Uncentered = **0.0610**; $Z_h$ đơn lẻ = **0.0688**. Độ lạc quan in-sample rất nhỏ ($\Delta = +0.0004$).
    - Trọng số trung bình qua 12 fold: Centered $[w_w, w_h, w_g] = [0.0701, 0.6680, 0.2619]$ (rất gần với in-sample $[0.0807, 0.6634, 0.2560]$).
  - **Báo cáo đồng thời Macro và Pooled (D18, D20):**
    - Macro AbsRel không trọng số qua 12 drive: (d) OOF = **0.0646** | $Z_h$ = **0.0628** | $Z_g$ = **0.1602**.
    - Trên các drive có $n \ge 30$ ($k=8$ cụm): (d) OOF = **0.0608** | $Z_h$ = **0.0651**.
    - Tuân thủ D20: Ghi nhận cảnh báo "CI thô ($k=12$ cụm)", không khẳng định "có ý nghĩa thống kê" hay "chứng minh không overfit".
  - **Sign Count qua 12 Drive (vs $Z_h$):**
    - (d) thắng trong **9/12** drive, thua 3, hòa 0.
    - $p$-value một phía: **$p = 0.0730$** (tính chính xác bằng `sign_test_one_sided(wins=9, losses=3)` qua phân phối nhị thức $299/4096$).
  - **Paired Cluster Bootstrap (vs $Z_h$ trên Common Support $n=4,676$):**
    - Ước lượng độ chênh lệch $(d - Z_h)$: **$-0.0095$** (Fused tốt hơn $Z_h$).
    - 95% CI thô (12 cụm): **$[-0.0123, -0.0062]$** (loại trừ 0, `excludes_zero: true`).
  - **Phân tầng Góc quan sát $\theta = \min(|\alpha|, \pi - |\alpha|)$ (D19):**
    - Nhìn Ngang (Side, $\theta < 30^\circ$, $N=183$): $Z_w = 0.4241$, $Z_h = 0.0482$, $Z_g = 0.2091$, (d) OOF = **0.0800**. $Z_h$ đơn lẻ thắng Fused vì $w/h$ phóng đại do chiều dài xe nhìn ngang, kéo $Z_w$ vào làm tăng sai số. Đây là động lực tự nhiên cho mô hình residual (f).
    - Nhìn Chéo (Diagonal, $30^\circ \le \theta \le 60^\circ$, $N=1,043$): $Z_w = 0.3080$, $Z_h = 0.0715$, $Z_g = 0.1125$, (d) OOF = **0.0568**.
    - Nhìn Đầu/Đuôi (Front/Rear, $\theta > 60^\circ$, $N=3,550$): $Z_w = 0.2611$, $Z_h = 0.0691$, $Z_g = 0.1554$, (d) OOF = **0.0573**.
  - **Bản chất sai số âm ở gần (D21):**
    - Pattern 111 (đủ 3 cue hợp lệ, không viền) ở 0–10m vẫn có bias âm $-0.1071$, xác nhận bias do khoảng cách $Z_{\text{center}}$ (tâm hộp 3D trong nhãn KITTI) so với $Z_{\text{surface}}$ (mặt gần nhất đo bởi cue), tạo độ lệch vật lý $\approx l / (2Z)$.

### Day 5 — 03/10/2026 (sáng): Xây dựng Khung Đánh giá Chuẩn hóa (eval.py, test_eval.py)
- **Mục tiêu:** Xây dựng module dùng chung `src/evaluation/eval.py` cho toàn bộ các khâu đánh giá từ Ngày 6 đến Tuần 3, bảo đảm tuân thủ v4 §6, D3, D11 và D18.
- **Thành phần cài đặt:**
  - Định nghĩa 5 dải khoảng cách chuẩn theo D3: `0-10`, `10-20`, `20-30`, `30-50`, `>50` m, kèm hàng gộp `>30` m và gắn cờ `low_n` ($n < 100$).
  - Phân tầng theo độ khó KITTI (`difficulty_masks`): Easy, Moderate, Hard.
  - Các hàm tính chỉ số: MAE, RMSE, AbsRel, SqRel, Delta1 ($\delta < 1.25$), RMSE_log.
  - Hàm `evaluate_report`: tổng hợp báo cáo phân nhóm theo dải hoặc theo nhãn nhóm, tự động đếm $n$, $n_{\text{valid}}$, tính `valid_frac` (không bỏ rơi các ca NaN/invalid).
  - Thuật toán cụm: `cluster_bootstrap` và `paired_cluster_bootstrap` lấy mẫu lại nguyên cụm drive độc lập với seed cố định, có cảnh báo khi số cụm $< 20$ (`MIN_CLUSTERS_WARN`).
  - Hàm ghi log tái lập: `make_log_record` và `append_jsonl`.
- **Kiểm thử đơn vị:**
  - Tạo `tests/test_eval.py` bao phủ toàn diện: tính tay các chỉ số, trường hợp hoàn hảo, phân dải, kiểm tra cờ `low_n`, cluster bootstrap và paired bootstrap, guard ghi log.
  - Toàn bộ 19 unit tests ban đầu pass 100% (và nay thêm test `sign_test_one_sided`).

### 03/10/2026 (chiều): Queue 100 epochs hoàn tất, Đánh giá last.pt (D12), Chốt conf thresholds (D6), Phân bổ B/C/T (D14) đóng băng splits-v2 & geometry-v2
- **Hoàn thành hàng đợi huấn luyện 3 detector (D7, 100 epochs, patience=100):**
  - Cả 3 detector đã hoàn thành 100 epochs trên GPU RTX 5060:
    - `yolov8s_640`: 70.56 phút (best.pt tại epoch 16, last.pt tại epoch 100).
    - `yolo11s_640`: 69.31 phút (best.pt tại epoch 15, last.pt tại epoch 100).
    - `yolov5su_640`: 65.22 phút (best.pt tại epoch 30, last.pt tại epoch 100).
- **Đánh giá trên Split V theo Quyết định D12 (chốt last.pt):**
  - Chạy `scripts/eval_detectors_val.py` trên Split V:
    - YOLOv8s (last): Car P = 0.6476, R = 0.8607, mAP50 = 0.8084, mAP50-95 = 0.5778.
    - YOLO11s (last): Car P = 0.6026, R = 0.8463, mAP50 = 0.8057, mAP50-95 = 0.5800.
    - YOLOv5su (last): Car P = 0.5353, R = 0.8511, mAP50 = 0.7727, mAP50-95 = 0.5495.
  - Toàn bộ 3 mô hình đều đạt mAP50 Car > 0.77 và Recall Car > 0.84 trên `last.pt`.
- **Chốt ngưỡng Confidence theo F1 làm trơn moving average 0.05 (D6):**
  - `yolov8s`: conf = 0.790 (smoothed F1 = 0.8355, raw F1 = 0.8368, P = 0.8807, R = 0.7971).
  - `yolo11s`: conf = 0.700 (smoothed F1 = 0.8370, raw F1 = 0.8363, P = 0.8283, R = 0.8445).
  - `yolov5su`: conf = 0.740 (smoothed F1 = 0.8240, raw F1 = 0.8227, P = 0.8254, R = 0.8200).
  - Đã lưu vào `configs/detector/conf_thresholds.yaml`.
- **Thực thi D14: Phân chia lại B/C/T, giữ nguyên 100% A và V:**
  - Chạy `scripts/repartition_bct.py` với 200 seed simulated annealing: 137/200 seed pass toàn bộ tiêu chí khắt khe.
  - Seed 85 được chọn theo đúng luật đăng ký trước (max min n_eff, lowest max KS):
    - B (1,499 frames, 4,776 Car Hard, 12 drives): top1 = 0.268, $n_{\text{eff}} = 5.29$.
    - C (766 frames, 1,826 Car Hard, 10 drives): top1 = 0.287, $n_{\text{eff}} = 5.28$ (giảm ngoạn mục từ 50.8% của drive 0059).
    - T (1,102 frames, 3,212 Car Hard, 10 drives): top1 = 0.233, $n_{\text{eff}} = 5.28$.
    - KS depth: B-C = 0.0460, B-T = 0.0588, C-T = 0.0556 (tất cả đều $\le 0.07$).
  - Ghi đóng băng `splits-v2` (`splits/split_metadata.json`, version v2).
  - Cập nhật frame counts trong `tests/test_splits.py`. Toàn bộ 84/84 test pass 100%.
- **Refit tham số hình học trên Split B mới (geometry-v2):**
  - Chạy lại `scripts/eval_day4_cues.py` trên Split B mới (1,499 frames, 4,776 Car Hard, 12 drives có xe):
    - Trọng số hợp nhất $[w_w, w_h, w_g] = [0.0807, 0.6634, 0.2560]$ (tất cả đều dương, tổng = 1.0).
    - Shrinkage alpha = 0.0008.
    - Gate §8.1: Thắng 3/4 dải độc lập $n \ge 100$ (10–20m: 0.0594 vs 0.0704, 20–30m: 0.0564 vs 0.0645, 30–50m: 0.0553 vs 0.0638; thua 0–10m: 0.1283 vs 0.0619).
    - Đóng băng vào `configs/geometry_params.yaml` với tag `geometry-v2`.
- **[SUPERSEDED (B-v1)] Xác minh 2.1 (Sửa cách tính Gate §8.1 - Quyết định D9):**
  - *(Lưu ý: Các số liệu trong phân tích này được thực hiện trên Split B-v1 cũ làm cơ sở lịch sử [AbsRel 0–10m = 0.1572, n=9 ở >50m]. Bảng trên Split B-v2 mới tương ứng là: AbsRel 0–10m = 0.1283, n=19 ở >50m; xem mục Day 6 hiện hành)*.
  - Gate chỉ tính trên các dải $n \ge 100$ độc lập: 0–10, 10–20, 20–30, 30–50 m (4 dải; không tính dải gộp `>30 m`, dải `>50 m` có $n=9 < 100$).
  - (d) thắng ở 3/4 dải và thua ở dải 0–10 m (AbsRel 0.1572 so với $Z_g$ 0.0651 và $Z_h$ 0.1463).
  - Gate ($\ge 3/4$) vẫn **ĐẠT CHUẨN**, nhưng tỷ lệ chính xác là **3/4 dải**, không phải 4/5. Đã nêu rõ dải thua 0–10 m trong báo cáo.
- **[SUPERSEDED (B-v1)] Xác minh 2.2 (Giải mã dải 0–10 m: So sánh tập chung & tổ hợp pattern):**
  - *(Lưu ý: Phân rã lịch sử trên Split B-v1: 69 xe pattern 100, 145 xe common support; đã SUPERSEDED bởi Split B-v2 và phân tích Day 6 hiện hành)*.
  - Đã chạy phân rã theo pattern: Đúng chính xác **69 xe** chênh lệch ở 0–10 m mang pattern `100` (chỉ còn $Z_w$ hợp lệ do xe gần chạm viền trên/dưới ảnh làm mask $Z_h$ và $Z_g$).
  - Ở nhóm `100`, AbsRel của (d) là **0.2284** (thoái hóa 100% về cue yếu nhất $Z_w$), kéo AbsRel chung từ 0.1238 lên 0.1572.
  - Trên tập chung (Common Support, 145 xe đủ 3 cue): (d) đạt AbsRel = **0.1238**, thắng $Z_w$ (0.1793) và $Z_h$ (0.1460).
  - Kết luận: Sai số 0–10 m là do thoái hóa cue khi bị cắt mép ảnh, không phải do trọng số hợp nhất sai. Sẽ được xử lý bằng validity flags trong mạng residual ở Tuần 2.
- **Xác minh 2.3 (Bản chất $H_{\text{cam}}$: Tham số hiệu dụng vs Chiều cao camera - Quyết định D10):**
  - Đã thực hiện phép thử nghiệm hồi quy Huber trên Split A:
    - Khi fit với khoảng cách tâm xe ($Z_{\text{center}}$): $H_{\text{cam}} = 2.0101$ m (hoặc 2.0422 m), $\delta = -4.985$ px.
    - Khi fit với khoảng cách góc tiếp đất gần nhất ($Z_{\text{closest}}$): $H_{\text{cam}} = \mathbf{1.7217}$ m, $\delta = \mathbf{0.5225}$ px $\approx 0$.
  - Kết luận: Mép đáy 2D ($y_{\text{bottom}}$) là góc tiếp đất gần nhất ($Z_{\text{closest}}$). Vì GT lấy tại tâm xe ($Z_{\text{center}} > Z_{\text{closest}}$), $H_{\text{cam}}$ bị nâng lên ~2.04 m để bù trừ. Đây là **effective ground-plane parameters**, đã quy ước chuẩn hóa trong bài báo.
- **Xác minh 2.4 (Bốn chi tiết kỹ thuật):**
  - (1) Mask biên: Đọc kích thước thực tế $(W, H)$ từng ảnh KITTI qua header PIL, dung sai $\epsilon = 2$ px.
  - (2) In-sample vs OOF: Bảng (d) là in-sample trên Split B (suy luận trên 10 drive có Car Hard của B).
  - (3) Prior: Sử dụng **median** ($W_{\text{eff}} = 2.6184$ m, $H_{\text{obj}} = 1.6797$ m).
  - (4) $n_{\text{gt}}$ trên V: Đã xác nhận $N_{\text{Car, Hard}} = 611$ trên Split V (thay cho con số 833 tổng Car chưa lọc Hard).
- **Đóng băng tham số hình học (D10 - SUPERSEDED bởi geometry-v2):** Tạo file `configs/geometry_params.yaml` bản đầu (`geometry-v1`) chứa toàn bộ tham số, trọng số $[0.0691, 0.6500, 0.2809]$ và hash của Split A (`4402...`) và Split B-v1 (`1242...`). Đóng băng tag `geometry-v1`. Bản hiện hành là `geometry-v2` fit trên Split B-v2 (hash `0f83c354...`) với trọng số $[0.0807, 0.6634, 0.2560]$ theo D14.
- **Chốt Quyết định D11:** Yêu cầu tách triệt để cột `gt_*` khỏi feature extractor và có test chặn.

### 02/10/2026 (chiều muộn): Chốt D7, D8, hoàn thành Day 4 & kiểm thử hình học
- **Cập nhật Quyết định D7 & Chạy lại hàng đợi detector:**
  - Đổi tên các checkpoint cũ thành `runs/detector/prelim_*_640` (superseded).
  - Cấu hình `patience: 100`, `epochs: 100` trong `train_config.yaml` để detector hoàn tất annealing LR và close_mosaic.
  - Chạy `scripts/train_queue.py` ở nền trên GPU RTX 5060.
- **Cập nhật Quyết định D8:**
  - Chuẩn hóa quần thể khớp: GT là Car thỏa Hard filter. Khớp GT Hard trước; detection khớp GT ngoài Hard hoặc DontCare bị bỏ qua, không tính TP/FP và không ranging.
  - Cập nhật `scripts/find_conf_thresholds.py` đọc `imgsz` từ config, ghi nhận commit git và cờ `git_dirty`.
- **Hoàn thành Ngày 4 (Hình học & Hợp nhất log-space trên GT bbox tập B):**
  - Prior Split A: $W_{eff}$ median = 2.6184 m (std rel = 35.5%), $H_{obj}$ median = 1.6797 m (std rel = 10.7%).
  - Hồi quy Huber mặt đất trên A: $\delta = -4.678$ px, $H_{cam} = 2.042$ m (giải thích chênh lệch so với 1.65m do Z lấy tại tâm xe thay vì góc gần nhất).
  - Masking biên ($\epsilon = 2$ px) trên B (4,210 Car Hard): $Z_w$ 2.26%, $Z_h$ 2.45%, $Z_g$ 2.64%.
  - Hợp nhất log-space: Ước lượng $\Sigma$ bằng grouped CV theo drive (10 drive, 4,038 mẫu), co shrinkage Ledoit-Wolf ($\alpha \approx 0.0009$). Trọng số tối ưu: $w_w = 0.0691$, $w_h = 0.6500$, $w_g = 0.2809$ (tất cả dương, tổng bằng 1.0).
  - Kết quả độ chính xác:
    - (a) $Z_w$: AbsRel = 0.2755 (cue yếu nhất, khớp giả thuyết H1)
    - (b) $Z_h$: AbsRel = 0.0719
    - (c*) $Z_g$ (fitted): AbsRel = 0.0975
    - (d*) Fused: AbsRel = **0.0635**, MAE = **1.50 m**, $\delta < 1.25$ = **98.0%**.
  - **Điều kiện sang Tuần 2 (§8.1):** ĐẠT CHUẨN! (d) thắng đơn cue tốt nhất ở 4/5 dải (10–20m, 20–30m, 30–50m, >30m).
  - Bộ kiểm thử `tests/test_geometry.py`: 23/23 tests pass 100%.

### 02/10/2026 (tối): [SUPERSEDED] Huấn luyện 3 detector (patience=15) & Ngưỡng conf thô (D6)
- **Huấn luyện thành công hàng đợi 3 detector (imgsz=640, batch=16, epochs=100, patience=15, seed=42 - SUPERSEDED):**
  - *(LƯU Ý QUAN TRỌNG: Cả checkpoint prelim do dừng sớm ở patience=15 và ngưỡng conf thô 0.430/0.600/0.650 trong phiên này đã chính thức bị thay thế / SUPERSEDED bởi phiên 03/10 chiều: Quyết định D7 huấn luyện 100 epochs tắt early stopping, D12 dùng last.pt, và D6 ngưỡng F1 trơn 0.790 / 0.700 / 0.740)*.
  - **YOLOv8s:** 22.93 phút. Car mAP50 = 0.8224, mAP@0.5:0.95 = **0.5754**, Recall = 0.8679.
  - **YOLO11s:** 22.89 phút. Car mAP50 = 0.8228, mAP@0.5:0.95 = **0.5923**, Precision = 0.7880. (Độ chính xác cao nhất).
  - **YOLOv5su:** 18.44 phút. Car mAP50 = 0.7748, mAP@0.5:0.95 = **0.5453**, Recall = 0.8463.
  - Tổng thời gian huấn luyện cả 3 detector: **64.26 phút** (rất nhanh nhờ tối ưu bộ nhớ VRAM 3.66 GB).
  - Trọng số tốt nhất đã lưu tại `runs/detector/{model}_640/weights/best.pt` (sau đó đổi tên thành `prelim_*_640`).
- **Chốt ngưỡng confidence threshold (Quyết định D6 - SUPERSEDED):**
  - Quét ngưỡng tìm $F_1$ tối đa trên lớp **Car** của tập V (bỏ qua detection khớp với DontCare theo §5.1):
    - [SUPERSEDED] - **YOLOv8s:** `conf = 0.430` $\rightarrow$ Max F1 = **0.8052** (Precision = 0.7921, Recall = 0.8187)
    - [SUPERSEDED] - **YOLO11s:** `conf = 0.600` $\rightarrow$ Max F1 = **0.7957** (Precision = 0.8469, Recall = 0.7503)
    - [SUPERSEDED] - **YOLOv5su:** `conf = 0.650` $\rightarrow$ Max F1 = **0.7798** (Precision = 0.8586, Recall = 0.7143)
  - Lưu cấu hình vào `configs/detector/conf_thresholds.yaml` (sau đó được ghi đè bằng cấu hình v2 chuẩn).

### 02/10/2026 (chiều): Chạy Pilot D5 YOLOv8s (imgsz = 640)
- **Kết quả Pilot YOLOv8s (imgsz=640, 30 epochs, batch=16, seed=42):**
  - Thời gian: 17.02 phút (24 epochs hoàn thành, Early Stopping kích hoạt tại epoch 24 vì đỉnh rơi vào epoch 9 với patience=15).
  - VRAM sử dụng: 3.66 GB / 8.15 GB (~45% VRAM, chạy mượt mà không bị thrashing).
  - **Chỉ số trên tập V (Val):**
    - **Toàn bộ (all):** P = 0.652, R = 0.507, mAP50 = 0.555, mAP@0.5:0.95 = 0.376
    - **Lớp Car (trọng tâm bài toán):**
      - **mAP@0.5:0.95 = 0.526**
      - **mAP@0.5 = 0.760**
      - **Recall = 0.826**
      - **Precision = 0.647**
    - **Van:** mAP50 = 0.354, mAP@0.5:0.95 = 0.256
    - **Truck:** mAP50 = 0.551, mAP@0.5:0.95 = 0.347
  - Tốc độ suy luận: **1.0 ms/ảnh (~1,000 FPS)** trên GPU RTX 5060!
  - Trọng số tốt nhất đã lưu tại `runs/detector/pilot_yolov8s_640/weights/best.pt`.
- **Đánh giá phần cứng imgsz = 960:** Thử nghiệm trước đó cho thấy 960 đẩy VRAM lên 8.02 GB (chạm trần 8GB VRAM của RTX 5060 Laptop GPU), kích hoạt cơ chế shared memory paging của Windows và làm chậm tốc độ huấn luyện xuống ~3 phút 50 giây/epoch (chậm hơn 6.5 lần so với 36s/epoch ở 640).

### 02/10/2026 (trưa): Chốt D1–D6, đóng băng splits-v1, YOLO dataset, priors, guard
- **Chốt D1:** Chạy KS test Car-only (Hard). C vs T có KS stat = 0.0391 ≤ 0.07 → Đạt chuẩn đóng băng split!
- Đã gắn tag git `splits-v1`. Split cũ (hash `fd3c...`, V=6 drive) chính thức superseded.
- **Chốt D2–D6:**
  - D2: Residual/CQR/CI chỉ cho Car; Van chạy mô tả; Truck chỉ ở mức detector.
  - D3: 5 dải khoảng cách thống nhất kèm cờ `*` nếu n<100, thêm hàng gộp >30 m.
  - D4: Tuần 2 không suy luận trên T; T khóa bằng guard.
  - D5: imgsz chốt bằng pilot YOLOv8s (640 vs 960, ~30 epoch).
  - D6: Ngưỡng conf F1 max Car trên V áp riêng từng detector. Đã sửa comment trong `train_config.yaml`.
- **Tạo `scripts/verify_data.py`:** Kiểm tra cấu trúc KITTI, P2 consistency = 1.0000, split hash & drive disjoint. Chạy pass 100%.
- **Tạo `tests/test_splits.py`:** 5 tests pass (kèm regression test H0 vs H1 và test guard Split T).
- **Tạo `scripts/make_yolo_dataset.py`:** Chuyển đổi thành công:
  - Train (A): 3,740 frames, 17,243 xe (Car: 14,902, Van: 1,519, Truck: 822).
  - Val (V): 374 frames, 1,051 xe (Car: 833, Van: 178, Truck: 40).
  - Tạo `data/yolo_kitti/dataset.yaml` và `SPLIT_HASH.json`.
- **Tạo `scripts/compute_priors.py`:** Tính prior hình học chỉ từ A:
  - $H_{cam}$ median = 1.886 m, $y_{horizon}$ cy = 174.8 px.
  - Car: $W_{eff}$ median = 2.618 m, $H_{obj}$ median = 1.680 m. Lưu vào `configs/geometry_priors.yaml`.
- **Cài đặt guard khóa tập T:** `load_split` chặn load T khi `allow_test=False` (bảo vệ exchangeability).
- **Tạo `src/detection/train_detector.py`:** Chuẩn hóa tham số Ultralytics, ghi commit git, split hashes và metrics vào JSONL.

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

