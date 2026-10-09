# ANTIGRAVITY_TASKS.md — kế hoạch giao việc cho agent (từ 04/10/2026)

Điểm xuất phát: Tuần 1 xong, `splits-v2` / `geometry-v2` / `detectors-v1` đã đóng băng, W2-1 xong (parquet B/C cho 3 detector). Bắt đầu từ W2-2. Ngày nộp thật chưa có trong project knowledge; có ngày nộp thì dồn phần dư vào ngày đệm.

## 1. Cách dùng

1. Đặt `AGENT_RULES.md` ở gốc repo. IDE có chỗ nạp rules/instructions cố định thì trỏ vào file này; nếu không, lệnh khởi động bên dưới đã đủ.
2. Mỗi tác vụ là một hội thoại/agent mới (context sạch). Không gộp tác vụ.
3. Lệnh khởi động, dán nguyên văn và thay `Txx`:

```
Đọc AGENT_RULES.md rồi thực hiện đúng tác vụ Txx trong docs/ANTIGRAVITY_TASKS.md, không làm gì ngoài Txx. Trước khi sửa file hoặc chạy lệnh, đưa plan ngắn (file sẽ tạo/sửa, lệnh sẽ chạy) và chờ tôi duyệt. Kết thúc bằng "Báo cáo tác vụ" đúng mẫu.
```

4. Sau mỗi tác vụ: bạn review diff, tự chạy `pytest -q`, commit thủ công, ghi nhật ký.
5. Chỉ sang tác vụ kế tiếp khi gate của tác vụ trước xanh. Trễ lịch thì cắt theo §6; không bao giờ nén các gate pre-register, freeze và T.

## 2. Lịch

| Ngày | Ngày v4 | Tác vụ | Gate người |
|---|---|---|---|
| 04/10 | W2-2 | T00, T01, T02, T03 (T03 chạy nền GPU) | Review diff nhật ký; `summary` của T01; bảng mask rate của T02 |
| 05/10 | W2-3 | T04 (A: pre-register → B: chạy) | Tag `prereg-residual-v1`; gate (f) vs (d) |
| 06/10 | W2-4 | T05, T06 | Xem bảng ablation, latency |
| 07/10 | W2-5 | T07 | Xem coverage theo drive |
| 08/10 | W2-6 | T08 (pre-register → chạy) | Tag `prereg-coverage-v1` |
| 09/10 | W2-7 | T09, T10, T11 | Review diff `run_inference.py`; tag `final-config-v1` |
| 10/10 | W3-1 | T12 (người chạy), T13 | Kiểm lock + log của T12 |
| 11/10 | W3-2 | T14 | |
| 12/10 | W3-3 | T15 | |
| 13/10 | W3-4 | T16 | Tier 2 chỉ khi thầy xác nhận |
| 14–16/10 | W3-5..7 | T17, T18, đệm thật | |

## 3. Điều kiện trước khi giao agent (bạn tự kiểm)

- [ ] `git status` sạch; `git tag -l` có `splits-v2`, `geometry-v2`, `detectors-v1` (thiếu thì tạo trước T01).
- [ ] `pytest -q` xanh.
- [ ] Có đủ 18 file parquet trong `results/predictions/` (3 model × {B, C} × {detections, matches, gt}).
- [ ] Đã duyệt D23–D30 ở §4 (đổi ⏳ thành ✅ hoặc sửa).
- [ ] Đã gửi 4 câu hỏi cho thầy (ảnh hưởng T06 Tier 2 và T16).

## 4. Quyết định chờ duyệt (D23–D30)

| # | Nội dung | Trạng thái |
|---|---|---|
| D23 | Quần thể ranging = detection TP ở `pass_thr`; `z_gt` chỉ join sau khi đặc trưng đã tính. Sensitivity ở sàn `conf ≥ 0.05` và ở eps mask của detector chỉ là ablation, không đổi cấu hình đóng băng. | ✅ |
| D24 | Tuning và chọn cấu hình chỉ trên OOF của B (nested grouped CV). OOF B∪C chỉ để báo cáo mô tả sau khi đóng băng. | ✅ |
| D25 | Grid XGBoost nhỏ, pre-register: `max_depth ≤ 4`, `min_child_weight` lớn, `n_estimators` cố định trong fold (không early stopping trên fold giữ lại), ≤ 24 cấu hình. Bắt buộc có baseline (f0) tuyến tính và (e). `configs/residual/residual_config.yaml` (depth 6, 500 cây) bị thay thế. | ✅ |
| D26 | Giao thức coverage: LODO calibration trong C cho dev; 20 lần chia lại B∪C theo drive (seed 0–19, ba phần fit/calib/eval, luật ghi trong pre-register); Mondrian bin theo Ẑ, gộp bin ≥ 30 m; mọi CI ghi "thô (k cụm)". | ✅ |
| D27 | T chạy một lần, chỉ qua `scripts/run_final_T.py`: tham số `allow_test` kiểm soát, lock file chặn lần chạy thứ hai, bắt buộc có tag `final-config-v1` trùng HEAD, kiểm SHA checkpoint + hash split. Guard công khai của `run_inference.py` giữ nguyên. | ✅ |
| D28 | `fallback_flag` là metadata, không nằm trong feature whitelist. | ✅ |
| D29 | Z_d trên B dùng trọng số LODO (OOF theo drive) làm nền cho target residual; C và T dùng trọng số fit trên toàn B. | ✅ |
| D30 | `ln_z_base` là đặc trưng dẫn xuất (suy ra được lúc test) cho (f0), (f); nằm ngoài whitelist D11 với guard test riêng; T05 có ablation bỏ nó. | ✅ |
| D33 | Pre-registration residual: `configs/residual/residual_prereg_v1.yaml` đóng băng grid 12 cấu hình (depth ≤ 4), bộ đặc trưng (f: 17, f0: 5, e: 10), loại bỏ `class_id` và `fallback_flag`. | ✅ |
| D34 | Dựng $Z_{\text{base}}$ qua inner-LODO 11 fold trên 11 drive train để tạo $Z_e$ cho pattern 000, tránh rò rỉ target trong tập train. | ✅ |
| D35 | Code freeze sau khi phát triển trên `yolo11s_640`; sửa code thì chạy lại toàn bộ 3 detector. | ✅ |
| D36 | Thứ tự detector: chạy `yolo11s_640` trước, sau đó chạy `yolov8s_640` và `yolov5su_640` với cùng mã nguồn. | ✅ |
| D37 | Ablation (T05) chỉ chạy trên OOF của Split B (12 folds), cấm chạm Split C (giữ nguyên vẹn cho conformalization). | ✅ |
| D38 | Quy chuẩn Drop Single Cue ($Z_k$): loại bỏ $Z_k$ khỏi fusion, bỏ $\ln z_k$ và $\text{valid}_k$ khỏi Model (f); pattern 000 đi fallback (e); `fit_fusion_weights` hỗ trợ `cue_names` tùy chọn. | ✅ |
| D39 | Thứ tự cắt khi trễ: cắt ngay Jitter (J1) trước khi chạy; nếu trễ T06 dời về W3-4 (T16), tuyệt đối không dời sang W2-5 (đường găng của T07 CQR). | ✅ |
| D40 | Chuẩn đo đạc Latency Tier 1 (T06): GPU chính dùng PyTorch FP16 trên CUDA; CPU chính dùng ONNX Runtime CPU FP32; không chạy song song T05 và T06. | ✅ |
| D41 | Đánh giá OOF của B: báo cáo cả pooled và macro (theo 12 drive của B). Gate T04 yêu cầu AbsRel OOF của (f) < (d) ở cả pooled và macro. (Đổi từ D31 tasks cũ để tránh trùng D31 nhật ký). | ✅ |
| D42 | Cơ chế Canary xáo nhãn trong train: kiểm tra rò rỉ label trong pipeline nested LODO B. OOF khi train trên nhãn xáo không được tốt hơn (d). (Đổi từ D32 tasks cũ để tránh trùng D32 nhật ký). | ✅ |
| D43 | Protocol fit toàn B: Model cuối (f)/(f0) fit trên Split B phải dùng nền $Z_{\text{base}}$ OOF (LODO $Z_d$ và OOF $Z_e$). Trọng số toàn cục `full_fw` được serialize cùng mã băm SHA. | ✅ |
| D44 | Latency Tier 1 sơ bộ: kết quả T06 là preliminary; Parity báo cả Count và IoU (không đổi ngưỡng). Đo lại toàn diện ở T16 kèm CQR. | ✅ |
| D45 | Ngôn ngữ kết quả T05: chỉ báo cáo mô tả số liệu, tôn trọng CI thô chứa 0 (không khẳng định vượt trội hay trực giao); giữ nguyên 17 features theo prereg. | ✅ |
| D46 | Mức conformal dùng order statistic chính xác $k = \lceil (n+1)(1-\alpha) \rceil$; $k > n \implies \hat{Q} = +\infty$; tránh lỗi off-by-one của numpy. | ✅ |
| D47 | Interval $[q_{lo} - \hat{Q}, q_{hi} + \hat{Q}]$ trong log-space; sort quantiles nhất quán; đếm crossing không silent clip; Winkler score ở log-space. | ✅ |
| D48 | Tiêu chí Gate T07 chấp nhận pooled $[85\%, 95\%]$; báo song song macro drive $n \ge 30$; fallback_flag chỉ báo n mô tả. | ✅ |
| D49 | Module suy luận out-of-sample dùng chung C và T tại `src/pipeline/apply_frozen.py`, assert khớp cues parquet. | ✅ |
| D50 | Cảnh báo drive-level heterogeneity: pooled coverage lệch do cụm lớn 0057 và 0004 under-cover; T08 phải chẩn đoán sâu. | ✅ |
| D51 | Nhóm fallback_flag không claim coverage do Model (e) under-predict lớn (|r| ≈ 2.56); chỉ báo cáo mô tả 0/n. | ✅ |

## 5. Hợp đồng dữ liệu (agent dùng chung)

**Đầu vào** (D22): `results/predictions/{model_key}_{split}_{detections|matches|gt}.parquet`, với `model_key ∈ {yolov8s_640, yolo11s_640, yolov5su_640}`, `split ∈ {A, B, C}`.

- `detections`: `frame_id, drive, pred_idx, x1, y1, x2, y2, confidence, class_id, pass_thr, fx, fy, cx, cy, img_w, img_h`
- `matches`: `frame_id, pred_idx, status (TP|FP|IGNORED_NONHARD|IGNORED_DONTCARE), matched_gt_idx (-1 nếu không TP), matched_iou`
- `gt`: `frame_id, drive, gt_idx, z_gt, x1, y1, x2, y2, difficulty, truncated, occluded, alpha`

**Khóa join:** `detections ⋈ matches` theo `(frame_id, pred_idx)`; `matches ⋈ gt` theo `(frame_id, matched_gt_idx == gt_idx)`. Không join theo vị trí hàng.

**Quần thể** (D23): hàng detection có `pass_thr == True` và `status == TP`. FN = GT Hard không có TP nào ở `pass_thr`; đếm riêng và luôn báo cáo. FP / IGNORED_* không đi vào ranging nhưng vẫn có trong số liệu detector.

**Đầu ra dữ liệu** (`results/datasets/`), hai file tách biệt theo D11/D22:
- `{model_key}_{split}_features.parquet`: cột quan sát được lúc test + định danh (`frame_id, drive, pred_idx, x1..y2, confidence, class_id, fx, fy, cx, cy, img_w, img_h`).
- `{model_key}_{split}_eval.parquet`: `frame_id, drive, pred_idx, gt_idx, z_gt, cls="Car", difficulty, truncated, occluded, alpha, matched_iou`, và về sau các cột dự đoán.
- `{model_key}_{split}_fn.parquet`: GT bị bỏ sót (chỉ để đánh giá).

Khung đánh giá cho `evaluate_report` cần các cột `z_gt, z_pred, cls, difficulty, drive`.

---

## 6. Tác vụ

### T00 — Dọn nhật ký/README + sign test (W2-2, dưới 1 giờ)
**Mục tiêu:** nhật ký/README phản ánh đúng hiện trạng; bỏ giá trị p hard-code.
**Đọc trước:** `docs/NHAT_KY_QUYET_DINH.md`, `README.md`, `scripts/eval_day6_analysis.py`, `results/tables/{day4_gt_bbox_evaluation.json, day6_gt_bbox_analysis.json, detector_eval_b_c.md, split_migration_v1_to_v2.md, repartition_bct_report.json}`.
**Làm:**
1. Chỉ chạy lệnh đọc: `git tag -l`, `git status --short`, `pytest -q`. Ghi baseline số test.
2. Soạn diff cho nhật ký (không xóa nội dung cũ, chỉ chú thích):
   - Thêm mục phiên cho Day 5 (`eval.py`, `test_eval.py`), Day 6 (D18–D21, số lấy từ `day6_gt_bbox_analysis.json`), W2-1 (D22, số lấy từ `detector_eval_b_c.json`).
   - Đánh dấu **SUPERSEDED (B-v1)** cho các bullet "Xác minh 2.1/2.2" ngày 03/10 và cho mục 02/10 (tối) (patience=15, conf 0.430/0.600/0.650); kèm con trỏ tới số v2 hiện hành.
   - D10: ghi rõ đây là bản `geometry-v1` (trọng số [0.0691, 0.6500, 0.2809], hash B `1242…`); hiện hành là `geometry-v2`.
   - Tick Ngày 5, Ngày 6; sửa dòng Ngày 5 đang ghi nhầm `test_geometry.py` (phải gồm `tests/test_eval.py`).
3. README: "Hungarian" → Greedy (D16); `data/yolo_format` → `data/yolo_kitti`; V "early stop" → "chọn checkpoint và ngưỡng conf (D7: không early stopping)".
4. Thêm vào `src/evaluation/eval.py` (additive) hàm `sign_test_one_sided(wins: int, losses: int) -> float` dùng `scipy.stats.binomtest(wins, wins + losses, 0.5, alternative="greater")` (ties loại trước khi gọi). Test: 9 thắng / 3 thua → 299/4096 ≈ 0.0730. Sửa `eval_day6_analysis.py` gọi hàm này thay cho `0.073`. KHÔNG chạy lại script Day 6.
**Xong khi:** `pytest -q` xanh (baseline + test mới); `git diff --stat` chỉ gồm nhật ký, README, `eval.py`, `test_eval.py`, `eval_day6_analysis.py`.
**Gate người:** đọc kỹ diff nhật ký (đây là nguồn chuẩn), rồi commit.

### T01 — `build_dataset`: parquet → tập ranging (W2-2)
**Mục tiêu:** dựng quần thể D23 và tách features/eval đúng D11/D22.
**Đọc trước:** §5 Hợp đồng dữ liệu; `src/residual/feature_extractor.py`; `tests/test_feature_guard.py`; schema trong `scripts/run_inference.py`; `results/tables/detector_eval_b_c.json`.
**Làm:**
1. `src/pipeline/__init__.py`, `src/pipeline/build_dataset.py`: `load_artifacts(model_key, split)`, `build_population(model_key, split, population: Literal["pass_thr", "floor"])` trả về `features`, `eval`, `fn`. Mode `floor` (mọi TP ở sàn 0.05) để T09 dùng.
2. `scripts/build_datasets.py --model all --split B C --population pass_thr`: ghi 3 file mỗi (model, split) theo §5, cộng `*_summary.json` (n_gt, n_TP, n_FN, n_FP, n_ignored, hash split, SHA file nguồn).
3. `tests/test_build_dataset.py` (dữ liệu giả): join đúng dù xáo thứ tự hàng; FP/IGNORED_* bị loại; mỗi TP ↔ đúng 1 GT, không GT trùng; cột `features` không chứa chuỗi cấm (`gt`, `depth`, `alpha`, `occluded`, `truncated`, `iou`, `status`, `target`); `extract_inference_features(features)` qua whitelist; `floor ⊇ pass_thr`.
**Xong khi:** số TP khớp `detector_eval_b_c.json` — B: v8s 3427, 11s 3523, v5su 3496; C: 1445, 1489, 1430 (script tự assert; lệch thì dừng). FN = n_gt − TP.
**Gate người:** xem `summary`, commit.

### T02 — Hình học trên bbox detector (D17, D29) (W2-2)
**Mục tiêu:** chạy (a)–(d) trên bbox detector và phân rã lỗi hình học vs lỗi detector.
**Đọc trước:** `src/geometry/{geometric_cues,fusion}.py`; `configs/geometry_params.yaml`; `scripts/eval_day4_cues.py` và `scripts/eval_day6_analysis.py` (tham chiếu); `docs/NHAT_KY_QUYET_DINH.md` D9, D17, D18, D20.
**Làm:**
1. `src/pipeline/geometry_stage.py`:
   - `load_geometry_priors(path)` đọc `geometry_params.yaml` (đừng dùng `load_priors_from_yaml`, hàm đó đọc `geometry_priors.yaml` cũ).
   - `add_cues(features)` gọi `compute_cues_batch` theo từng `frame_id` (intrinsics theo ảnh), eps 2 px.
   - `fit_fusion_lodo(...)`: LODO theo drive trên B cho Z_d OOF (D29), cộng trọng số/Σ fit toàn B cho C. Tái dùng `fit_fusion_weights`, `fuse_depths_vectorised`; không sửa chúng.
2. Thêm vào `eval.py` (additive) `macro_by_cluster(df, metric, ...)` + test tính tay.
3. Với mỗi detector: ghi `{model_key}_{B|C}_cues.parquet` và `results/tables/geometry_on_detector_bbox.{json,md}` gồm:
   - mask rate Z_w/Z_h/Z_g: bbox GT vs bbox detector, trên cùng tập TP;
   - (a)–(d) trên GT-bbox vs detector-bbox cùng đối tượng, pooled + macro, qua `evaluate_report` (n, n_valid, valid_frac);
   - cột trọng số GT-bbox đối chứng vs trọng số refit theo detector (D17);
   - `paired_cluster_bootstrap` (d)-detector vs (d)-GT-bbox, ghi "CI thô (12 cụm)";
   - FN và recall theo dải (từ `fn.parquet`).
4. Test: LODO không rò (drive test không có trong fit); NaN không lan sang cue khác.
**Xong khi (regression):** chạy hình học trên GT bbox của toàn B (4776 đối tượng, không qua detector) phải tái lập pooled OOF AbsRel ≈ 0.0609 (Day 6) và in-sample ≈ 0.0605 (Day 4), lệch ≤ 0.002. Lệch hơn thì dừng, báo.
**Cấm:** nếu mask rate của detector thấp hơn GT rõ rệt (bbox YOLO không chạm biên) thì chỉ báo cáo; KHÔNG đổi eps (biến thể eps là ablation ở T09).
**Gate người:** xem bảng mask rate và phân rã.

### T03 — Lệch bbox A vs B (chẩn đoán, GPU) (W2-2, chạy nền)
**Mục tiêu:** minh họa vì sao phải tách A khỏi B (v4 §0.1, §5.3).
**Đọc trước:** `scripts/run_inference.py`, `docs/KE_HOACH_V4.md` §5.3.
**Làm:**
1. Chạy `python scripts/run_inference.py --model all --split A` (script tự kiểm hash split và SHA checkpoint; tạo file `*_A_*.parquet` mới). Báo trước thời gian ước tính.
2. `scripts/diag_bbox_shift.py`: trên TP ở `pass_thr` của A, B, C: IoU với GT, lệch cạnh dưới (`y2_pred − y2_gt`, px và chia `h_gt`), lệch chiều cao/rộng tương đối; median, IQR, KS (A vs B, B vs C); recall A vs B.
3. Ghi `results/tables/bbox_shift_A_vs_B.md` và `results/figures/bbox_shift_A_vs_B.png`.
**Xong khi:** bảng + hình có đủ 3 detector. Nếu bbox A không chặt hơn B thì báo trung thực (ảnh hưởng tới luận điểm §0.1).
**Cấm:** dùng số liệu A để fit bất cứ thứ gì ngoài chẩn đoán này.

### T04 — Residual (f0), (f), (e) + pre-register (W2-3)
**Mục tiêu:** dựng mô hình residual và OOF trên B đúng D13, D16b, D24, D25, D28–D30.
**Đọc trước:** `docs/KE_HOACH_V4.md` §5.3, §5.5; D13, D16b, D25; `src/residual/feature_extractor.py`; `configs/residual/residual_config.yaml`; `results/tables/day6_gt_bbox_analysis.json`.
**Pha A — pre-register (xong thì DỪNG):**
- Tạo `configs/residual/residual_prereg_v1.yaml` chứa: grid XGBoost (≤ 24 cấu hình, depth ≤ 4, `min_child_weight` lớn, `n_estimators` cố định, `learning_rate` 0.05), inner CV (GroupKFold theo drive, ≤ 5 fold), tiêu chí chọn (rmse_log của OOF), tập đặc trưng của (f0), (f), (e), seed, `n_jobs`, gate, và mục `ablation:` cho T05 (xem T05).
- Thêm dòng "SUPERSEDED bởi residual_prereg_v1 (D25)" vào đầu `residual_config.yaml`.
- Bạn commit + tag `prereg-residual-v1`, rồi nhắn "GO pha B".
**Pha B — chạy:**
1. `src/residual/models.py`: `fit_f0` (Ridge trên tập đặc trưng tối thiểu + `ln_z_base`), `fit_f` (XGBoost, grid ở trên), `fit_e` (hồi quy trực tiếp ln Z, bỏ `ln_z_*` và `valid_*`). `ln_z_base` nằm trong tập `DERIVED_FEATURES` riêng + guard test (D30); `fallback_flag` là metadata (D28).
2. `src/pipeline/oof.py`: LODO 12 fold trên B. Z_base = Z_d nếu ≥ 1 cue hợp lệ, ngược lại Z_e OOF (D13, D16b). Target r = ln Z_gt − ln Z_base. Z_d dùng trọng số LODO (D29).
3. Đánh giá: `evaluate_report` + macro theo drive; `paired_cluster_bootstrap` cho (f) vs (d), (f) vs (f0), (f) vs (e); sign count (`sign_test_one_sided`); theo dải.
4. Lưu `results/datasets/{model_key}_B_oof.parquet` (z_base, z_hat_f0, z_hat_f, z_hat_e, fallback_flag) và model fit toàn B vào `runs/residual/{model_key}/` + `manifest.json` (SHA). Phát triển trên `yolo11s_640` trước, rồi chạy cùng code cho 2 detector còn lại, không chỉnh giao thức riêng từng detector.
5. Test: LODO không rò; **canary** hoán vị nhãn (xáo `z_gt` trong train → OOF không được tốt hơn (d)); guard `DERIVED_FEATURES`.
**Gate:** (f) có AbsRel OOF < (d) ở cả pooled và macro. Không đạt thì báo cáo đúng như vậy. Nếu (f0) ≈ (f) thì ghi "mô hình đơn giản đủ" (đây là kết quả, không phải lỗi).

### T05 — Ablation trên OOF của B (W2-4)
**Mục tiêu:** bảng ablation theo danh sách đã pre-register ở T04.
**Làm:** chạy đúng mục `ablation:` trong `residual_prereg_v1.yaml`: (1) bỏ từng nhóm đặc trưng: bbox geometry, edge flags, confidence, cues `ln_z_*`, validity flags, `ln_z_base` (D30); (2) bỏ từng cue (refit hợp nhất không có cue đó rồi chạy lại (f)); (3) MLP (sklearn `MLPRegressor` (128, 64), relu, cố định epoch/seed) vs XGBoost; (4) jitter bbox khi train, chỉ nếu còn thời gian (cắt đầu tiên). Chỉ dùng OOF của B, so với (f) đầy đủ, ghi "CI thô (12 cụm)".
**Cấm:** chạm C, T; thêm ablation ngoài danh sách đã đăng ký.

### T06 — Latency Tier 1 (W2-4)
**Mục tiêu:** bảng độ trễ từng khâu (v4 §5.6).
**Làm:** `scripts/bench_latency.py`: xuất ONNX 3 detector (`YOLO.export(format="onnx", imgsz=640)`, file `.onnx` đã nằm trong `.gitignore`); đo ONNX Runtime CPU (warmup 20, N = 200 ảnh từ B, cố định số luồng và ghi lại) và GPU FP16; tách khâu preprocess / detector / postprocess, hình học, residual (XGBoost predict); khâu CQR để trống, hoàn thiện ở T16. Kiểm tra parity ONNX vs `.pt` (số detection xấp xỉ). Ghi cấu hình phần cứng, phiên bản `onnxruntime`/`ultralytics`. Ra `results/tables/latency_tier1.{md,json}`.
**Cấm:** suy ra kết luận về độ chính xác từ ONNX.

### T07 — CQR lõi (W2-5) — [x] HOÀN THÀNH
**Mục tiêu:** CQR trên r, conformalize trên C, đánh giá dev.
**Đọc trước:** `docs/KE_HOACH_V4.md` §5.4; D13, D24, D26, D46–D49.
**Làm:**
1. `src/uncertainty/cqr.py`: `fit_quantile_xgb(q)` (cùng bộ đặc trưng và cấu hình đã chọn ở T04, không tune thêm), `conformalize(scores, alpha)` (mức order statistic thứ $\lceil (n+1)(1-\alpha) \rceil$, D46), `predict_interval` (sắp `q_lo ≤ q_hi`, đổi sang Z qua `exp`, đếm crossing không silent clip, D47), `winkler_score` ở log-space (D47), `assert_disjoint_drives(fit, calib)`.
2. `src/pipeline/apply_frozen.py`: chuẩn hóa luồng suy luận Split C và T, assert khớp `C_cues.parquet` (D49).
3. Fit hai phân vị 5%/95% trên B; conformalize trên C với α = 0.1. C đi qua đúng pipeline của T (cùng ngưỡng, cùng matching, cùng Hard, cùng fallback).
4. Dev: LODO calibration trong C (hiệu chỉnh trên C∖d, test trên d). Báo cáo coverage pooled + macro + macro ($n \ge 30$) + từng drive, độ rộng (Z_hi/Z_lo), Winkler score, coverage theo `fallback_flag`. Calibrate toàn C xuất `cqr_calib_C.json` cho T10/T11.
5. Test (`tests/test_cqr.py`): 10 tests pass (order statistic hand-crafted, LODO invariance, exchangeable coverage 200 trials, heteroscedasticity, crossing counter, disjoint guard, OOF Z_base invariance, Winkler properties). Toàn repo đạt 154 passed tests.
**Gate:** coverage tổng trên C (LODO) trong $[85\%, 95\%]$: **ĐẠT (PASS)** ở cả 3 detector (yolo11s: 87.17%, v8s: 87.13%, v5su: 87.97%; macro $n \ge 30$ đạt 90.01%–90.14%). Output: `results/tables/cqr_coverage_dev.{json,md}`.


### T08 — Conformal biến thể + 20 lần chia lại + coverage có điều kiện (dev) (W2-6) — [x] HOÀN THÀNH
**Mục tiêu:** split conformal, Mondrian CQR, độ ổn định coverage.
**Pha A — pre-register:** Đã hoàn tất và gắn tag `prereg-coverage-v1` tại commit `e1f9e5b`.
**Pha B — chạy:**
1. Ba phương án trên cùng mô hình: Split Conformal thường, Standard CQR, Mondrian CQR trên 3 detector (`yolo11s_640`, `yolov8s_640`, `yolov5su_640`).
2. 20 lần chia lại: báo cáo mean ± std và đầy đủ 20 giá trị cho cả 3 phương án và 3 detector tại `results/tables/coverage_stability_20resplits.{json,md}`. Độ phủ trung bình đạt 85.15% (CQR), 85.56% (Split), 84.53% (Mondrian) cho YOLO11s (tương ứng 86.14%, 87.41%, 84.49% cho YOLOv8s; 84.99%, 85.85%, 83.79% cho YOLOv5su). Độ rộng khoảng hẹp ($Z_{hi}/Z_{lo} \approx 1.24 - 1.26$), 0 crossing.
3. Coverage có điều kiện theo dải Ẑ, Z thật, truncated, occluded, cờ chạm biên, θ (D19), difficulty, fallback pattern 000 (D51) tại `results/tables/coverage_conditional_dev.{json,md}`. Mondrian CQR cải thiện độ phủ ở dải gần 0–10m (+6.8% đến +9.3%) và nhóm chạm biên (+4.4% đến +8.2%).
4. Test: Toàn bộ 164 tests pass (`tests/test_resplit.py`, `tests/test_cqr.py`). Đã log `T08-Coverage-Stability-20Resplits` vào `runs/pipeline_log.jsonl`.


### T09 — Sensitivity (W2-7 / W3-2) — [x] HOÀN THÀNH
**Mục tiêu:** ba phép kiểm độ nhạy, không đổi cấu hình đóng băng (D14, D23, D36, D39, D85).
**Kết quả thực hiện:**
1. T09.1: Bỏ `drive_0059` và `drive_0104` khỏi B trên detector chính `yolo11s_640`. Trọng số $w_h$ ổn định 69.8%–75.8%, $w_g$ ổn định 24.2%–30.2%, $w_w = 0.0$ (NNLS). Sai số ngoài mẫu OOS trên 2 drive lớn bị loại dao động rất nhỏ ($\Delta \le +0.0014$), Macro AbsRel qua 12 drives giữ nguyên 0.0739–0.0755. Output: `results/sensitivity/sensitivity_drives_drop.md`.
2. T09.2: Đánh giá sàn tin cậy `conf >= 0.05` vs `pass_thr`. Hạ ngưỡng giúp tăng 11%–15% TP nhưng Precision giảm từ ~81% xuống ~67%. Giữ nguyên cấu hình đóng băng `pass_thr` (D23). Output: `results/sensitivity/sensitivity_floor_pop.md`.
3. T09.3 (eps mask ablation) chủ động cắt giảm theo đúng thứ tự ưu tiên D39 do vượt mốc thời gian 16:30.

### T10 — `run_final_T.py` + dry-run trên C (W2-7)
**Mục tiêu:** một script duy nhất chạy T, đã kiểm chứng trên C.
**Đọc trước:** D4, D22, D27; `scripts/run_inference.py`; `src/utils/split_builder.py` (`load_split` guard).
**Làm:**
1. Refactor `run_inference.py` (additive): tách lõi `run_inference_core(..., allow_test: bool = False)`; `run_inference_for_model` giữ nguyên chữ ký, vẫn `PermissionError` với T; `tests/test_inference.py` phải xanh nguyên vẹn. **Diff này cần bạn review kỹ.**
2. `configs/pipeline_frozen_v1.yaml` (bản nháp): danh sách model, hash split A/V/B/C/T, SHA checkpoint, ngưỡng conf, hash `geometry_params.yaml`, cấu hình residual đã chọn/detector, SHA model đã fit, α, bin Mondrian, eps, seed.
3. `scripts/run_final_T.py`: trước khi chạy kiểm (a) tag `final-config-v1` tồn tại và trùng HEAD, cây làm việc sạch (bỏ qua `runs/`, `results/`), (b) hash split + SHA checkpoint + SHA model residual khớp config, (c) kiểm tra preflight (đĩa trống >= 1GB, thư mục ghi được, GPU có mặt) trước khi tạo lock (D67), (d) tạo `runs/final_T.lock` bằng `open(..., "x")`, lỗi nếu đã tồn tại, (e) cờ `--confirm FINAL_T_RUN` bắt buộc. Quy trình: inference T → `build_dataset` → hình học → Z_base → (f) → CQR/split/Mondrian → ghi `results/final/*` + một cặp sự kiện START/COMPLETED trong `runs/final_T_log.jsonl` (hoặc FAILED nếu sự cố). Không có tham số tinh chỉnh.
4. Chế độ `--dry-run C` (không tạo lock, không chạm T): chạy đúng quy trình trên C.
**Xong khi:** dry-run C tái lập **đúng** số liệu dev của T07/T08 trên C (lệch thì dừng); test cho lock/tag/hash (dùng thư mục tạm).

### T11 — Freeze (W2-7)
**Mục tiêu:** đóng băng cấu hình trước khi chạm T.
**Làm:** hoàn tất `pipeline_frozen_v1.yaml` (điền SHA/hash thật, ghi commit); chạy `pytest -q` toàn bộ; chạy `scripts/verify_data.py`; liệt kê file thay đổi từ tag `geometry-v2`; đề xuất message commit và lệnh tag. Soạn nháp mục Data/Method (không trích dẫn bịa, thiếu nguồn ghi `[CẦN TRÍCH DẪN]`).
**Gate người:** chạy tay `pytest -q`, review diff `run_inference.py`, commit, `git tag final-config-v1`. Gate sang Tuần 3: có (a)–(g) cho ≥ 2 detector, coverage dev gần 90%.

### T12 — Chạy T một lần (W3-1) — **người chạy, KHÔNG giao agent**
1. `git status` sạch; `git rev-parse HEAD` trùng `git rev-parse final-config-v1`; `runs/final_T.lock` chưa tồn tại.
2. Chạy: `python scripts/run_final_T.py --confirm FINAL_T_RUN` (không thêm tham số).
3. Sau khi xong: kiểm `runs/final_T.lock`, đúng một cặp sự kiện START/COMPLETED trong `runs/final_T_log.jsonl`, đủ file trong `results/final/`; commit kết quả, `git tag final-run-T-v1`.
4. Nếu script sập vì lỗi hạ tầng (đĩa, OOM) mà chưa sinh kết quả: chỉ chạy lại sau khi bạn ghi nhật ký lý do (D-mới) và tự xóa lock. Không chạy lại vì kết quả "xấu".

### T13 — Bảng chính trên T (W3-1)
**Mục tiêu:** bảng (a)–(g) trên T từ `results/final/`.
**Làm:** `evaluate_report` cho (a)–(g) từng detector và trên **tập khớp chung** của các detector; pooled + macro; bảng per-drive (10 cụm); `paired_cluster_bootstrap` cho (d) vs (f) và giữa các detector trên tập khớp chung (CI thô, 10 cụm). Lưu ý (g) có cùng ước lượng điểm với (f), nên (f) vs (g) chỉ có nghĩa ở chỉ số khoảng: so coverage/độ rộng/interval score giữa CQR, split và Mondrian. RQ2: liên hệ sai số ranging với IoU và lệch cạnh dưới (dữ liệu eval).
**Cấm:** refit, tune, chạy lại pipeline trên T. Kết quả xấu thì báo cáo như vậy; không xếp hạng detector khi CI chứa 0 hoặc chồng lấn.

### T14 — Phân tích lỗi (W3-2) — [x] HOÀN THÀNH
**Mục tiêu:** Phân tích lỗi chuyên sâu trên Split T tuân thủ Zero-Touch (D70), báo cáo thiên lệch kẻ sống sót qua $n_{TP}, n_{FN}$, Recall (D32), kiểm chứng D19, D21, D84 và chuẩn bị dữ liệu cho T16.
**Kết quả thực hiện:**
1. Phân rã theo 5 canonical distance bins (kèm >30m và cờ `*` khi $n < 100$), difficulty (nested và disjoint), occlusion (0, 1, 2), truncation, và 10 cụm drive của Split T cho cả 3 detector (`yolo11s_640`, `yolov8s_640`, `yolov5su_640`). Báo cáo đầy đủ $n_{TP}, n_{FN}, n_{GT}$, Recall, $k$, AbsRel, MAE, RMSE, $\delta_1$. Output: `results/tables/error_analysis_breakdown_T.{json,md}`.
2. Kiểm chứng góc nhìn D19: Cue $z_w$ suy biến mạnh ở góc nhìn Side (< 30°, AbsRel ~ 0.395), trong khi $z_h$ và $z_g$ ổn định (AbsRel ~ 0.066 và 0.114), Residual model $z_{\hat{f}}$ bù trừ tốt nhất (AbsRel ~ 0.053). Output: `results/tables/viewing_angle_d19_verification_T.md`.
3. Kiểm chứng sai số tâm vật lý D21 & D84 (D107): Ở 0–10m trên nhóm Pattern 111 không chạm biên (`valid_w & valid_h & valid_g`), mô hình hình học thuần $z_d$ thể hiện độ lệch âm hệ thống (Mean Bias: -0.95m đến -1.02m; Median Rel Bias: -10.9% đến -12.2%), nhất quán với giả thuyết lệch tâm 3D $l/2 \approx 1.0\text{ m}$ của D21. Mô hình Residual $z_{\hat{f}}$ hấp thụ và triệt tiêu thành công độ lệch này (Median Rel Bias giảm về -0.27% đến -1.16%; Median Bias: -0.02m đến -0.10m). Chi tiết số liệu phân rã theo detector lấy trực tiếp từ file nguồn: `results/tables/physical_bias_d21_verification_T.{md,json}`.
4. Trích xuất Top 50 thất bại lớn nhất (AbsRel từ 0.165 đến 0.429, 12 ca fallback, 20 ca ở gần 0-10m) và 10 ca thành công đại diện (seed=42) sẵn sàng cho T16. Output: `results/tables/error_analysis_top_failures_T.md` và `results/final/top_failures_manifest.json`.
5. Unit tests: 8/8 tests pass (`tests/test_error_analysis.py`). Guard 3 pass 100%. Split T lockfile giữ nguyên vẹn.

### T15 — Coverage có điều kiện trên T (W3-3) — [x] HOÀN THÀNH
**Mục tiêu:** coverage và độ rộng của CQR / split / Mondrian trên T theo dải, truncated, occluded, cờ chạm biên, θ, `fallback_flag`, từng drive; kiểm exchangeability C vs T (KS mô tả cho Ẑ, confidence, tỉ lệ mask) và chỉ ra chỗ vỡ nếu có; mọi CI "thô (10 cụm)".
**Kết quả thực hiện (D86–D93):**
1. Bóc tách độ phủ qua 7 phân nhóm có điều kiện trên cả 3 detector (`yolo11s_640`, `yolov8s_640`, `yolov5su_640`). Báo cáo đầy đủ $n_{\text{TP}}, n_{\text{FN}}, n_{\text{GT}}$, Recall, $k$, Coverage, Mean Width, Winkler score trong log-space và 0 crossing. Thống nhất cờ `*` cảnh báo cỡ mẫu phân nhóm khi $n_{\text{TP}} < 100$ (D54, D86). Tuân thủ D91: Recall chỉ tính cho các phân nhóm xác định bằng thuộc tính Ground Truth; các nhóm theo biến suy luận test-time (Ẑ, cờ hợp lệ `valid_*`, Fallback) hiển thị "---" vì False Negatives không có thông tin dự đoán. Output: `results/tables/coverage_conditional_T.{json,md}`.
2. Bóc tách theo cụm drive và Macro kép (D50, D75, D86): Cụm `drive_0002` ($n=2$, cờ `*` khi $n < 30$) kéo macro 10 drive của YOLO11s xuống 87.93%, trong khi trên 8 drive có $n \ge 30$, macro coverage đạt 97.41%–97.50% đồng đều ở cả 3 detector. Output: `results/tables/coverage_per_drive_T.{json,md}`.
3. Chẩn đoán tính khả hoán $C \leftrightarrow T$ (KS mô tả, D87): Báo cáo chỉ số phân kỳ phân bố thực nghiệm $D_{\text{KS}}$ (0.1427–0.1621 trên sai số log-residual $|r|$, Mean $|r|$ trên C là 0.081–0.085 vs 0.067 trên T). Không suy diễn nominal p-values để tránh lỗi pseudo-replication trên 10 cụm drive (AGENT_RULES §6.2, D20). Ghi nhận hiện tượng over-coverage ngoài mẫu (96.4%–97.1%) là phát hiện hậu nghiệm (post-hoc / exploratory), đối chiếu với tiên đoán under-coverage ban đầu tại D68. Output: `results/tables/exchangeability_c_vs_t.{json,md}`.
4. Xác nhận độ phủ nhóm Fallback Pattern 000 ($n=36$) đạt 77.8%–86.1%, củng cố việc sửa lỗi cú pháp `base_score` v1.1 theo D74. Giữ CQR theo tiên nghiệm D55; Mondrian cải thiện ở cự ly gần nhưng tăng độ rộng khoảng ($1.45\times$ lên $1.80\times$), không xếp hạng ép buộc khi CI chồng lấn (D52, D88).
5. Ghi nhận độ phủ CQR dải gần 0–10m trên T đạt 89.4%–93.6% (AbsRel ~6.6%), duy trì ổn định nhờ ngưỡng sai số log-residual $\hat{Q}$ toàn cục kế thừa từ C bao trùm an toàn (hiện tượng bảo thủ dải gần mang tính exploratory). Bác bỏ giả thuyết võ đoán trước đó về rung lắc tiếp đất.
6. Unit tests: 9/9 tests pass (`tests/test_conditional_coverage.py`), bao gồm test kiểm tra D91. Toàn bộ repo đạt 196 passed tests. Đã sửa bug log và ghi mã băm Split T `82c7f9e4...` vào `runs/pipeline_log.jsonl`. Tự động sinh bảng số liệu từ JSON có assert đối chiếu 100% (D89).

### T16 — Latency cuối + hình định tính (W3-4) — [x] HOÀN THÀNH
**Mục tiêu:** Đo độ trễ Tier 1 chính thức khắc phục 4 lỗi phương pháp luận của T06 (kèm khâu CQR, D40, D44, D94, D95, D98, D99) và trực quan hóa định tính 6–10 hình chuẩn mực tuân thủ Zero-Touch Split T (D70, D90, D96, D97). Tier 2 (INT8 YOLO11n) chủ động bỏ qua do chưa có xác nhận phần cứng nhúng từ thầy (§2, §8.0, D39).
**Kết quả thực hiện (D94–D99):**
1. **Benchmark Độ trễ Tier 1 Chuẩn hóa (`scripts/bench_latency.py`, D94, D98, D99):**
   - Khắc phục 4 lỗi phương pháp luận của bản sơ bộ T06: (1) Tách riêng tiền xử lý, đo model forward thô bằng `model_pt.model(x)` (Half FP16 CUDA synchronize), NMS decode riêng, loại bỏ hoàn toàn double-counting; nạp trước (preload) 200 ảnh vào RAM nhằm loại bỏ Disk I/O khỏi vòng lặp đo (In-memory benchmark); (2) Đo $t_{\text{total}}^{(i)}$ end-to-end độc lập trên từng ảnh, tính trực tiếp Median, Mean, P95, IQR, FPS trên chuỗi tổng per-image, loại bỏ sai lệch sum-of-medians ($0.20–0.63$ ms, $\le 2.0\%$); (3) Trích xuất đặc trưng residual vector hóa 17 features thực tế với confidence thật và P2 intrinsics từng frame (sửa `cx_offset_norm = (box_cx - cx) / fx` khớp 100% với `feature_extractor.py`); (4) Đo đạc đầy đủ khâu Conformal Quantile Regression (CQR 2 mô hình XGBoost $q_{0.05}$ và $q_{0.95}$, scaling log-residual và $\hat{Q}$).
   - Đo trên 200 ảnh Split B nạp sẵn trong RAM (20 warmup):
     * `yolo11s_640`: GPU = **38.01 ms (P95: 76.64 ms, 26.3 FPS)**; CPU = **116.73 ms (P95: 122.17 ms, 8.6 FPS)**.
     * `yolov8s_640`: GPU = **31.70 ms (P95: 60.33 ms, 31.5 FPS)**; CPU = **142.54 ms (P95: 148.49 ms, 7.0 FPS)**.
     * `yolov5su_640`: GPU = **33.12 ms (P95: 64.57 ms, 30.2 FPS)**; CPU = **119.87 ms (P95: 124.53 ms, 8.3 FPS)**.
     * Nhãn dữ liệu (D98): Gán nhãn `PRELIMINARY-v2` cho kết quả đo một lượt trên máy trạm laptop; cấm xếp hạng throughput giữa các detector khi chênh lệch nằm trong biên độ nhiễu phần cứng.
     * Tiền xử lý in-memory: GPU ~9.59–9.88 ms, CPU ~6.12–6.17 ms.
     * Chi phí khâu hậu detector: Hình học ~0.18 ms, Residual ~0.57–0.60 ms, CQR ~0.58–0.60 ms (tổng cộng ~1.35 ms/ảnh, ~3.5%–4.2% tổng thời gian GPU), chứng minh overhead của hybrid và conformal là không đáng kể so với detector.
   - Parity Check PyTorch vs ONNX (D95, D99): Count Parity đạt [0.95, 1.05] (0.974–1.009), IoU Match Rate đạt 0.9095–0.9437 (dưới ngưỡng 0.95 do khác biệt đệm dynamic letterbox vs static square 640). Đánh nhãn minh bạch: "Count ĐẠT, IoU KHÔNG ĐẠT"; nguyên nhân giải thích được xác định là giả thuyết kỹ thuật chưa kiểm chứng độc lập (D99).
   - Artifacts: `results/tables/latency_tier1.json`, `results/tables/latency_tier1.md`, log `runs/pipeline_log.jsonl` (ghi mã băm Split B `0f83c354...`).
2. **Hình ảnh Định tính Split T (`scripts/make_qualitative.py`, D90, D96, D97):**
   - Tuân thủ nghiêm ngặt Zero-Touch Split T (D70, D90, D96 mở rộng có kiểm soát): Chỉ đọc read-only tệp ảnh PNG gốc (`data/kitti/image_2/{frame_id}.png`) theo `frame_id` và static predictions parquet. Khóa `runs/final_T.lock` giữ nguyên vẹn 100%. Ghi log vào `runs/pipeline_log.jsonl`.
   - Xuất 8 hình ảnh nghiên cứu điển hình 300 DPI + 1 lưới tổng hợp 4x2 (`qualitative_grid_summary.png`) + manifest JSON (`results/figures/qualitative_manifest.json`), loại bỏ 100% số liệu hardcode trong text mô tả theo D97:
     * `qualitative_01_side_view_d19.png`: Góc nhìn ngang (D19, frame `000006`, $Z_{\text{gt}}=19.72$m, $Z_w=9.81$m under-predict sâu do phình $w$ thành chiều dài xe, $\hat{Z}_f=20.60$m, CQR $[17.44, 23.63]$m, AbsRel 4.48%, Covered=True).
     * `qualitative_02_near_physical_bias_d21.png`: Lệch tâm vật lý 3D dải gần (D21/D92, frame `000385`, $Z_d=7.02$m lệch -11.2%, $\hat{Z}_f=7.99$m, $Z_{\text{gt}}=7.91$m, CQR $[6.83, 9.00]$m, AbsRel 0.96%, Covered=True).
     * `qualitative_03_truncated_edge_d84.png`: Cắt biên ảnh chạm đáy (D84, frame `000152`, truncation 0.35, CQR nở rộng $[4.08, 7.07]$m chứa an toàn $Z_{\text{gt}}=6.37$m, $\hat{Z}_f=6.14$m, AbsRel 3.55%, Covered=True).
     * `qualitative_04_fallback_pattern000_d74.png`: Fallback mất sạch cue hình học (D74, frame `000211`, $\hat{Z}_e=7.75$m, $\hat{Z}_f=8.23$m, CQR $[7.22, 10.33]$m chứa an toàn $Z_{\text{gt}}=7.91$m, AbsRel 4.03%, Covered=True).
     * `qualitative_05_top1_failure.png`: Ca lỗi lớn nhất (frame `001414`, $Z_{\text{gt}}=5.92$m, $\hat{Z}_f=8.46$m, CQR $[7.05, 10.52]$m, AbsRel 42.92%, Covered=False; xe bị cắt góc viền nặng, fallback direct model chịu lỗi ngoại suy phối cảnh - giả thuyết quan sát mô tả, chưa kiểm chứng).
     * `qualitative_06_success_10_20m.png`: Thành công 10–20m (frame `003811`, $Z_{\text{gt}}=11.19$m, $\hat{Z}_f=11.19$m, CQR $[9.67, 12.63]$m, AbsRel 0.04%, Covered=True).
     * `qualitative_07_success_20_30m.png`: Thành công 20–30m (frame `003314`, $Z_{\text{gt}}=24.41$m, $\hat{Z}_f=24.27$m, CQR $[22.27, 26.70]$m, AbsRel 0.59%, Covered=True).
     * `qualitative_08_success_hard.png`: Thành công ca khó Hard KITTI (frame `004401`, box `#5`, $Z_{\text{gt}}=20.19$m, $\hat{Z}_f=19.67$m, CQR $[17.54, 23.33]$m, AbsRel 2.59%, Covered=True).
3. **Kiểm thử và Hồi quy:**
   - 8/8 unit tests pass trong `tests/test_latency_bench.py`.
   - 13/13 unit tests pass trong `tests/test_qualitative.py` (bao gồm test D97 kiểm tra zero hardcode floats).
   - Toàn bộ test suite dự án `pytest -q`: **217 passed** 100%.

### T17 — Xuất bảng/hình + tư liệu bài (W3-5..6) — [x] HOÀN THÀNH TOÀN DIỆN (PHA 1, PHA 2 & T17b) (D108–D118)
**Mục tiêu:** Xuất toàn bộ 7 cặp bảng chính thức sang `results/tables/final/` (CSV + LaTeX booktabs); xây dựng `results/final/numbers_manifest.json` ánh xạ 100% con số trong bài tới file nguồn và SHA-256; đồng bộ hóa 5 hình vẽ khoa học $\ge 300$ DPI vào `results/figures/final/` kèm `figures_manifest.json`; xây dựng tệp trích dẫn BibTeX chuẩn `docs/paper/references.bib`; hoàn thiện bản thảo `docs/paper/MANUSCRIPT_DRAFT.md` không còn placeholder số ảo và bao quát đủ 14 Hạn chế cốt lõi. Khắc phục triệt để mọi lỗi P0 và P1 từ đánh giá phản biện độc lập qua tác vụ T17b.
**Kết quả thực hiện (D108–D118):**
1. **Numbers Manifest (`scripts/build_numbers_manifest.py`, D108):**
   - Sinh `results/final/numbers_manifest.json` gồm **81 metrics chuẩn hóa** trích xuất động 100% từ 7 artifact nguồn đã kiểm định (đọc động `n_frames` và `n_gt`).
   - 100% metrics có `source_file`, `source_sha256`, `data_path` kiểm chứng bit-by-bit.
2. **7 Bảng xuất bản CSV & LaTeX booktabs (`scripts/export_final_tables.py`, D109, D111, D114, D118):**
   - 14 tệp trong `results/tables/final/` (`tab_01` đến `tab_07`).
   - 100% tuân thủ booktabs (không kẻ dọc `|`, không double `\hline`, escape `%` và `_` triệt để).
   - Bảng 2 đọc động single cues từ `geometry_on_detector_bbox.json` (Split B eval_split_B).
   - Bảng 3 ghi rõ $\Delta(f - e)$ với ước lượng âm $-0.0002\ [-0.0012, 0.0023]$, loại bỏ khả năng nhầm chiều hiệu số.
   - Bảng 5 hiển thị `---` cho cột `Macro (n >= 30)` của Split Conformal và Mondrian CQR (không fallback nhầm sang macro 10 cụm).
   - Bảng 7 (Latency Tier 1) mang nhãn bắt buộc `PRELIMINARY-v2`, phân rã khâu hậu xử lý chỉ tốn $\approx 1.35$ ms/ảnh ($< 4.2\%$ GPU pipeline).
3. **5 Hình vẽ khoa học xuất bản (`scripts/export_final_figures.py`, D112, D114):**
   - Xuất bản 5 hình vẽ đạt chuẩn $\ge 300$ DPI, palette colorblind-safe vào `results/figures/final/`: `fig_01` (Kiến trúc tổng thể không trọng số gõ tay), `fig_02` (Phân bổ split A/V/B/C/T đồng bộ 100% Car Hard), `fig_03` (Sai số AbsRel theo 5 dải khoảng cách tính động từ parquet, dải $>50$m gắn cờ `*`), `fig_04` (Độ phủ CQR và độ rộng khoảng tin cậy theo ODD tính động từ json), và `fig_05` (Lưới 8 trường hợp định tính fail-loud nếu thiếu file, kèm Disclaimer).
   - Tích hợp bản đồ đồ họa `results/figures/final/figures_manifest.json` có mã băm SHA-256 đối chiếu 100%.
4. **Hệ thống trích dẫn BibTeX & Bản thảo hoàn thiện (`scripts/render_manuscript.py`, D108, D110, D113, D115, D116, D117):**
   - Tạo tệp `docs/paper/references.bib` với đầy đủ 12 mục trích dẫn chuẩn mực theo Kế hoạch v4 §12, không tự chế metadata (D115).
   - Bổ sung Section 5.1 Error Decomposition & Geometric Baseline Analysis (kết nối Table 2, T02, D19, D21, D31).
   - Loại bỏ 100% mã quyết định nội bộ (Dxx) trong `MANUSCRIPT_TEMPLATE.md` và `MANUSCRIPT_DRAFT.md` (D116).
   - Chuẩn hóa văn phong khoa học (D117): cấm claim "equivalence", "finite-sample coverage guarantees", "fail-safe", "comfortably", "fails catastrophically"; hạ giọng giải thích IoU parity thành giả thuyết kỹ thuật.
   - Tiêu đề mục Limitation 8 đổi thành "Empirical Indistinguishability of Residual and Direct Regression under 10-Cluster Sample".
5. **Kiểm thử và An toàn (D118):**
   - `runs/final_T.lock` giữ nguyên vẹn 100% (Zero-Touch Split T).
   - Triệt tiêu hoàn toàn vacuous tests: `test_csv_tex_value_consistency` kiểm tra đúng tên cột số liệu thực tế (`absrel_pooled`, `pooled_coverage`) và assert $\ge 15$ dòng (Tab 3) và $\ge 9$ dòng (Tab 5).
   - 11/11 unit tests pass trong `tests/test_final_tables_and_manifest.py`.
   - Toàn bộ test suite dự án `pytest -q`: **229 passed** 100%.

### T18 — Audit checklist §11 (W3-6..7)
**Làm:** `docs/CHECKLIST_AUDIT.md`: với từng mục §11 của v4, ghi bằng chứng (lệnh + kết quả, file, test) hoặc FAIL. Tối thiểu có các kiểm tra tự động: (1) không cột GT trong mọi `*_features.parquet`; (2) hash A/V/B/C/T khớp metadata; (3) T chỉ được chạm một lần (một cặp sự kiện START/COMPLETED trong `runs/final_T_log.jsonl`, có lock, tag khớp); (4) `grep` toàn repo: không đường code nào ngoài `run_final_T.py` mở T; (5) mọi bảng chính có n, n_valid, k cụm; (6) không còn cụm "có ý nghĩa thống kê"/"đầu tiên" trong tài liệu bài.
**Gate người:** FAIL nào cũng phải xử lý hoặc ghi vào Limitations.

---

## 7. Thứ tự cắt khi trễ

Cắt theo thứ tự: jitter (T05.4) → YOLOv5su → MLP → Mondrian → eps ablation (T09.3) → Van/Truck. Không cắt: pre-register (T04A, T08A), freeze (T11), quy trình T (T10, T12), audit (T18). Cross-fitting và ablation imgsz 960 chuyển sang hướng phát triển.
