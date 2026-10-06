# AGENT_RULES.md — luật cố định cho coding agent

Đọc trước MỌI tác vụ. Tác vụ nằm trong `docs/ANTIGRAVITY_TASKS.md`; mỗi lần chỉ làm đúng một tác vụ.
Nguồn chuẩn: `docs/KE_HOACH_V4.md` (thiết kế) và `docs/NHAT_KY_QUYET_DINH.md` (quyết định). Hai file lệch nhau thì nhật ký thắng. Tác vụ lệch nhật ký thì DỪNG và báo (mục 7), không tự chọn.

## 1. Vùng cấm

1. **Split T.** Code mới (pipeline, script phân tích, notebook) không được đọc, suy luận hay nhập T dưới bất kỳ hình thức nào: không `load_split(..., "T", allow_test=True)`, không đặt `ALLOW_TEST_SPLIT`, không sửa guard. Test/verify có sẵn (`test_splits.py`, `verify_data.py`) chỉ đếm frame/kiểm hash, giữ nguyên. Ngoại lệ duy nhất: `scripts/run_final_T.py` do con người chạy (T12).
2. **File chỉ đọc** (không sửa, không ghi đè): `splits/**` (cả `superseded_v1/`), `configs/geometry_params.yaml`, `configs/geometry_priors.yaml`, `configs/detector/*`, `runs/detector/**`, mọi file đã có trong `results/predictions/`. Thêm file mới thì được.
3. **Chỉ được thêm, không đổi hành vi:** `src/geometry/*`, `src/detection/matching.py`, `src/utils/*`, `src/evaluation/eval.py`, `src/evaluation/metrics.py`. Test cũ phải tiếp tục xanh; không nới ngưỡng test.
4. **Feature guard (D11).** Không sửa `DEFAULT_FEATURE_WHITELIST`, `FORBIDDEN_GT_*` trong `src/residual/feature_extractor.py`, không nới `tests/test_feature_guard.py`. Đặc trưng dẫn xuất ngoài whitelist chỉ khi tác vụ nói rõ, kèm guard test riêng.
5. **Git.** Không `commit`, `tag`, `push`, `reset`, `clean`, `checkout --`. Đề xuất lệnh + message, con người chạy.
6. **Môi trường.** Không cài package, không truy cập mạng, không tải dữ liệu. Thiếu gói thì dừng và báo.
7. **Rò rỉ GT.** Ma trận đặc trưng chỉ lấy từ `*_detections.parquet`. `z_gt`, `matched_iou`, `status`, `difficulty`, `truncated`, `occluded`, `alpha` chỉ nằm trong file/khung "eval", join sau khi đặc trưng đã tính. `alpha` chỉ dùng để nhóm phân tích.
8. **Chọn cấu hình.** Chỉ dùng OOF của B (D24). C chỉ để conformalize (và coverage mô tả ở dev). Không chọn gì bằng C hay T.
9. **Không "sửa cho qua".** Gate không đạt, test đỏ, số lệch thì báo đúng như vậy. Không đổi ngưỡng/eps/seed/bộ lọc để đạt.

## 2. Thiết kế cần nhớ (khỏi đọc lại cả kế hoạch)

- **Dữ liệu:** KITTI Object, chia theo drive, `splits-v2` đóng băng: A 3740 · V 374 · B 1499 · C 766 · T 1102 frame. Số drive có Car Hard: B 12, C 10, T 10 (dưới `MIN_CLUSTERS_WARN = 20`).
- **Vai trò tập:** A fine-tune detector + prior; V chọn checkpoint/ngưỡng conf; B fit hợp nhất/residual/quantile; C chỉ conformalize; T chạy một lần.
- **Target:** Z = `location_z` (không Euclid). Quần thể: Car, Hard. Dải: 0–10, 10–20, 20–30, 30–50, >50 (mở), kèm hàng ">30"; cờ `*` khi n < 100. C không có xe >50 m.
- **Detector:** YOLOv8s / YOLO11s / YOLOv5su, `last.pt` epoch 100 (D12), imgsz 640, FP32. Ngưỡng conf: 0.79 / 0.70 / 0.74 (D6). Matching Greedy IoU 0.5 (D15, D16).
- **Hình học (`geometry-v2`):** W_eff 2.6184, H_obj 1.6797, H_cam 2.0422, δ −4.6782 px, eps 2 px. Trọng số fit trên GT-bbox: [0.0807, 0.6634, 0.2560]. Ba cue Z_w, Z_h, Z_g, hợp nhất log-space theo hiệp phương sai.
- **Pipeline:** bbox → cue → Z_d → Z_base (Z_d nếu ≥ 1 cue hợp lệ, ngược lại Z_e, D13) → r̂ → Ẑ = Z_base·exp(r̂) → CQR trên r = ln Z_gt − ln Z_base.
- **Đã chốt đáng nhớ:** D11 (không GT trong đặc trưng) · D13 (fallback, pattern `000`) · D16b (Z_e OOF trên B) · D17 (refit Σ theo detector, giữ cột trọng số GT-bbox) · D18/D20 (báo pooled + macro; không "có ý nghĩa thống kê" khi ≤ 12 cụm) · D19 (θ = min(|α|, π − |α|); |α| ≈ π/2 là đầu/đuôi) · D21 (GT đo tâm xe, cue đo mặt gần ⇒ bias âm ở gần) · D22 (3 artifact parquet).
- **Đang chờ duyệt:** D23–D30 trong `docs/ANTIGRAVITY_TASKS.md` §4. Chỉ coi là ràng buộc khi đã ghi ✅.

## 3. Quy trình mỗi tác vụ

1. Đọc đúng các file "Đọc trước" của tác vụ. Đưa plan ngắn (file tạo/sửa, lệnh sẽ chạy) và chờ duyệt.
2. Viết code + test trước khi chạy trên dữ liệu thật. Tác vụ có bước "pre-register": viết file cấu hình, DỪNG, chờ con người commit/tag rồi mới chạy tiếp.
3. Chạy `pytest -q` toàn bộ, không chỉ test mới.
4. Viết "Báo cáo tác vụ" (mục 8). Không tự sang tác vụ khác.

## 4. Chuẩn code

Python, type hint, `from __future__ import annotations`, `pathlib`. Logic trong `src/`, CLI mỏng trong `scripts/`. Test đơn vị cho mọi phần xử lý dữ liệu/đánh giá, ưu tiên dữ liệu giả nhỏ. Windows: UTF-8 stdout (`sys.stdout.reconfigure`), Ultralytics `workers=0`. XGBoost/sklearn: đặt `random_state`/`seed`, cố định `n_jobs` và ghi vào log. Mọi bảng kết quả có n, n_valid, valid_frac, low_n.

## 5. Log tái lập

Mỗi script phân tích/huấn luyện ghi 1 dòng vào `runs/pipeline_log.jsonl` bằng `make_log_record` / `append_jsonl` (trong `eval.py`), kèm: seed, hash split liên quan, SHA checkpoint (nếu dùng), git commit + dirty, tham số chính, đường dẫn đầu ra.

## 6. Ngôn ngữ kết quả

Báo cáo song song pooled và macro theo drive. CI từ cluster bootstrap ghi "CI thô (k cụm)" khi k < 20; không viết "có ý nghĩa thống kê", "chứng minh", "đầu tiên". Không xếp hạng detector khi CI chồng lấn hoặc chứa 0. Không bịa trích dẫn: thiếu nguồn thì ghi `[CẦN TRÍCH DẪN]`.

## 7. Dừng và hỏi khi

Hash split / SHA checkpoint không khớp · số TP hoặc n không khớp số đã có · cần sửa file ở mục 1.2–1.4 · cần đụng T · gate/test đỏ cần quyết định · tác vụ mâu thuẫn nhật ký · bước chạy ước tính > 30 phút mà chưa báo.

## 8. Mẫu "Báo cáo tác vụ"

```
## Báo cáo tác vụ Txx
1. Đã làm: (mỗi file tạo/sửa một dòng)
2. Kiểm thử: (lệnh + kết quả, số test pass/fail)
3. Số liệu chính: (bảng ngắn; ghi n, n_valid, số cụm k)
4. Sai khác so với tác vụ / điều bất ngờ:
5. Việc cần người làm: (diff cần review, commit message đề xuất, tag đề xuất)
6. Ứng viên quyết định cho nhật ký: (D-mới, 1–2 dòng mỗi cái)
7. Rủi ro / câu hỏi mở:
```
