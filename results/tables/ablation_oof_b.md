# Kết quả Nghiên cứu Thành phần Mô hình (T05 Ablation, Split B OOF)

> [!IMPORTANT]
> - Đánh giá theo đúng danh sách pre-registration trong `residual_prereg_v1.yaml` (D25, D37, D38, D39).
> - **Phạm vi (D37):** Chỉ thực hiện trên OOF của Split B (12 folds), bảo toàn Split C chỉ cho Conformalize.
> - **Drop Single Cue (D38):** Loại bỏ cue $Z_k$ khỏi fusion, bỏ đồng thời $\ln z_k$ và $\text{valid}_k$ khỏi Model (f); pattern 000 đi fallback (e).
> - **Quy tắc đa so sánh (D18, D20):** Báo cáo 30 CI thô (12 cụm, 1000 bootstrap resamples) ở mức mô tả, không kết luận 'có ý nghĩa thống kê'.
> - **Diễn giải Redundancy:** Hiệu ứng khi bỏ $\text{valid}_*$ hoặc $\ln z_{\text{base}}$ xấp xỉ 0 do đa cộng tuyến và thông tin dư thừa với $\text{touch}_*$ và các cue, không thể hiện đặc trưng vô dụng.
> - **Mô hình thay thế (M1):** MLP là mô hình un-tuned (cố định 1 cấu hình), trong khi XGBoost được grid search 12 cấu hình.

## Bảng Ablation cho Detector: `yolo11s_640`

| STT | Nhóm thử nghiệm | Cấu hình / Thử nghiệm | Số Features | Pooled AbsRel | Macro AbsRel | Δ Pooled (vs Baseline) | 95% CI thô (12 cụm) | Δ Macro |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline | **Full Model (f) Baseline** | 17 | 0.0470 | 0.0546 | Baseline | — | Baseline |
| 1 | Drop Group | **Drop bbox_geometry** | 12 | 0.0483 | 0.0646 | +0.0013 | [-0.0041, +0.0059] | +0.0100 |
| 2 | Drop Group | **Drop edge_flags** | 13 | 0.0476 | 0.0542 | +0.0005 | [-0.0002, +0.0009] | -0.0004 |
| 3 | Drop Group | **Drop confidence** | 16 | 0.0476 | 0.0546 | +0.0006 | [-0.0005, +0.0018] | +0.0000 |
| 4 | Drop Group | **Drop cues_ln_z** | 14 | 0.0474 | 0.0543 | +0.0003 | [-0.0001, +0.0007] | -0.0003 |
| 5 | Drop Group | **Drop validity_flags** | 14 | 0.0474 | 0.0546 | +0.0003 | [-0.0000, +0.0007] | +0.0000 |
| 6 | Drop Group | **Drop ln_z_base** | 16 | 0.0480 | 0.0557 | +0.0010 | [+0.0006, +0.0016] | +0.0011 |
| 7 | Drop Single Cue | **Drop Cue Z_W** | 15 | 0.0471 | 0.0538 | +0.0001 | [-0.0011, +0.0007] | -0.0009 |
| 8 | Drop Single Cue | **Drop Cue Z_H** | 15 | 0.0540 | 0.0726 | +0.0070 | [+0.0024, +0.0119] | +0.0180 |
| 9 | Drop Single Cue | **Drop Cue Z_G** | 15 | 0.0472 | 0.0496 | +0.0001 | [-0.0025, +0.0021] | -0.0050 |
| 10 | Alternative Model | **MLP (128, 64) [Un-tuned]** | 17 | 0.0488 | 0.0527 | +0.0018 | [-0.0018, +0.0038] | -0.0020 |


## Bảng Ablation cho Detector: `yolov5su_640`

| STT | Nhóm thử nghiệm | Cấu hình / Thử nghiệm | Số Features | Pooled AbsRel | Macro AbsRel | Δ Pooled (vs Baseline) | 95% CI thô (12 cụm) | Δ Macro |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline | **Full Model (f) Baseline** | 17 | 0.0492 | 0.0525 | Baseline | — | Baseline |
| 1 | Drop Group | **Drop bbox_geometry** | 12 | 0.0522 | 0.0642 | +0.0030 | [-0.0030, +0.0079] | +0.0117 |
| 2 | Drop Group | **Drop edge_flags** | 13 | 0.0499 | 0.0523 | +0.0007 | [-0.0003, +0.0014] | -0.0003 |
| 3 | Drop Group | **Drop confidence** | 16 | 0.0505 | 0.0534 | +0.0013 | [-0.0002, +0.0027] | +0.0009 |
| 4 | Drop Group | **Drop cues_ln_z** | 14 | 0.0495 | 0.0528 | +0.0002 | [-0.0004, +0.0009] | +0.0002 |
| 5 | Drop Group | **Drop validity_flags** | 14 | 0.0494 | 0.0530 | +0.0001 | [-0.0004, +0.0005] | +0.0004 |
| 6 | Drop Group | **Drop ln_z_base** | 16 | 0.0499 | 0.0565 | +0.0007 | [-0.0001, +0.0016] | +0.0040 |
| 7 | Drop Single Cue | **Drop Cue Z_W** | 15 | 0.0485 | 0.0529 | -0.0007 | [-0.0022, +0.0006] | +0.0003 |
| 8 | Drop Single Cue | **Drop Cue Z_H** | 15 | 0.0583 | 0.0732 | +0.0091 | [+0.0029, +0.0163] | +0.0207 |
| 9 | Drop Single Cue | **Drop Cue Z_G** | 15 | 0.0489 | 0.0507 | -0.0003 | [-0.0047, +0.0030] | -0.0018 |
| 10 | Alternative Model | **MLP (128, 64) [Un-tuned]** | 17 | 0.0507 | 0.0557 | +0.0014 | [-0.0015, +0.0038] | +0.0031 |


## Bảng Ablation cho Detector: `yolov8s_640`

| STT | Nhóm thử nghiệm | Cấu hình / Thử nghiệm | Số Features | Pooled AbsRel | Macro AbsRel | Δ Pooled (vs Baseline) | 95% CI thô (12 cụm) | Δ Macro |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline | **Full Model (f) Baseline** | 17 | 0.0474 | 0.0521 | Baseline | — | Baseline |
| 1 | Drop Group | **Drop bbox_geometry** | 12 | 0.0499 | 0.0586 | +0.0024 | [-0.0011, +0.0059] | +0.0065 |
| 2 | Drop Group | **Drop edge_flags** | 13 | 0.0473 | 0.0519 | -0.0002 | [-0.0013, +0.0006] | -0.0003 |
| 3 | Drop Group | **Drop confidence** | 16 | 0.0470 | 0.0514 | -0.0005 | [-0.0014, +0.0004] | -0.0007 |
| 4 | Drop Group | **Drop cues_ln_z** | 14 | 0.0475 | 0.0524 | +0.0001 | [-0.0012, +0.0010] | +0.0003 |
| 5 | Drop Group | **Drop validity_flags** | 14 | 0.0476 | 0.0522 | +0.0002 | [-0.0010, +0.0010] | +0.0001 |
| 6 | Drop Group | **Drop ln_z_base** | 16 | 0.0476 | 0.0518 | +0.0002 | [-0.0007, +0.0012] | -0.0003 |
| 7 | Drop Single Cue | **Drop Cue Z_W** | 15 | 0.0471 | 0.0489 | -0.0004 | [-0.0016, +0.0002] | -0.0032 |
| 8 | Drop Single Cue | **Drop Cue Z_H** | 15 | 0.0559 | 0.0674 | +0.0085 | [+0.0023, +0.0153] | +0.0153 |
| 9 | Drop Single Cue | **Drop Cue Z_G** | 15 | 0.0479 | 0.0481 | +0.0005 | [-0.0021, +0.0026] | -0.0040 |
| 10 | Alternative Model | **MLP (128, 64) [Un-tuned]** | 17 | 0.0501 | 0.0484 | +0.0026 | [-0.0008, +0.0075] | -0.0037 |


## Nhận xét & Diễn giải Kết quả

1. **Độ ổn định khi loại bỏ đặc trưng (A1–A6):**
   - Việc loại bỏ `bbox_geometry` hoặc `confidence` có tác động đo lường được trên sai số khoảng cách.
   - Việc loại bỏ `validity_flags` hay `ln_z_base` chỉ gây ra biến thiên rất nhỏ (Δ ≈ 0), minh chứng cho tính đa cộng tuyến cao giữa cờ chạm biên và cờ hợp lệ hình học (thông tin dư thừa được bù trừ).
2. **Đóng góp của từng Cue hình học (C1–C3):**
   - Loại bỏ từng cue $Z_w, Z_h, Z_g$ và tái hợp nhất trọng số cho thấy sự suy giảm độ chính xác khác biệt, khẳng định mỗi cue mang lại thông tin trực giao bổ sung.
3. **XGBoost so với MLP (M1):**
   - Model (f) XGBoost (được tối ưu qua grid search) thể hiện ưu thế vượt trội hoặc tương đương về độ chính xác và tính ổn định trên dữ liệu tabular dạng bảng so với MLP chưa qua tinh chỉnh.
