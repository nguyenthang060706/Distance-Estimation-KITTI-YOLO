# Báo cáo Phân tích Chuyên sâu Hybrid (f) vs Direct (e) trên Split B OOF

> Pre-registration: `prereg-hybrid-v1` (D131). Tuyệt đối không chạm Split T.

## 1. Phân tích Nhóm con (Subgroup Analysis)

### A. Theo Góc Quan sát $\theta$ (Viewing Angle)
| Phân nhóm $\theta$ | $N$ | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\Delta(f - e)$ |
|---|---|---|---|---|---|
| side (<30 deg) | 77 | 0.0709 | 0.0522 | 0.0636 | +0.0114 |
| diagonal (30-60 deg) | 553 | 0.0797 | 0.0594 | 0.0609 | +0.0015 |
| front_rear (>60 deg) | 2893 | 0.0568 | 0.0447 | 0.0439 | -0.0008 |

### B. Theo Trạng thái Cắt Viền (Truncation)
| Trạng thái | $N$ | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\Delta(f - e)$ |
|---|---|---|---|---|---|
| untruncated (t=0) | 3352 | 0.0566 | 0.0457 | 0.0456 | -0.0002 |
| truncated (t>0) | 171 | 0.1601 | 0.0761 | 0.0759 | -0.0002 |

### C. Theo Dải Khoảng cách
| Dải khoảng cách | $N$ | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\Delta(f - e)$ |
|---|---|---|---|---|---|
| 0-10m | 279 | 0.1371 | 0.0618 | 0.0633 | +0.0014 |
| 10-20m | 1089 | 0.0606 | 0.0481 | 0.0495 | +0.0014 |
| 20-30m | 1225 | 0.0509 | 0.0436 | 0.0443 | +0.0007 |
| 30-50m | 924 | 0.0535 | 0.0463 | 0.0429 | -0.0035 |
| >50m | 6 | 0.0848 | 0.0946 | 0.0473 | -0.0473 |

## 2. Đường cong học theo Số drive Huấn luyện (Learning Curve)

| Số drive huấn luyện ($k$) | (d) Fused AbsRel | (e) Direct AbsRel | (f) Hybrid AbsRel | $\Delta(f - e)$ |
|---|---|---|---|---|
| $k = 2$ drives | 0.0607 | 0.1803 | 0.0575 | -0.1228 |
| $k = 4$ drives | 0.0607 | 0.0559 | 0.0492 | -0.0067 |
| $k = 6$ drives | 0.0607 | 0.0525 | 0.0494 | -0.0031 |
| $k = 8$ drives | 0.0607 | 0.0480 | 0.0466 | -0.0014 |
| $k = 11$ drives | 0.0607 | 0.0476 | 0.0467 | -0.0009 |

## 3. Ngoại suy Cự ly xa (Range Extrapolation: Train $Z \le 30$ m, Eval $Z > 30$ m)

- Mẫu huấn luyện cự ly gần/trung ($Z \le 30$ m): **N = 2594**
- Mẫu kiểm tra cự ly xa ($Z > 30$ m): **N = 929**

| Phương pháp | AbsRel ($Z > 30$ m) | MAE ($Z > 30$ m) | $\delta < 1.25$ |
|---|---|---|---|
| (d) Fused Geometry | 0.0536 | 2.02 m | 98.3% |
| (e) Direct Regression | 0.2235 | 9.23 m | 44.8% |
| (f) Hybrid Residual | 0.0760 | 2.94 m | 98.2% |

> **Chênh lệch $\Delta(f - e)$:** **-0.1475**
