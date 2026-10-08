# Physical Distance Bias Analysis & Decision D21 Verification (Split T)

> **Decision D21 Context:** Hypothesis regarding near-range visual bounding box center vs physical vehicle 3D center offset $\Delta Z = Z_{pred} - Z_{gt}$.
> Decision D84 specifies isolating **Pattern 111 without border touch** (`valid_w == 1 & valid_h == 1 & valid_g == 1`) in near range (0-10m) to decouple genuine physical bounding center shift from bbox clipping artifacts.

## Detector: `yolo11s_640`

### Signed Bias across Distance Bins
| Distance Bin (m) | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias | IQR Rel Bias |
|---|---|---|---|---|---|---|
| 0-10 | 261 | -0.006 | -0.061 | +0.0012 | -0.0076 | 0.0995 |
| 10-20 | 831 | -0.041 | -0.010 | -0.0029 | -0.0008 | 0.0686 |
| 20-30 | 804 | +0.066 | -0.046 | +0.0027 | -0.0019 | 0.0673 |
| 30-50 | 809 | +0.265 | +0.111 | +0.0073 | +0.0031 | 0.0763 |
| >50 | 7* | -5.664 | -5.653 | -0.1057 | -0.1124 | 0.0565 |
| >30 | 816 | +0.214 | +0.094 | +0.0063 | +0.0027 | 0.0770 |

### Near-Range (0-10m) Pattern 111 vs All Objects
| Stratum (0-10m) | $n$ | Mean Rel Bias | Median Rel Bias |
|---|---|---|---|
| All Detected Objects | 261 | +0.0012 | -0.0076 |
| Pattern 111 (No Border Cut) | 118 | -0.0025 | -0.0027 |

## Detector: `yolov8s_640`

### Signed Bias across Distance Bins
| Distance Bin (m) | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias | IQR Rel Bias |
|---|---|---|---|---|---|---|
| 0-10 | 266 | -0.063 | -0.095 | -0.0053 | -0.0108 | 0.0935 |
| 10-20 | 836 | -0.122 | -0.080 | -0.0085 | -0.0053 | 0.0751 |
| 20-30 | 780 | +0.036 | -0.072 | +0.0012 | -0.0030 | 0.0613 |
| 30-50 | 776 | +0.382 | +0.282 | +0.0103 | +0.0082 | 0.0732 |
| >50 | 2* | -3.606 | -3.606 | -0.0703 | -0.0703 | 0.0241 |
| >30 | 778 | +0.372 | +0.274 | +0.0101 | +0.0080 | 0.0733 |

### Near-Range (0-10m) Pattern 111 vs All Objects
| Stratum (0-10m) | $n$ | Mean Rel Bias | Median Rel Bias |
|---|---|---|---|
| All Detected Objects | 266 | -0.0053 | -0.0108 |
| Pattern 111 (No Border Cut) | 129 | -0.0106 | -0.0116 |

## Detector: `yolov5su_640`

### Signed Bias across Distance Bins
| Distance Bin (m) | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias | IQR Rel Bias |
|---|---|---|---|---|---|---|
| 0-10 | 264 | +0.017 | -0.057 | +0.0048 | -0.0067 | 0.0943 |
| 10-20 | 835 | -0.052 | -0.080 | -0.0038 | -0.0058 | 0.0703 |
| 20-30 | 781 | +0.043 | -0.052 | +0.0013 | -0.0020 | 0.0658 |
| 30-50 | 785 | +0.026 | +0.028 | +0.0015 | +0.0008 | 0.0755 |
| >50 | 9* | -7.344 | -8.072 | -0.1386 | -0.1497 | 0.0339 |
| >30 | 794 | -0.057 | -0.041 | -0.0001 | -0.0012 | 0.0774 |

### Near-Range (0-10m) Pattern 111 vs All Objects
| Stratum (0-10m) | $n$ | Mean Rel Bias | Median Rel Bias |
|---|---|---|---|
| All Detected Objects | 264 | +0.0048 | -0.0067 |
| Pattern 111 (No Border Cut) | 127 | -0.0061 | -0.0090 |

## Verification Synthesis (Decisions D21 & D84)

Quan sát thực nghiệm từ bảng số liệu trên Split T:
1. **Xu hướng độ lệch có dấu $\Delta Z$ theo cự ly:**
   - Ở cự ly gần (0-10m): Mean Rel Bias dao động từ -0.53% đến +0.48%, Median Rel Bias từ -1.08% đến -0.67% across 3 detectors. Sai số tuyệt đối trung vị nhỏ hơn 0.1m.
   - Ở cự ly trung bình (10-30m): Độ lệch tương đối thực tế duy trì rất nhỏ quanh mức 0 (Median Rel Bias từ -0.58% đến -0.08%).
   - Ở cự ly xa (>50m, $n < 10$): Độ lệch có dấu mang giá trị âm rõ rệt (Mean Bias từ -3.6m đến -7.3m, Rel Bias -7.0% đến -14.9%) do kích thước bounding box suy biến về mức vài pixel và hiệu ứng hồi quy về giá trị trung vị của mô hình học máy.
2. **Ảnh hưởng của Pattern 111 không chạm biên (Decisions D21, D84):**
   - Ở cự ly 0-10m, khi lọc bỏ các bounding box bị chạm biên (chỉ giữ mẫu có đủ 3 cue hợp lệ $w, h, g$, chiếm ~45-48% số phát hiện ở dải này), Median Rel Bias nằm trong khoảng -1.16% đến -0.27% (sai lệch dưới 1.2% cự ly thực).
   - Kết quả thực nghiệm này cho thấy: độ lệch giữa tâm quang học bounding box và tâm vật lý 3D của xe không tạo ra sai lệch một chiều đáng kể trên mô hình hoàn chỉnh, và sự dao động sai số ở cự ly gần chủ yếu gắn liền với hiện tượng cắt xén biên ảnh (clipping) và che khuất.