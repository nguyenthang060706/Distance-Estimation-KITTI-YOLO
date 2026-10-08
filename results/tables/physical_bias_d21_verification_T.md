# Physical Distance Bias Analysis & Decision D21 Verification (Split T)

> **Decision D21 Context:** Hypothesis regarding near-range visual bounding box surface vs physical vehicle 3D center offset $\Delta Z = Z_{pred} - Z_{gt} \approx -l/2$.
> Decision D84 specifies isolating **Pattern 111 without border cut** (`valid_w == 1 & valid_h == 1 & valid_g == 1`) in near range (0-10m) based on border mask tolerance $\epsilon = 2\text{ px}$.
> To properly evaluate D21, bias must be measured on raw geometric cues ($z_w, z_h, z_g$), geometric fusion ($z_d$), and compared against residual model ($z_{\hat{f}}$).

## Detector: `yolo11s_640`

### 1. Residual Model ($z_{\hat{f}}$) Signed Bias across Distance Bins
| Distance Bin (m) | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias | IQR Rel Bias |
|---|---|---|---|---|---|---|
| 0-10 | 261 | -0.006 | -0.061 | +0.0012 | -0.0076 | 0.0995 |
| 10-20 | 831 | -0.041 | -0.010 | -0.0029 | -0.0008 | 0.0686 |
| 20-30 | 804 | +0.066 | -0.046 | +0.0027 | -0.0019 | 0.0673 |
| 30-50 | 809 | +0.265 | +0.111 | +0.0073 | +0.0031 | 0.0763 |
| >50 | 7* | -5.664 | -5.653 | -0.1057 | -0.1124 | 0.0565 |
| >30 | 816 | +0.214 | +0.094 | +0.0063 | +0.0027 | 0.0770 |

### 2. Near-Range (0–10m) Cue-Level Bias: Pattern 111 (No Border Cut, $\epsilon=2\text{ px}$) vs All Objects
| Stratum | Cue / Model | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias |
|---|---|---|---|---|---|---|
| **Pattern 111 (No Cut, $n=118$)** | Width cue ($z_w$) | 118 | -1.728 | -1.712 | -0.1915 | -0.1965 |
| | Height cue ($z_h$) | 118 | -1.271 | -1.209 | -0.1422 | -0.1422 |
| | Ground cue ($z_g$) | 118 | -0.371 | -0.406 | -0.0414 | -0.0489 |
| | Geometric Fused ($z_d$) | 118 | -1.015 | -0.995 | -0.1135 | -0.1159 |
| | Residual Model ($z_{\hat{f}}$) | 118 | -0.027 | -0.024 | -0.0025 | -0.0027 |
| **All Detected Objects ($n=261$)** | Width cue ($z_w$) | 194 | -1.733 | -1.717 | -0.2120 | -0.2182 |
| | Height cue ($z_h$) | 149 | -1.125 | -1.153 | -0.1273 | -0.1292 |
| | Ground cue ($z_g$) | 149 | -0.211 | -0.254 | -0.0223 | -0.0307 |
| | Geometric Fused ($z_d$) | 225 | -1.162 | -1.194 | -0.1470 | -0.1412 |
| | Residual Model ($z_{\hat{f}}$) | 261 | -0.006 | -0.061 | +0.0012 | -0.0076 |

## Detector: `yolov8s_640`

### 1. Residual Model ($z_{\hat{f}}$) Signed Bias across Distance Bins
| Distance Bin (m) | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias | IQR Rel Bias |
|---|---|---|---|---|---|---|
| 0-10 | 266 | -0.063 | -0.095 | -0.0053 | -0.0108 | 0.0935 |
| 10-20 | 836 | -0.122 | -0.080 | -0.0085 | -0.0053 | 0.0751 |
| 20-30 | 780 | +0.036 | -0.072 | +0.0012 | -0.0030 | 0.0613 |
| 30-50 | 776 | +0.382 | +0.282 | +0.0103 | +0.0082 | 0.0732 |
| >50 | 2* | -3.606 | -3.606 | -0.0703 | -0.0703 | 0.0241 |
| >30 | 778 | +0.372 | +0.274 | +0.0101 | +0.0080 | 0.0733 |

### 2. Near-Range (0–10m) Cue-Level Bias: Pattern 111 (No Border Cut, $\epsilon=2\text{ px}$) vs All Objects
| Stratum | Cue / Model | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias |
|---|---|---|---|---|---|---|
| **Pattern 111 (No Cut, $n=129$)** | Width cue ($z_w$) | 129 | -1.738 | -1.670 | -0.1945 | -0.1970 |
| | Height cue ($z_h$) | 129 | -1.254 | -1.241 | -0.1409 | -0.1452 |
| | Ground cue ($z_g$) | 129 | -0.289 | -0.403 | -0.0321 | -0.0431 |
| | Geometric Fused ($z_d$) | 129 | -0.996 | -1.003 | -0.1118 | -0.1220 |
| | Residual Model ($z_{\hat{f}}$) | 129 | -0.102 | -0.100 | -0.0106 | -0.0116 |
| **All Detected Objects ($n=266$)** | Width cue ($z_w$) | 201 | -1.742 | -1.711 | -0.2130 | -0.2159 |
| | Height cue ($z_h$) | 158 | -1.148 | -1.160 | -0.1300 | -0.1359 |
| | Ground cue ($z_g$) | 158 | -0.200 | -0.277 | -0.0207 | -0.0298 |
| | Geometric Fused ($z_d$) | 230 | -1.163 | -1.153 | -0.1463 | -0.1361 |
| | Residual Model ($z_{\hat{f}}$) | 266 | -0.063 | -0.095 | -0.0053 | -0.0108 |

## Detector: `yolov5su_640`

### 1. Residual Model ($z_{\hat{f}}$) Signed Bias across Distance Bins
| Distance Bin (m) | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias | IQR Rel Bias |
|---|---|---|---|---|---|---|
| 0-10 | 264 | +0.017 | -0.057 | +0.0048 | -0.0067 | 0.0943 |
| 10-20 | 835 | -0.052 | -0.080 | -0.0038 | -0.0058 | 0.0703 |
| 20-30 | 781 | +0.043 | -0.052 | +0.0013 | -0.0020 | 0.0658 |
| 30-50 | 785 | +0.026 | +0.028 | +0.0015 | +0.0008 | 0.0755 |
| >50 | 9* | -7.344 | -8.072 | -0.1386 | -0.1497 | 0.0339 |
| >30 | 794 | -0.057 | -0.041 | -0.0001 | -0.0012 | 0.0774 |

### 2. Near-Range (0–10m) Cue-Level Bias: Pattern 111 (No Border Cut, $\epsilon=2\text{ px}$) vs All Objects
| Stratum | Cue / Model | $n$ | Mean Bias (m) | Median Bias (m) | Mean Rel Bias | Median Rel Bias |
|---|---|---|---|---|---|---|
| **Pattern 111 (No Cut, $n=127$)** | Width cue ($z_w$) | 127 | -1.739 | -1.633 | -0.1939 | -0.1896 |
| | Height cue ($z_h$) | 127 | -1.244 | -1.209 | -0.1391 | -0.1376 |
| | Ground cue ($z_g$) | 127 | -0.303 | -0.377 | -0.0332 | -0.0432 |
| | Geometric Fused ($z_d$) | 127 | -0.948 | -0.923 | -0.1058 | -0.1094 |
| | Residual Model ($z_{\hat{f}}$) | 127 | -0.061 | -0.079 | -0.0061 | -0.0090 |
| **All Detected Objects ($n=264$)** | Width cue ($z_w$) | 197 | -1.770 | -1.738 | -0.2151 | -0.2228 |
| | Height cue ($z_h$) | 158 | -1.142 | -1.138 | -0.1290 | -0.1345 |
| | Ground cue ($z_g$) | 158 | -0.196 | -0.275 | -0.0195 | -0.0326 |
| | Geometric Fused ($z_d$) | 228 | -1.147 | -1.147 | -0.1435 | -0.1357 |
| | Residual Model ($z_{\hat{f}}$) | 264 | +0.017 | -0.057 | +0.0048 | -0.0067 |

## Verification Synthesis (Decisions D21 & D84)

Số liệu thực nghiệm trên Split T đối chiếu với giả thuyết D21:
1. **Độ lệch âm trên cue thô và mô hình hình học thuần ($z_d$):**
   - Ở cự ly 0–10m trên nhóm Pattern 111 không chạm biên (mask $\epsilon = 2\text{ px}$), mô hình hình học $z_d$ có độ lệch âm rõ rệt: Mean Bias dao động từ **-0.95m đến -1.02m**, Median Rel Bias từ **-10.9% đến -12.2%** across 3 detectors.
   - Các cue đơn lẻ cũng thể hiện độ lệch âm tương ứng: Cue bề rộng $z_w$ lệch -1.71m đến -1.74m (-19.0% đến -19.7%), cue chiều cao $z_h$ lệch -1.21m đến -1.25m (-13.8% đến -14.5%).
   - Phát hiện này **nhất quán với giả thuyết D21**: Bounding box thị giác đo đến mặt trước/gần của xe ($Z_{\text{surface}}$) thay vì tâm hộp 3D ($Z_{\text{center}}$), tạo ra độ lệch âm xấp xỉ nửa chiều dài xe $l/2 \approx 1.0\text{ m}$.
2. **Vai trò hấp thụ sai số của mô hình Residual ($z_{\hat{f}}$):**
   - Sau khi qua mô hình residual XGBoost, Median Rel Bias của $z_{\hat{f}}$ trên nhóm Pattern 111 giảm từ -11.6% xuống còn **-0.27% đến -1.16%** (Median Bias chỉ từ -0.02m đến -0.10m).
   - Điều này thể hiện rằng mô hình học máy dư (residual learning) đã hấp thụ thành công độ lệch tâm vật lý có hệ thống này của mô hình hình học.