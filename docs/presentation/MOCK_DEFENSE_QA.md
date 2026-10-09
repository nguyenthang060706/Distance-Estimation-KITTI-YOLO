# Bộ Câu Hỏi Phản Biện Giả Lập & Kịch Bản Trả Lời (Mock Defense Q&A)
## Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors
**Môn học:** DSR301m — Fall 2026 | **Tài liệu hỗ trợ phản biện trước Hội đồng**

Tài liệu này tổng hợp 7 câu hỏi phản biện chuyên sâu và gai góc nhất mà Hội đồng chấm đề tài có khả năng chất vấn, kèm theo kịch bản trả lời sắc bén dựa trên bằng chứng dữ liệu thực nghiệm (Data-Driven Answers) và các mã quyết định đã chốt trong `NHAT_KY_QUYET_DINH.md`.

---

### Câu Hỏi 1: Về Tính Cần Thiết Của Mô Hình Lai (Hybrid Value)
> **Hội đồng hỏi:**  
> *"Nhóm nghiên cứu dành rất nhiều công sức xây dựng mô hình lai Hybrid (f) kết hợp hình học pinhole và XGBoost. Tuy nhiên, khi nhìn vào Bảng 3 trên tập kiểm định Split T, sai số AbsRel của mô hình Residual (f) là 0.0463, trong khi mô hình Hồi quy trực tiếp từ bounding box Direct Model (e) không cần hình học cũng đạt tới 0.0465. Hiệu số chênh lệch chỉ là -0.0002 và khoảng tin cậy Bootstrap chứa 0. Vậy mô hình hình học có thực sự cần thiết không, hay chỉ làm phức tạp hóa pipeline?"*

**Kịch bản trả lời (Dẫn chứng Quyết định D78 & Limitation 8):**
1. **Thừa nhận trung thực kết quả thực nghiệm:**  
   *"Kính thưa Thầy/Cô, nhóm xin hoàn toàn đồng ý và xác nhận phát hiện thực nghiệm này. Trên 10 cụm drive của Split T, kiểm định Paired Cluster Bootstrap 95% CI cho hiệu số $\Delta(f - e)$ là $[-0.0012, +0.0023]$, hoàn toàn bao hàm số 0. Do đó, nhóm không hề tuyên bố mô hình lai vượt trội về mặt số học so với hồi quy trực tiếp."*
2. **Khẳng định 3 giá trị cốt lõi phi số học của Mô hình Lai:**  
   *"Tuy nhiên, giá trị của mô hình lai trong hệ thống ADAS/tự hành an toàn không nằm ở vài phần vạn sai số, mà nằm ở 3 trụ cột mang tính nguyên lý:*
   - **Thứ nhất là Tính khả giải thích vật lý (Physical Interpretability):** *Mô hình Direct (e) là một hàm hộp đen phi tuyến, khi xe bị che khuất hoặc có biến dạng thị giác lạ, mô hình có thể ngoại suy ra khoảng cách hoàn toàn vô lý. Ngược lại, mô hình lai luôn neo chặt vào nghiệm quang học $Z_d$ có cơ sở vật lý rõ ràng.*
   - **Thứ hai là Khả năng Phân rã Sai số (Error Decomposition):** *Chỉ khi có mô hình hình học, chúng ta mới có thể tách bạch được bao nhiêu phần sai số đến từ giả định vật lý (kích thước xe, mặt đường) và bao nhiêu phần đến từ rung lắc bounding box của detector.*
   - **Thứ ba là Cơ chế Suy thoái Êm dịu (Graceful Fallback):** *Hình học pinhole đóng vai trò như một lớp bảo vệ. Nếu mạng nơ-ron gặp sự cố, hệ thống vẫn có thể lùi về nghiệm hình học thuần túy (AbsRel ~6.4%) để phanh khẩn cấp thay vì sập hoàn toàn."*

---

### Câu Hỏi 2: Về Độ Phủ Conformal Vượt Mức (Over-Coverage Phénomène)
> **Hội đồng hỏi:**  
> *"Nhóm đặt mức tin cậy danh nghĩa cho Conformal Quantile Regression (CQR) là 90% ($\alpha = 0.1$), nhưng trên Split T độ phủ thực tế lại lên tới 96.39% – 97.14%. Tại sao độ phủ lại lệch nhiều như vậy? Phải chăng mô hình đang quá bảo thủ (over-conservative) và làm nở rộng khoảng tin cậy không cần thiết?"*

**Kịch bản trả lời (Dẫn chứng Quyết định D79, D87 & Limitation 10):**
1. **Lý giải bằng sự dịch chuyển độ khó giữa các cụm hành trình:**  
   *"Kính thưa Thầy/Cô, đây là một phát hiện hậu nghiệm (post-hoc discovery) rất giá trị của đề tài. Về mặt lý thuyết, CQR bảo đảm độ phủ biên 90% dưới giả định tính khả hoán (exchangeability). Tuy nhiên, trên dữ liệu giao thông thực tế KITTI, các ảnh thuộc cùng một drive có tương quan chuỗi rất mạnh.*
   *Khi tiến hành chẩn đoán phân kỳ phân bố Kolmogorov-Smirnov giữa tập hiệu chuẩn C và tập kiểm định T ($D_{\text{KS}} \approx 0.14\text{--}0.16$), nhóm phát hiện Split C tập trung 2 drive khó (`drive_0057` và `drive_0004`) chiếm tới 39.3% mẫu, khiến sai số log-residual trung bình trên C cao hơn T ($|r| \approx 0.083$ trên C so với $0.067$ trên T)."*
2. **Ý nghĩa an toàn trong kỹ thuật ô tô:**  
   *"Do tập hiệu chuẩn C có độ khó cao hơn, ngưỡng không tuân thủ $\hat{Q}$ tính ra bị nới rộng một cách tự nhiên. Khi áp ngưỡng này sang tập T có điều kiện dễ hơn, độ phủ thực nghiệm tăng lên 96–97%. Trong các ứng dụng an toàn ô tô (ADAS), việc một khoảng tin cậy có tính chất bảo thủ ngoài mẫu (conservative over-coverage) là hoàn toàn chấp nhận được và an toàn hơn nhiều so với việc bị under-coverage (thiếu độ phủ)."*

---

### Câu Hỏi 3: Về Thuật Toán Matching (Greedy vs Hungarian)
> **Hội đồng hỏi:**  
> *"Tại sao nhóm lại chọn thuật toán Greedy Matching theo confidence giảm dần mà không sử dụng thuật toán Hungarian (Kuhn-Munkres) tối ưu toàn cục như một số bài báo trước đây?"*

**Kịch bản trả lời (Dẫn chứng Quyết định D15 & D16):**
1. **Tính chất bảo toàn phân cấp (Nested Filtering Invariance):**  
   *"Kính thưa Thầy/Cô, thuật toán Hungarian tối ưu hóa tổng IoU trên toàn bộ ma trận bipartite, điều này dẫn đến việc gán nhãn của một detection phụ thuộc vào toàn bộ tập ứng viên xung quanh. Khi chúng ta thay đổi ngưỡng confidence (ví dụ từ sàn 0.05 lên ngưỡng tối ưu F1 `pass_thr = 0.70`), Hungarian có thể gán lại các cặp ghép hoàn toàn khác nhau, phá vỡ tính phân cấp của pipeline.*
   *Ngược lại, thuật toán Greedy Matching theo confidence giảm dần có tính chất toán học độc lập: Trạng thái ghép của một detection có điểm số cao chỉ phụ thuộc vào các detection cao hơn nó. Do đó, việc ghép một lần ở sàn 0.05 rồi lọc ngưỡng sau tương đương 100% với việc lọc ngưỡng rồi mới ghép. Điều này bảo đảm tính công bằng tuyệt đối khi so sánh các ngưỡng confidence khác nhau."*

---

### Câu Hỏi 4: Về Trọng Số Chiều Rộng Bị Kẹp Về 0
> **Hội đồng hỏi:**  
> *"Tại sao khi huấn luyện mô hình hợp nhất trên nhãn Ground Truth thì trọng số chiều rộng $w_w$ dương (~0.08), nhưng khi áp dụng lên bounding box của detector thì $w_w$ lại bị kẹp về 0.0000? Phải chăng cue chiều rộng hoàn toàn vô dụng?"*

**Kịch bản trả lời (Dẫn chứng Quyết định D19 & D31):**
1. **Bản chất biến dạng hình học khi xe quay ngang:**  
   *"Kính thưa Thầy/Cô, camera chụp xe từ nhiều góc nhìn $\theta$. Khi nhìn thẳng đầu hoặc đuôi xe, chiều rộng bounding box phản ánh đúng bề ngang xe (~1.8m). Nhưng khi xe rẽ hoặc đi ngang (góc $\theta < 30^\circ$), chiều rộng bounding box trên ảnh 2D phình to tương ứng với **chiều dài thân xe (~4.5m)**, làm công thức pinhole $Z_w$ ước lượng thiếu nghiêm trọng (AbsRel dọt lên 42.4%).*
   *Khi dùng nhãn GT hoàn hảo, sự bù trừ hiệp phương sai âm giữa chiều rộng và chiều cao cho phép $w_w$ giữ giá trị nhỏ 0.08. Nhưng khi chuyển sang detector thực tế, nhiễu cạnh biên và rung lắc 2D làm phương sai sai số chiều rộng $\Sigma_{ww}$ tăng gấp 10 lần chiều cao $\Sigma_{hh}$. Thuật toán Non-Negative Least Squares (NNLS) đã nhận diện đúng sự suy thoái này và tự động kẹp $w_w \to 0$ để bảo vệ mô hình không bị khuếch đại nhiễu."*

---

### Câu Hỏi 5: Về Sai Số Âm Hệ Thống Ở Cự Ly Gần 0–10m
> **Hội đồng hỏi:**  
> *"Tại sao ở cự ly gần 0–10m, mô hình hình học thuần túy $Z_d$ lại có sai số âm hệ thống khoảng -1.0 mét (ước lượng gần hơn thực tế) ngay cả trên những xe không hề bị chạm viền ảnh (Pattern 111)?"*

**Kịch bản trả lời (Dẫn chứng Quyết định D21 & D92):**
1. **Lệch tâm vật lý 3D giữa mặt cản sau và tâm hộp xe:**  
   *"Kính thưa Thầy/Cô, Ground Truth của bộ dữ liệu KITTI đo tọa độ $Z$ đến **tâm hình học của hộp 3D (Bounding Box 3D Center)**. Trong khi đó, các cue thị giác 2D của camera (chiều cao, cạnh đáy tiếp đất) đo đến **mặt cản sau gần nhất của xe (Closest Surface)**.*
   *Khoảng cách từ mặt cản sau đến tâm xe là nửa chiều dài xe ($l/2 \approx 1.0\text{ m}$). Ở cự ly xa 30–50m, 1 mét này chỉ chiếm 2–3% nên không đáng kể. Nhưng ở cự ly gần 5–8m, 1 mét này tạo ra độ lệch âm hệ thống từ -11% đến -14%.*
   *Mô hình Residual XGBoost của nhóm đã học được chính xác cơ chế vật lý này và bù trừ thành công, đưa độ lệch trung vị (median bias) từ -10.9% về chỉ còn -0.27%."*

---

### Câu Hỏi 6: Về Nhãn PRELIMINARY-v2 Trong Benchmark Độ Trễ
> **Hội đồng hỏi:**  
> *"Kết quả đo độ trễ trên GPU đạt ~38 ms (26 FPS). Tại sao nhóm vẫn gắn nhãn PRELIMINARY-v2 mà không khẳng định luôn là đã đạt chuẩn thời gian thực ADAS thương mại?"*

**Kịch bản trả lời (Dẫn chứng Quyết định D98, D100, D104):**
1. **Cam kết liêm chính khoa học và giới hạn phần cứng:**  
   *"Kính thưa Thầy/Cô, nhóm giữ nhãn `PRELIMINARY-v2` vì hai lý do phương pháp luận trung thực:*
   - **Thứ nhất:** *Phép đo được thực hiện trên máy trạm laptop cá nhân (RTX 5060 Laptop GPU), chịu ảnh hưởng bởi hiện tượng điều tiết nhiệt độ (thermal throttling) và quản lý tiến trình của hệ điều hành Windows, chưa phải bo mạch nhúng ô tô chuyên dụng (như NVIDIA Drive Orin hay Jetson).*
   - **Thứ hai:** *Để tương thích tĩnh với ONNX Runtime, ảnh đầu vào được đệm vuông $640 \times 640$, chứa số pixel gấp $\approx 2.9$ lần so với letterbox thực tế của KITTI ($640 \times 224$). Do đó thời gian GPU thực tế khi triển khai tối ưu có thể còn nhanh hơn.*
   *Điểm mấu chốt nhóm khẳng định là: **Khâu xử lý hậu detector (hình học + residual + CQR) chỉ tốn 1.35 ms**, chiếm dưới 4.2% pipeline, chứng minh phương pháp đề xuất không gây nghẽn phần cứng."*

---

### Câu Hỏi 7: Về Đánh Giá Trên Tập Test Server KITTI
> **Hội đồng hỏi:**  
> *"Tại sao nhóm chỉ đánh giá trên 1.102 ảnh của Split T mà không nộp kết quả lên KITTI Test Server chính thức (7.518 ảnh)?"*

**Kịch bản trả lời (Dẫn chứng Kế hoạch v4 §2 & §4):**
1. **Giới hạn nhãn công khai của KITTI Test Server:**  
   *"Kính thưa Thầy/Cô, KITTI Test Server chính thức chỉ chấm điểm bài toán 2D/3D Object Detection (mAP và AP3D/AOS) thông qua server nộp bài tự động. Họ **không công khai nhãn 3D tọa độ $Z$** và **không có bảng xếp hạng riêng cho bài toán Ranging (Khoảng cách điểm Z)** với các chỉ số như AbsRel, RMSE hay Conformal Coverage.*
   *Do đó, thông lệ chuẩn mực của cộng đồng nghiên cứu monocular depth estimation quốc tế là chia tập 7.481 ảnh có nhãn thành các split độc lập theo drive (như split Chen et al. hoặc split tự dựng). Nhóm đã tự dựng `splits-v2` hoàn toàn rời rạc theo drive và đóng băng khóa `final_T.lock` chạy một lần duy nhất để bảo đảm tính khách quan tuyệt đối tương đương một server chấm độc lập."*
