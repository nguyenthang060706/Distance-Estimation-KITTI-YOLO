# ADIP 2026 Submission & Formatting Guide
## 8th Asia Digital Image Processing Conference (Tokyo, Japan) — SPIE Proceedings

---

## 1. Thông Tin Hội Nghị & Quy Chuẩn Bài Báo

- **Hội nghị:** 8th Asia Digital Image Processing Conference (ADIP 2026)
- **Địa điểm:** Tokyo, Japan
- **Trang chủ:** [https://adip.org/index.html](https://adip.org/index.html)
- **Trang nộp bài:** [https://adip.org/submission.html](https://adip.org/submission.html)
- **Nhà xuất bản:** International Conference Proceedings (SPIE Proceedings format)
- **Template bài viết chính thức:**
  - Full Paper Template: [`template-SPIE.doc`](https://adip.org/template-SPIE.doc)
  - Abstract Template: [`Template-Abstract.doc`](https://adip.org/Template-Abstract.doc)
- **Quy định số trang:** **8 – 14 trang** (phí đăng ký chuẩn bao gồm tối đa 14 trang, tính cả hình ảnh, bảng biểu và tài liệu tham khảo).
- **Ngôn ngữ:** 100% tiếng Anh học thuật chuẩn mực (academic English).

---

## 2. Kế Hoạch Phân Bổ Dung Lượng Số Trang (Page Budget: ~11–12 Trang)

Bản thảo SPIE sử dụng định dạng 1 cột (single-column), font Times New Roman 10pt (hoặc 11pt cho body text theo template). Dưới đây là sơ đồ dàn trang tối ưu để bài báo nằm chính xác trong khoảng **11 – 12 trang** (chuẩn tuyệt đối trong giới hạn 8–14 trang):

| Trang | Nội dung chi tiết | Bảng & Hình ảnh tương ứng |
|:---:|---|---|
| **Trang 1** | Tiêu đề, Khối tác giả & Đơn vị (Affiliations), Abstract (~250 từ), Keywords, **1. INTRODUCTION** (Mở đầu, bài toán ADAS/FCW) | --- |
| **Trang 2** | **1. INTRODUCTION** (3 Gaps nghiên cứu & 3 Đóng góp chính), **2. RELATED WORK & METHODOLOGICAL POSITIONING** | **Table 1** (Ma trận so sánh phương pháp đối chuẩn) |
| **Trang 3** | **3. PROPOSED METHODOLOGY** (3.1 Perspective Geometry Cues: $Z_w, Z_h, Z_g$, Boundary Masking; 3.2 OAS Covariance Shrinkage Fusion) | **Figure 1** (Sơ đồ kiến trúc tổng thể pipeline hybrid) |
| **Trang 4** | **3. PROPOSED METHODOLOGY** (3.3 Residual Calibration Model & Fallback Mechanism, 3.4 Conformalized Quantile Regression CQR) | Công thức toán học (1)–(8) |
| **Trang 5** | **4. DATASET AND EXPERIMENTAL SETUP** (4.1 Drive-Disjoint Dataset Partitioning `splits-v2`, Phân tích lệch phân phối $D_{\text{KS}} > 0.53$, 4.2 Huấn luyện detector YOLO) | **Table 2** (Bảng phân chia 5 tập A/V/B/C/T) & **Figure 2** (`fig_02_splits_spatial_distribution.png`) |
| **Trang 6** | **4.3 Evaluation Metrics and Target Scope** (Greedy matching, Car Hard TP), **5. EXPERIMENTAL RESULTS AND DISCUSSION** (5.1 Phân rã sai số hình học & 9 ablations trên Split B OOF) | **Table 3** (Bảng ablation và phân rã đơn cue trên Split B) |
| **Trang 7** | **5.2 Main Benchmark on Held-Out Split T** (So sánh mô hình (d), (e), (f0), (f) trên 3 detector), **5.3 Mechanistic Explanation** (Data efficiency & Range extrapolation) | **Table 4** (Bảng kết quả kiểm định chính trên Split T) |
| **Trang 8** | **5.4 Cross-Detector Comparison on Common Support (RQ2)** ($N=2,528$), **5.5 Uncertainty Quantification & Conformal Coverage (RQ3)** | **Table 5** (Common support & Spearman rank correlation) & **Figure 3** (`fig_03_ranging_error_by_distance.png`) |
| **Trang 9** | **5.5 Conformal Coverage cont.** (Tripartite coverage comparison, 20 resplits, Drive heterogeneity `0057`/`0004`), **5.6 Hardware Latency Benchmark (RQ4)** | **Table 6** (CQR Benchmark Split T), **Table 7** (Conditional ODD coverage), **Figure 4** (`fig_04_conformal_intervals_and_coverage.png`) |
| **Trang 10** | **5.6 Hardware Latency cont.** (Benchmark RTX 5060 & CPU ONNX), **5.7 Qualitative Error Analysis & Case Studies** (8 ca phân tích thực nghiệm) | **Table 8** (Bảng độ trễ phần cứng Tier 1) & **Figure 5** (`fig_05_qualitative_case_studies.png`) |
| **Trang 11** | **6. LIMITATIONS AND THREATS TO VALIDITY** (14 hạn chế phương pháp luận: survivorship bias, cluster correlation $k \le 12$, truck imbalance, exchangeability shift, v.v.) | --- |
| **Trang 12** | **7. CONCLUSION AND FUTURE WORK**, **DECLARATIONS & SCIENTIFIC INTEGRITY** (Anti-plagiarism, AI disclosure), **REFERENCES** ([1]–[12]) | Danh mục trích dẫn chuẩn SPIE |

---

## 3. Hướng Dẫn Chuyển Đổi Sang Template Word (`template-SPIE.doc`)

### Bước 1: Chuẩn bị tệp template
1. Tải tệp `template-SPIE.doc` từ trang hội nghị: [https://adip.org/template-SPIE.doc](https://adip.org/template-SPIE.doc).
2. Mở file trong Microsoft Word (hoặc LibreOffice / WPS Office) và lưu dưới dạng `.docx` trong quá trình chỉnh sửa (trước khi xuất bản nộp PDF hoặc DOC theo yêu cầu).
3. Đảm bảo khổ giấy là **Letter** (8.5 x 11 inch) hoặc **A4** (theo mặc định của template SPIE đi kèm, lề tiêu chuẩn: Top = 1 inch, Bottom = 1 inch, Left = 1 inch, Right = 1 inch).

### Bước 2: Thiết lập Typography & Styles
- **Paper Title:** Font 16 pt, Bold, Canh giữa (Centered), Title Case.
- **Authors & Affiliations:** Font 12 pt, Canh giữa. Tác giả đánh số mũ $^a, ^b$ trỏ đến đơn vị công tác tương ứng.
- **Abstract & Keywords:** Font 10 pt (hoặc 11 pt tùy template), Abstract in đậm tiêu đề `ABSTRACT`, nội dung 1 đoạn văn duy nhất (~200–250 từ). `Keywords:` in đậm, gồm 4–7 từ khóa cách nhau bởi dấu phẩy.
- **Heading 1 (Chương chính):** Font 11 pt, Bold, In hoa (ALL CAPS), Đánh số thứ tự kiểu La Mã hoặc số Ả Rập: `1. INTRODUCTION`, `2. RELATED WORK...`.
- **Heading 2 (Mục con):** Font 10 pt, Bold, Viết hoa chữ cái đầu (Title Case): `3.1 Perspective Geometry Cues`.
- **Body Text:** Font 10 pt, Regular, Canh đều hai bên (Justified), thụt đầu dòng (hoặc giãn đoạn 6 pt sau mỗi đoạn theo style template).

### Bước 3: Chèn Bảng biểu khoa học (Tables)
Toàn bộ 7 bảng khoa học chính thức đã được xuất bản sẵn trong thư mục `results/tables/final/`:
- Mở các file `.csv` bằng Excel để copy bảng, hoặc copy trực tiếp các bảng markdown trong file [ADIP_2026_MANUSCRIPT_SPIE.md](file:///d:/T%C3%A0i%20li%E1%BB%87u%20h%E1%BB%8Dc/FALL2026/DSR301m/Distance-Estimation-KITTI-YOLO/docs/paper/ADIP_2026_MANUSCRIPT_SPIE.md).
- **Quy chuẩn SPIE cho Bảng:**
  - Tiêu đề bảng đặt **PHÍA TRÊN** bảng: ví dụ `Table 1. Methodological positioning matrix against existing literature.`
  - Đường kẻ kiểu Booktabs: chỉ có đường viền trên cùng (1 pt), đường phân cách tiêu đề cột (0.5 pt), và đường viền đáy bảng (1 pt). **Tuyệt đối không dùng đường kẻ dọc (`|`)**.
  - Canh lề: Cột chữ canh trái, cột số liệu canh giữa hoặc canh phải theo dấu phẩy thập phân.

### Bước 4: Chèn Hình ảnh phân giải cao (Figures $\ge 300$ DPI)
Sử dụng các file ảnh gốc xuất bản độ nét cao tại `results/figures/final/`:
1. **Figure 1 (Architecture):** Vẽ sơ đồ vector hoặc chèn sơ đồ lưu trình modular.
2. **Figure 2 (Splits):** Chèn `results/figures/final/fig_02_splits_spatial_distribution.png`.
3. **Figure 3 (Ranging Error):** Chèn `results/figures/final/fig_03_ranging_error_by_distance.png`.
4. **Figure 4 (Conformal Coverage):** Chèn `results/figures/final/fig_04_conformal_intervals_and_coverage.png`.
5. **Figure 5 (Qualitative Cases):** Chèn `results/figures/final/fig_05_qualitative_case_studies.png` (Ảnh độ phân giải cực cao 300 DPI bao gồm lưới 8 trường hợp nghiên cứu điển hình).
- **Quy chuẩn SPIE cho Hình vẽ:**
  - Chú thích hình đặt **PHÍA DƯỚI** hình: ví dụ `Figure 2. Spatial and feature distribution across drive-disjoint splits.`
  - Canh giữa ảnh trong cột văn bản.

### Bước 5: Định dạng công thức toán (Equations)
- Các công thức toán (tính $Z_w, Z_h, Z_g$, hàm mục tiêu OAS shrinkage, log-residual $\hat{r}$, và khoảng conformal) được gõ bằng Microsoft Word Equation (Alt + =) hoặc MathType.
- Đánh số công thức bên phải lề trang: ví dụ `(1)`, `(2)`, v.v.

---

## 4. Checklist Liêm Chính Học Thuật & Phòng Tránh Đạo Văn (Turnitin / iThenticate)

Để đảm bảo tỷ lệ trùng lặp (Similarity Score) ở mức thấp nhất an toàn (< 10–12%) và không vi phạm bất kỳ chính sách nào của SPIE / ADIP:

1. **Số liệu kiểm chứng 100% (Zero Data Hallucination):**
   - Toàn bộ số liệu trong bài khớp từng bit với `results/final/numbers_manifest.json` và đã vượt qua bộ kiểm toán `python scripts/audit_paper.py`. Không có bất kỳ con số nào được bịa đặt hay ước chừng.
2. **Không sao chép nguyên văn (Zero Verbatim Copying):**
   - Mọi câu văn trong phần Related Work và Methodology đều được diễn đạt lại độc lập dựa trên tư duy mô hình hóa của nhóm nghiên cứu, có trích dẫn nguồn chuẩn xác ([1] đến [12]).
3. **Khác biệt rõ rệt so với nghiên cứu tiền lệ:**
   - Bảng 1 nêu rõ 7 chiều phân định phương pháp luận: phân tách rõ đóng góp của bài (3-cue OAS fusion + CQR + Drive-disjoint benchmark) so với Dist-YOLO (2022), DisNet (2018), DECADE (2024), AGL (2026).
4. **Văn phong khiêm tốn khoa học (Language Guard Passed):**
   - Không sử dụng từ ngữ tâng bốc cấm: không khẳng định "first work", không dùng "outperforms all", không dùng "statistically significant" khi số cụm $k \le 12$, không dùng "fail-safe" hay "comfortably".
   - Trung thực thừa nhận: Mô hình residual (f) và mô hình direct regression (e) có độ chính xác số học tương đương nhau trên Split T ($\Delta = -0.0002$, CI chứa 0); giá trị của mô hình hybrid nằm ở tính tường minh vật lý, khả năng ngoại suy dải xa và đường fallback khi bị cắt biên.
5. **Khai báo công cụ AI minh bạch:**
   - Đã đưa mục **Declarations & Scientific Integrity** vào cuối bài để khai báo việc sử dụng AI chỉ cho mục đích rà soát ngữ pháp và hỗ trợ định dạng theo đúng khuyến nghị của nhà xuất bản.

---

## 5. Các Kênh Nộp Bài Chính Thức Của ADIP 2026

Khi bản thảo đã hoàn thiện định dạng theo template và được xuất ra file PDF:

- **Kênh 1 (Hệ thống nộp bài trực tuyến):**
  - Truy cập: [https://www.zmeeting.org/submission/adip2026](https://www.zmeeting.org/submission/adip2026)
  - Đăng ký tài khoản Z-Meeting, chọn Submission Type: **Full Paper (for publication and presentation)**.
  - Điền đầy đủ Title, Abstract, Keywords, danh sách Authors & Affiliations.
  - Tải lên tệp bản thảo PDF và tệp nguồn Word `.doc`/`.docx`.
- **Kênh 2 (Email ban thư ký hội nghị):**
  - Địa chỉ tiếp nhận: `adip@acm-sg.org`
  - Tiêu đề email: `[ADIP 2026 Submission] Paper Title - Corresponding Author Name`
  - Đính kèm tệp PDF và file Word hoàn chỉnh.
