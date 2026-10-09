================================================================================
HƯỚNG DẪN BIÊN DỊCH VÀ NỘP BÀI ADIP 2026 TRÊN OVERLEAF
8th Asia Digital Image Processing Conference (Tokyo, Japan) - SPIE Proceedings
================================================================================

1. CÁCH ĐƯA DỰ ÁN LÊN OVERLEAF:
   - Bước 1: Đăng nhập vào tài khoản Overleaf (https://www.overleaf.com).
   - Bước 2: Bấm nút "New Project" (Dự án mới) -> Chọn "Upload Project" (Tải lên dự án).
   - Bước 3: Chọn file zip "ADIP_2026_Overleaf_Package.zip".
   - Bước 4: Overleaf sẽ tự động giải nén và mở file "main.tex".

2. CẤU HÌNH BIÊN DỊCH (OVERLEAF SETTINGS):
   - Bấm vào biểu tượng "Menu" ở góc trên bên trái màn hình Overleaf:
     + Compiler: pdfLaTeX (mặc định)
     + TeX Live version: 2023 hoặc 2024 (mặc định)
     + Main document: main.tex
   - Bấm "Recompile" (hoặc Ctrl + Enter).
   - Quá trình biên dịch sẽ sinh ra bản thảo PDF chuẩn định dạng SPIE Proceedings (~11-12 trang).

3. ĐIỀN THÔNG TIN TÁC GIẢ (DE-ANONYMIZE NẾU CẦN):
   - Mở file main.tex, tìm dòng 28-38:
     Thay thế:
     \author{Anonymous Authors\supit{a}
     \skiplinehalf
     \supit{a}Department of Computer Science \& Engineering...
     }
     Bằng tên thật của các tác giả và đơn vị công tác (Affiliation) cùng email tương ứng.
   - Lưu ý: Hội nghị ADIP theo thông lệ xem xét đơn mù đơn (single-blind), vì vậy tác giả 
     nên điền đầy đủ tên thật, đơn vị công tác và email tác giả liên hệ trước khi xuất bản PDF.

4. CÁC TỆP TIN TRONG GÓI:
   - main.tex: Mã nguồn LaTeX hoàn chỉnh (đã bao gồm đầy đủ 7 chương, 8 bảng biểu, công thức toán và trích dẫn).
   - spie.cls: File định dạng chính thức của SPIE Proceedings từ CTAN.
   - spiebib.bst: File định dạng tài liệu tham khảo SPIE Proceedings.
   - references.bib: Danh mục 13 tài liệu trích dẫn chuẩn học thuật quốc tế.
   - figures/: Thư mục chứa 5 hình vẽ khoa học độ phân giải cao (fig_01 đến fig_05).

5. NỘP BÀI CHO HỘI NGHỊ ADIP 2026:
   - Hạn chót: 10/10/2026 (Full Paper Submission Deadline).
   - Kênh 1 (Hệ thống Z-Meeting): https://www.zmeeting.org/submission/adip2026
     + Chọn loại: Full Paper (for publication and presentation)
     + Điền Title, Abstract, Keywords từ file main.tex
     + Tải lên tệp PDF tải về từ Overleaf.
   - Kênh 2 (Email dự phòng): adip@acm-sg.org
     + Tiêu đề: [ADIP 2026 Submission] Paper Title - Corresponding Author Name
     + Đính kèm file PDF.
================================================================================
