# Review độc lập: Insurance Anti-Fraud Detection

**Đối tượng review:** `analysis.ipynb`, `report.html`.
**Dữ liệu:** `insurance_claims.csv` @ commit `7d1b322792ba45878c0f1a879e6aee8328939d34`.

## Bảng đánh giá

| Hạng mục | Kết quả | Nhận xét |
|---|---|---|
| Tái lập | ✅ Đạt | Notebook chạy từ đầu tới cuối không lỗi (`nbconvert --execute`). `SEED = 42` được dùng cho split, CV, mô hình và bootstrap. Commit hash được ghi ở đầu notebook và report. |
| Chia dữ liệu | ✅ Đạt | Chia 80/20 có stratify (tỉ lệ gian lận 24,8% / 24,5%). `assert` xác nhận không trùng `policy_number` giữa hai tập. Chia ngẫu nhiên phù hợp vì không có nhóm lặp và dữ liệu chỉ kéo dài 2 tháng. Test chỉ dùng cho mô hình cuối; ngưỡng được chọn bằng OOF trên train. |
| Leakage | ✅ Đạt | Scaler, one-hot và SelectKBest đều nằm trong `Pipeline`, fit theo từng fold. Các biến đổi ngoài pipeline (`clean`) là biến đổi xác định, không học từ dữ liệu. Không có biến hậu kiểm hay biến sinh từ nhãn. EDA dùng nhãn chỉ chạy trên train. |
| Drift | ✅ Đạt | Kiểm định KS trên 8 biến số cho mọi p > 0,1. Không có tập test Kaggle riêng nên không cần adversarial validation. |
| Chất lượng dữ liệu | ⚠️ Cần lưu ý | Dấu `?` được giữ thành hạng mục "Unknown", hợp lý vì việc thiếu có thể mang thông tin. Giá trị "None" ở `authorities_contacted` được đọc đúng là hạng mục thật, không phải NaN; nếu dùng `read_csv` mặc định (pandas ≥ 2) sẽ mất 91 giá trị này. Nghi vấn lớn nhất là **dữ liệu có vẻ mô phỏng**: sở thích chess/cross-fit cho tỉ lệ gian lận ~86%, rất khó giải thích về nghiệp vụ. |
| Phương pháp | ✅ Đạt | Có baseline Dummy. Metric chính là PR-AUC, phù hợp với dữ liệu mất cân bằng; F1, ROC-AUC, precision và recall được báo kèm. CV dùng StratifiedKFold, cùng chiến lược với split. `class_weight='balanced'` được dùng cho LR, DT và RF. Đủ 5 mô hình theo yêu cầu README, cùng với feature selection và visualize cây. |
| Overfit | ⚠️ Cần lưu ý | Random Forest có train PR-AUC 1,0 so với CV 0,65, tức overfit rõ; mô hình này không được chọn. Mô hình cuối (NB) có train 0,726 ≈ CV 0,726. **PR-AUC test (0,551) thấp hơn CV (0,726) tới 0,18.** Đã điều tra: bootstrap CI của test là [0,43; 0,70]; repeated CV 5×5 cho 0,72 ± 0,05 (min 0,63); F1 test 0,73 khớp F1 OOF 0,755. Kết luận: phần lớn khoảng chênh đến từ phương sai của test nhỏ (49 ca dương) và từ thứ hạng bên trong nhóm bị gắn cờ, không phải leakage. Tuy vậy test vẫn nằm ở phía dưới phân phối, nên PR-AUC nên được xem là không chắc chắn. |
| Thống kê | ✅ Đạt | Chi-square có kèm Cramér's V; 4 kiểm định vẫn giữ kết luận sau hiệu chỉnh Bonferroni. Mann-Whitney cho `total_claim_amount` có p nhỏ nhưng chênh trung vị chỉ khoảng 9%, và report đã nói rõ đây là effect nhỏ. Kiểm định Mann-Whitney cho 8 biến chưa hiệu chỉnh đa kiểm định; `umbrella_limit` (p = 0,038) sẽ không còn ý nghĩa sau Bonferroni, nhưng biến này không được dùng để kết luận. |
| Diễn giải | ✅ Đạt | Report nêu rõ quan hệ là tương quan, không phải nhân quả. Report cũng nhấn mạnh rằng dự đoán của mô hình trùng 100% với quy tắc 2 điều kiện, thay vì thổi phồng "độ thông minh" của mô hình. Top 5 được lấy theo đồng thuận 4 phương pháp, kèm cảnh báo rằng hạng 3–5 có tác động rất nhỏ và không ổn định. |
| Số liệu | ✅ Đạt | Đã đối chiếu từng số trong report với output notebook: bảng CV (0,726/0,712/0,707/0,646/0,628/0,262), bảng test (0,551/0,837/0,730/0,636/0,857/0,845), CI (0,43–0,70; 0,63–0,81), tỉ lệ nhóm (60%, 10%, 15%, 8%, 87%, 86%, 4%, 88%), trung vị (60.995 / 56.080), Cramér's V (0,50 / 0,46), confusion matrix (42/49 bắt được, 66 hồ sơ gắn cờ). |
| Submission | — Không áp dụng | Đây không phải bài dạng competition và không có test Kaggle không nhãn. |
| Trực quan hóa & truyền đạt | ✅ Đạt | Biểu đồ có tiêu đề và nhãn trục. Cây quyết định dùng đơn vị gốc (USD) nhờ bỏ scaler cho các mô hình cây. Report có nêu rằng `value` trong cây là số mẫu đã cân trọng số. Khuyến nghị cụ thể, có mức ưu tiên. |
| Nguồn (repo) | ✅ Đạt | Dữ liệu khớp mô tả trong README (Kaggle buntyshah). Không có secret; không có LFS pointer. Dữ liệu có mã bưu chính và địa chỉ sự cố (PII giả lập): các cột này đã bị loại khỏi mô hình và không xuất hiện trong report. |

## Các điểm đã sửa trong quá trình review

1. **Top 5 feature.** Bản đầu lấy top 5 thuần theo permutation importance của mô hình cuối. Tuy nhiên hạng 3–5 ở đó có importance nhỏ hơn độ lệch chuẩn của chính nó (ví dụ `authorities_contacted` 0,011 ± 0,013), nên không đáng tin. Đã chuyển sang **xếp hạng đồng thuận 4 phương pháp** và thêm cảnh báo trong notebook và report.
2. **Chênh lệch CV–test.** Bổ sung bootstrap CI, repeated CV và so sánh với quy tắc 2 điều kiện để giải thích khoảng chênh, thay vì chỉ báo con số.
3. **Cây quyết định khó đọc.** Bản đầu hiển thị ngưỡng trên thang đã chuẩn hóa (ví dụ `vehicle_claim <= -0.02`). Đã bỏ scaler cho DT và RF (mô hình cây không cần chuẩn hóa), nên ngưỡng giờ tính bằng USD. Điểm CV không đổi.

## Rủi ro & giới hạn

- **Bảng "tất cả mô hình trên test" (mục 8.1)** là thông tin tham khảo theo yêu cầu đề bài. Nếu dùng bảng này để chọn mô hình thì sẽ vi phạm nguyên tắc "test chỉ dùng một lần". Trên bảng đó KNN có PR-AUC test cao nhất (0,61), khác với thứ tự trên CV, và đây chính là minh họa cho độ nhiễu của test nhỏ.
- **Khác biệt giữa NB, DT và LR nằm trong nhiễu.** Việc chọn NB dựa trên CV PR-AUC cao nhất, nhưng không có ưu thế thống kê rõ ràng. Nếu ưu tiên giải thích, DT hoặc quy tắc 2 điều kiện là lựa chọn tương đương.
- **Ngưỡng tối ưu F1** không phản ánh chi phí thực tế giữa báo động giả và bỏ sót.
- **Tổng quát hóa:** dữ liệu 1.000 dòng, 2 tháng, nghi mô phỏng. Không nên dùng trực tiếp trong sản xuất.
- **Đạo đức và pháp lý:** dùng sở thích cá nhân để gắn cờ gian lận có thể gây tranh cãi về phân biệt đối xử và quyền riêng tư.

## Đề xuất cải thiện

- **Dữ liệu:** lịch sử bồi thường theo khách hàng, thông tin gara và nhân chứng, khoảng thời gian từ sự cố tới lúc báo cáo, và dữ liệu nhiều năm để kiểm tra theo thời gian.
- **Mô hình:** calibration xác suất (NB thường cho xác suất lệch); chọn ngưỡng theo ma trận chi phí; xây mô hình xếp hạng riêng cho nhóm Major Damage để cải thiện precision ở nhóm này.
- **Đánh giá:** dùng nested CV hoặc repeated CV để báo cáo thay cho một lần split; kiểm tra công bằng theo giới tính, tuổi và nghề nghiệp.

## Độ tin cậy tổng quan: **Trung bình**

Quy trình sạch (không leakage, split đúng, tái lập được), và kết luận định tính (hai yếu tố chi phối) rất vững. Con số hiệu năng lại có khoảng bất định rộng do mẫu nhỏ, và khả năng áp dụng ra thực tế bị giới hạn bởi nghi vấn dữ liệu mô phỏng.
