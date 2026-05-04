# Individual Reflection — Lab 18

**Tên:** Nguyễn Triệu Gia Khánh  
**Module phụ trách:** M4

**Mã sinh viên:** 2A202600225  
**Phần hỗ trợ thêm:** M5 (`extract_metadata()`, `enrich_chunks()` và các test liên quan)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M4 - RAGAS Evaluation.
- Các hàm/class chính đã viết:
  - `evaluate_ragas()`: trả về 4 metric `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` và `per_question`.
  - `failure_analysis()`: phân tích `bottom_n` câu hỏi có điểm thấp nhất, xác định `worst_metric`, `diagnosis`, `suggested_fix`.
  - `load_test_set()`: đọc test set an toàn hơn khi file JSON có dấu phẩy cuối.
  - Bổ sung fallback heuristic scoring để phần M4 vẫn chạy ổn định khi chưa có điều kiện dùng RAGAS/LLM thật.
- Đóng góp hỗ trợ thêm ở M5:
  - `extract_metadata()`
  - `enrich_chunks()`
  - các test `test_extract_metadata_*` và `test_enrich_*` trong `tests/test_m5.py`
- Làm thêm cho phần nhóm:
  - hỗ trợ chạy full pipeline end-to-end và sinh `reports/ragas_report.json`, `reports/naive_baseline_report.json`
  - điền `analysis/group_report.md` và `analysis/failure_analysis.md` theo kết quả chạy thật
  - chỉnh `check_lab.py` để auto-test đọc đúng kết quả `pytest`
- Số tests pass:
  - `pytest tests/test_m4.py -v` -> `6 passed in 0.05s`
  - `pytest tests/test_m5.py -v` -> `16 passed in 0.04s`

Kết quả test M4:

```text
tests/test_m4.py::test_load_test_set PASSED
tests/test_m4.py::test_evaluate_returns_metrics PASSED
tests/test_m4.py::test_failure_analysis_returns PASSED
tests/test_m4.py::test_failure_has_diagnosis PASSED
tests/test_m4.py::test_failure_analysis_maps_worst_metric PASSED
tests/test_m4.py::test_save_report_writes_expected_shape PASSED

6 passed in 0.05s
```

---

## 2. Kiến thức học được

- Khái niệm mới nhất: Tôi hiểu rõ hơn cách đánh giá hệ thống RAG bằng các metric khác nhau thay vì chỉ nhìn một điểm số tổng hợp. `faithfulness` phản ánh mức độ bám context, còn `context_precision` và `context_recall` giúp tách lỗi retrieval ra khỏi lỗi generation.
- Điều bất ngờ nhất: Một fallback implementation đơn giản nhưng đúng interface có thể giúp việc code, test, và ghép pipeline nhóm ổn định hơn rất nhiều khi chưa sẵn API key hoặc dependency ngoài.
- Kết nối với bài giảng: Phần M4 gắn trực tiếp với nội dung đánh giá hệ RAG, failure analysis, và tư duy chẩn đoán lỗi theo pipeline thay vì chỉ sửa prompt một cách cảm tính. Phần M5 giúp tôi hiểu thêm vai trò của enrichment trước embedding, đặc biệt là metadata extraction để tăng precision khi retrieve.

---

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: `evaluate_ragas()` theo hướng chuẩn có thể phụ thuộc vào API key, model, hoặc môi trường cài đặt `ragas`, nên nếu bám hoàn toàn vào đường chạy thật thì test local dễ fail hoặc chạy không ổn định.
- Cách giải quyết: Tôi thiết kế hai tầng xử lý:
  - nếu có đủ điều kiện thì dùng RAGAS thật
  - nếu chưa có API key hoặc môi trường chưa sẵn sàng thì fallback sang heuristic scoring
  Ngoài ra tôi cũng làm `load_test_set()` robust hơn để xử lý trường hợp `test_set.json` có lỗi dấu phẩy cuối.
- Thời gian debug: Khoảng 1-2 giờ, chủ yếu để chốt interface đầu ra, kiểm tra test, và làm phần failure analysis đủ rõ để nhóm có thể dùng tiếp trong pipeline chung.

---

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Nếu có thêm thời gian, tôi sẽ chuẩn hóa test set kỹ hơn và chạy thêm một vòng end-to-end với dữ liệu thực để kiểm tra mức chênh giữa heuristic fallback và RAGAS thật.
- Module nào muốn thử tiếp: Tôi muốn thử tối ưu sâu hơn phần M5, đặc biệt là contextual prepend và hypothesis questions để xem enrichment có cải thiện retrieval score rõ rệt đến mức nào khi ghép vào pipeline nhóm.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
