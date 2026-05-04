# Individual Reflection — Lab 18

**Tên:** Trương Quang Lộc  
**Module phụ trách:** M3

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M3 - Reranking.
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker.rerank()`: rerank lại danh sách document theo mức độ liên quan với query.
  - `benchmark_reranker()`: đo `avg_ms`, `min_ms`, `max_ms` cho bước rerank.
  - Hỗ trợ M5 với `contextual_prepend()` để thêm ngữ cảnh trước chunk khi indexing.
- Số tests pass:
  - `pytest tests/test_m3.py -v` -> `5 passed`
  - `test_contextual_*` trong `tests/test_m5.py` -> `2/2 passed`

---

## 2. Kiến thức học được

- Khái niệm mới nhất: Tôi hiểu rõ hơn vai trò của reranker trong pipeline RAG, đặc biệt là việc sắp xếp lại kết quả sau retrieval để tăng chất lượng context đưa vào bước answer generation.
- Điều bất ngờ nhất: Nếu retrieval ban đầu kéo về nhiều chunk gần đúng nhưng còn nhiễu, chỉ cần reranking tốt hơn cũng có thể cải thiện đáng kể đầu vào của mô hình trả lời.
- Kết nối với bài giảng: Phần M3 gắn trực tiếp với ý tưởng “retrieve nhiều rồi lọc lại” trong production RAG, giúp precision tốt hơn so với chỉ lấy top-k retrieval thô.

---

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Bước reranking phụ thuộc model ngoài, nên khi môi trường chưa tải được model hoặc chưa có đủ dependency thì rất dễ fail khi test và chạy pipeline.
- Cách giải quyết: Dùng hướng triển khai vừa hỗ trợ model thật, vừa có fallback để pipeline vẫn giữ đúng interface và tiếp tục chạy được khi môi trường chưa hoàn chỉnh.
- Thời gian debug: Khoảng 1-2 giờ để kiểm tra logic xếp hạng, kiểu dữ liệu đầu ra và benchmark latency.

---

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Tôi sẽ benchmark thêm nhiều kiểu query hơn để đo rõ reranker cải thiện ở trường hợp nào nhiều nhất.
- Module nào muốn thử tiếp: Tôi muốn thử thêm M2 hoặc M5 để xem khi kết hợp retrieval tốt hơn với contextual enrichment thì precision có tăng mạnh hay không.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |
