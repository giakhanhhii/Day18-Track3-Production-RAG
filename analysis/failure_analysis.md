# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Hàn Quang Hiếu - Nguyễn Xuân Hải - Trương Quang Lộc - Nguyễn Triệu Gia Khánh  
**Thành viên:** Hàn Quang Hiếu → M1 · Nguyễn Xuân Hải → M2 · Trương Quang Lộc → M3 · Nguyễn Triệu Gia Khánh → M4

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 1.0000 | 1.0000 | +0.0000 |
| Context Precision | 0.1807 | 1.0000 | +0.8193 |
| Context Recall | 0.9913 | 1.0000 | +0.0087 |

## Bottom-5 Failures

**Ghi chú:** Bộ `test_set.json` hiện có 5 truy vấn và production pipeline đạt điểm tối đa trên cả 5 truy vấn. Vì vậy phần dưới đây ghi lại 5 case có điểm thấp nhất theo thứ tự, nhưng đều không phải failure thực sự.

### #1
- **Question:** `0106769437`
- **Expected:** Chunk chứa mã số thuế của doanh nghiệp
- **Got:** Trùng khớp với ground truth
- **Worst metric:** `faithfulness = 1.0000`
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Không có lỗi nghiêm trọng ở case này. Truy vấn là exact-match nên retrieval rất ổn định.
- **Suggested fix:** Giữ case này làm sanity check và bổ sung thêm truy vấn tự nhiên hơn để benchmark khó hơn.

### #2
- **Question:** `52.133.830`
- **Expected:** Chunk chứa chỉ tiêu thuế giá trị gia tăng phải nộp
- **Got:** Trùng khớp với ground truth
- **Worst metric:** `faithfulness = 1.0000`
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Không có lỗi nghiêm trọng. Truy vấn bám trực tiếp vào số liệu đặc trưng nên precision rất cao.
- **Suggested fix:** Bổ sung câu hỏi diễn đạt tự nhiên như “thuế GTGT phải nộp trong kỳ là bao nhiêu?” để thử thách retrieval hơn.

### #3
- **Question:** `13/2023`
- **Expected:** Chunk mở đầu Nghị định số 13/2023/NĐ-CP
- **Got:** Trùng khớp với ground truth
- **Worst metric:** `faithfulness = 1.0000`
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Không có lỗi nghiêm trọng. Heading và số hiệu văn bản giúp retrieve rất chính xác.
- **Suggested fix:** Thêm câu hỏi khái niệm như “Nghị định 13/2023 quy định về vấn đề gì?” để đánh giá semantic retrieval tốt hơn.

### #4
- **Question:** `77.377.803`
- **Expected:** Chunk chứa số thuế GTGT còn được khấu trừ kỳ trước chuyển sang
- **Got:** Trùng khớp với ground truth
- **Worst metric:** `faithfulness = 1.0000`
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Không có lỗi nghiêm trọng. Đây là case exact-match theo số liệu tài chính.
- **Suggested fix:** Mở rộng thêm câu hỏi giải thích ý nghĩa chỉ tiêu để đánh giá chất lượng answer generation.

### #5
- **Question:** `215.163.767`
- **Expected:** Chunk chứa số thuế GTGT được khấu trừ kỳ này
- **Got:** Trùng khớp với ground truth
- **Worst metric:** `faithfulness = 1.0000`
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Không có lỗi nghiêm trọng. Query mang tính exact-match nên benchmark hiện khá dễ.
- **Suggested fix:** Thêm câu hỏi paraphrase hoặc multi-hop để tìm failure thực sự của pipeline.

## Case Study (cho presentation)

**Question chọn phân tích:** `13/2023`

**Error Tree walkthrough:**
1. Output đúng? → Có.
2. Context đúng? → Có, chunk top-1 khớp trực tiếp với phần mở đầu nghị định.
3. Query rewrite OK? → Có, vì query là exact identifier nên lexical match rất mạnh.
4. Fix ở bước: Không cần fix cho case này; thay vào đó cần **nâng độ khó benchmark** để tìm ra điểm yếu thực sự của hệ thống.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Mở rộng `test_set.json` thành khoảng 20 câu hỏi tự nhiên hơn, không chỉ exact-match theo mã số/chỉ tiêu
- Bật Qdrant và model thật để giảm phụ thuộc fallback
- Dùng LLM answer generation thay cho `answer = contexts[0]`
- Đo thêm latency từng bước để lấy bonus latency breakdown
