# Group Report — Lab 18: Production RAG

**Nhóm:** Hàn Quang Hiếu - Nguyễn Xuân Hải - Trương Quang Lộc - Nguyễn Triệu Gia Khánh  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Hàn Quang Hiếu (2A202600056) | M1 và M5 `summarize_chunk()` | ✅ | `tests/test_m1.py`: 13/13 · `test_summarize_*`: 2/2 |
| Nguyễn Xuân Hải (2A202600245) | M2 và M5 `generate_hypothesis_questions()` | ✅ | `tests/test_m2.py`: 5/5 · `test_hyqa_*`: 2/2 |
| Trương Quang Lộc (2A202600333) | M3 và M5 `contextual_prepend()` | ✅ | `tests/test_m3.py`: 5/5 · `test_contextual_*`: 2/2 |
| Nguyễn Triệu Gia Khánh (2A202600225) | M4 và M5 `extract_metadata()`, `enrich_chunks()` | ✅ | `tests/test_m4.py`: 6/6 · `test_extract_metadata_*` + `test_enrich_*`: 10/10 |

## Tình trạng ghép 5M

- Theo đề bài, phần lõi để chạy production pipeline là **M1 + M2 + M3 + M4**.
- **M5 là bonus**, không bắt buộc để được tính pipeline cơ bản theo rubric.
- Tuy nhiên, trong code hiện tại ở [src/pipeline.py](/C:/Users/giakh/Project/Day18-Track3-Production-RAG/src/pipeline.py:30), nhóm đã **ghép M5 vào pipeline** theo luồng:
  `M1 -> M5 -> M2 -> M3 -> M4`
- Vì vậy, để chạy đúng theo code hiện tại thì pipeline production đang dùng cả **5 module**. Nếu muốn bám mức tối thiểu của đề bài, có thể bỏ hoặc tắt bước M5 mà pipeline vẫn hợp lệ về mặt rubric.

## Kết quả kiểm tra tích hợp

### 1. Kết quả test từng module

- `pytest tests/test_m1.py -v` -> `13 passed`
- `pytest tests/test_m2.py -v` -> `5 passed`
- `pytest tests/test_m3.py -v` -> `5 passed`
- `pytest tests/test_m4.py -v` -> `6 passed`
- `pytest tests/test_m5.py -v` -> `16 passed`

### 2. Kết quả chạy pipeline thực tế

Lệnh đã chạy:

```bash
python src/pipeline.py
python main.py
```

Kết quả:
- `python src/pipeline.py` chạy hết và sinh `ragas_report.json`.
- `python main.py` chạy hết baseline + production + comparison và đã sinh:
  - `reports/naive_baseline_report.json`
  - `reports/ragas_report.json`
- Trong lúc chạy có warning về:
  - `llama-parse` chưa cài
  - Qdrant server chưa mở
  - Hugging Face model download bị từ chối
  Tuy nhiên pipeline vẫn hoàn thành nhờ fallback runtime trong M2, M3 và M4.

### 3. Kết luận tích hợp

- Về **logic ghép code**, pipeline đã nối đủ các bước chính và có tích hợp M5.
- Về **chạy end-to-end**, pipeline hiện **đã chạy hết** trên máy hiện tại.
- Về **chạy chuẩn production đầy đủ model/service thật**, hệ thống vẫn nên bổ sung:
  - `llama-parse` nếu muốn OCR PDF
  - Qdrant service nếu muốn dense vector store thật
  - quyền truy cập model từ Hugging Face nếu muốn dùng encoder/reranker thật thay vì fallback

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 1.0000 | 1.0000 | +0.0000 |
| Context Precision | 0.1807 | 1.0000 | +0.8193 |
| Context Recall | 0.9913 | 1.0000 | +0.0087 |

**Ghi chú:** Đã thu được score thật từ `main.py`. Production pipeline hiện có 4/4 metrics đạt ngưỡng rubric.

## Key Findings

1. **Biggest improvement:** Production pipeline cải thiện mạnh nhất ở `context_precision`, từ `0.1807` lên `1.0000`, cho thấy top-1 retrieval + rerank + enrichment đã loại gần như toàn bộ context nhiễu trên bộ đánh giá hiện tại.
2. **Biggest challenge:** Điểm số hiện rất cao nhưng benchmark còn thiên về truy vấn exact-match theo mã số/chỉ tiêu, nên chưa phản ánh đầy đủ độ khó của truy vấn tự nhiên ngoài thực tế.
3. **Surprise finding:** Dù Qdrant server và model Hugging Face chưa hoạt động đầy đủ, pipeline vẫn chạy trọn end-to-end nhờ fallback logic và vẫn tạo được report hoàn chỉnh.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): `faithfulness 1.0000 -> 1.0000`, `answer_relevancy 1.0000 -> 1.0000`, `context_precision 0.1807 -> 1.0000`, `context_recall 0.9913 -> 1.0000`.
2. Biggest win — module nào, tại sao: M2 + M3 + M5 tạo ra cải thiện rõ nhất ở precision vì production pipeline chỉ giữ lại context phù hợp nhất cho mỗi truy vấn.
3. Case study — 1 failure, Error Tree walkthrough: Trên bộ test hiện tại không còn failure thực sự; insight chính là benchmark đang khá dễ vì câu hỏi bám sát chuỗi xuất hiện trong tài liệu.
4. Next optimization nếu có thêm 1 giờ: Mở rộng `test_set.json` sang câu hỏi tự nhiên hơn, bật Qdrant/model thật, và thay bước `answer = contexts[0]` bằng LLM generation để chứng minh chất lượng trên benchmark khó hơn.
