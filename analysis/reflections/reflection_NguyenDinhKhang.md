# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Đình Khang  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04-10-2026

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex, dùng `all-MiniLM-L6-v2` và cosine similarity; ngưỡng 0.85 tạo biên chunk khi chủ đề thay đổi. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Child tối đa 256 ký tự cho retrieval, parent tối đa 2048 ký tự cho context; `parent_id` liên kết hai mức. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | Đổi `_` của underthesea thành khoảng trắng cho BM25; RRF dùng `1/(k + rank + 1)` nên không cần chuẩn hóa score. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` chấm trực tiếp cặp query–document rồi chọn top-3 từ candidate. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Bốn metric được đóng gói thành `EvalResult` và diagnostic tree ánh xạ metric thấp sang hướng sửa. |
| Contextual embeddings | M5 | `_enrich_single_call()`, `contextual_prepend()` | Một JSON call/chunk cung cấp summary, HyQA, context, metadata; fallback cục bộ giữ pipeline chạy khi API unavailable. |

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:** `ModuleNotFoundError: No module named 'numpy'`/`pypdf` khi pytest dùng sai interpreter; `RuntimeWarning: coroutine ... was never awaited` từ RAGAS 0.1.22 trên Python 3.14; Hugging Face DNS retry dù model đã cache; và `APIConnectionError: Connection error` với OpenAI.
- **Nguyên nhân gốc rễ & Cách debug:** Kiểm tra `command -v python`/`pytest` và chạy `python -m pytest` trong `.venv`. RAGAS 0.1.x gọi `asyncio.as_completed` trước event loop, nên thêm compatibility adapter lazy iterator cho Python 3.14. Đổi model sang `local_files_only=True` sau khi xác nhận cache đầy đủ. Thêm circuit breaker: lỗi OpenAI một lần thì M5/LLM chuyển fallback thay vì retry theo mọi chunk/query.
- **Kiến thức còn thiếu & Cách khắc phục:** Report chưa persist answer/context theo query nên không audit được khi LLM judge unavailable. Cần thêm retrieval metric offline và regression tests cho version conflict v2023/v2024.

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý RAG tra cứu quy chế nội bộ tiếng Việt

#### 1. Hiện trạng

- **Pipeline hiện tại:** Nạp tài liệu quy chế, chunk, truy xuất và trả lời theo context.
- **Vấn đề / Bottlenecks:** Dense retrieval có thể bỏ sót số liệu/từ khóa; quy định cũ–mới dễ gây sai; thiếu trace để giải thích retrieval và đánh giá.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Hierarchical mặc định, structure-aware cho tài liệu có tiêu đề, giữ source/`parent_id`/version.
2. **Search retrieval:** BM25 tiếng Việt + BGE-M3 dense, hợp nhất RRF để kết hợp lexical exact-match và semantic match.
3. **Reranking:** Dùng `BAAI/bge-reranker-v2-m3` cho top-20 candidate và chỉ gửi top-3 vào LLM; benchmark warm latency.
4. **Evaluation:** RAGAS 4 metrics khi endpoint ổn định, kèm retrieval recall/precision offline trong CI.
5. **Enrichment:** Contextual prepend và metadata `version`, `effective_date`, `status`; chỉ giữ HyQA nếu uplift đủ bù chi phí index.

#### 3. Timeline triển khai

- **Tuần 1:** Chuẩn hóa nguồn, metadata hiệu lực, hierarchical/structure-aware chunking và regression set cũ–mới.
- **Tuần 2:** Hybrid Search, reranking, logging per-query; benchmark offline và chạy RAGAS khi OpenAI connectivity ổn định.
