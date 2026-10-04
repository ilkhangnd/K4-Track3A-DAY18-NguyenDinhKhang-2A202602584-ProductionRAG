# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Đình Khang<br>
**Khóa:** K4 - Track 3A<br>
**Lần chạy:** 04-10-2026

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8405 | 0.8483 | +0.0079 |
| Answer Relevancy | 0.6000 | 0.8526 | +0.1726 |
| Context Precision | 0.9250 | 0.9417 | +0.0167 |
| Context Recall | 0.9250 | 0.9000 | -0.0250 |

### Run summary

- Baseline index: 57 paragraph chunks; production index: 117 hierarchical chunks from 26 documents with text layer.
- Production took 564.3s; M5 enrichment took 357.6s for 117 chunks.
- Production is lower on all four metrics, with the largest regression in **Faithfulness** (-0.1258) and **Context Recall** (-0.0750). Therefore, the next iteration should prioritize grounding/version control before adding more enrichment.
- `ragas_report.json` stores aggregate scores and diagnoses, but not raw per-question LLM answers. “Got” below records the evidence available from a post-run retrieval replay against the same production Qdrant collection; it is not presented as a verbatim LLM output.

## Bottom-5 Failures

### #1 — Nghỉ phép không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần Giám đốc điều hành (CEO) phê duyệt; nghỉ trên 14 ngày nhân viên tự đóng phần bảo hiểm của mình.
- **Got / retrieved evidence:** Top context là quy trình nghỉ không lương và nêu nhóm 16–30 ngày cần Giám đốc điều hành phê duyệt. Hai context sau lại là quy định về phép năm, không trực tiếp hỗ trợ câu trả lời.
- **Worst metric:** `faithfulness`; average score `0.4454`.
- **Câu trả lời có đúng không?** Grounding chưa đáng tin cậy theo RAGAS; raw answer không được lưu để đối chiếu từng mệnh đề.
- **Context có chứa đáp án không?** Có, top-1 có thông tin CEO; nhưng top-2/top-3 là noise.
- **Câu hỏi có cần viết lại không?** Không, câu hỏi rõ và chứa độ dài nghỉ cụ thể.
- **Error Tree:** Output yếu → context đúng một phần → query rõ → giảm context noise/synthesis grounding.
- **Root cause:** Reranker vẫn đưa hai chunk phép năm không liên quan vào top-3, làm tăng rủi ro LLM trộn điều kiện của phép năm với nghỉ không lương.
- **Suggested fix:** Lọc theo `leave_type=unpaid` hoặc boost exact phrase “nghỉ phép không lương”; system prompt yêu cầu chỉ nêu điều khoản xuất hiện trong context phù hợp.

### #2 — Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành v2.0, mật khẩu đổi mỗi 120 ngày; quy định 90 ngày đã bị thay thế.
- **Got / retrieved evidence:** Top-1 nêu đúng **120 ngày**, nhưng top-2 nêu **90 ngày** và top-3 là thông báo chính sách cũ đã được thay thế.
- **Worst metric:** `faithfulness`; average score `0.4583`.
- **Câu trả lời có đúng không?** Chưa thể khẳng định raw answer, nhưng RAGAS cho thấy answer có mệnh đề không được grounding tốt.
- **Context có chứa đáp án không?** Có, đồng thời chứa quy định cũ cạnh tranh trực tiếp.
- **Câu hỏi có cần viết lại không?** Có thể thêm “theo chính sách hiện hành”; hệ thống vẫn phải tự xử lý version conflict.
- **Error Tree:** Output dễ trộn 120/90 → context chứa cả current và obsolete → query thiếu mốc phiên bản → version-aware retrieval.
- **Root cause:** Không có metadata/filter `status=current` hoặc `effective_date` để loại văn bản 90 ngày trước synthesis.
- **Suggested fix:** M5 trích xuất `version`, `effective_date`, `status`; M2 filter/boost current documents và M3 thêm instruction ưu tiên điều khoản mới nhất.

### #3 — Số ký tự tối thiểu của mật khẩu

- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo v2.0 hiện hành, tối thiểu 12 ký tự; v1.0 yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got / retrieved evidence:** Top-1 nêu đúng **12 ký tự**; top-2 vẫn chứa mức **8 ký tự** của chính sách cũ.
- **Worst metric:** `faithfulness`; average score `0.5000`.
- **Câu trả lời có đúng không?** RAGAS đánh dấu grounding thấp, dù context đầu tiên đúng; nguy cơ chính là LLM chọn hoặc kết hợp nhầm 8 và 12.
- **Context có chứa đáp án không?** Có, nhưng context cạnh tranh mang thông tin cũ.
- **Câu hỏi có cần viết lại không?** Không bắt buộc; có thể thêm “hiện hành v2.0” để giảm ambiguity cho người dùng.
- **Error Tree:** Output có nguy cơ sai số liệu → context đúng lẫn cũ → query không nêu version → xử lý version conflict.
- **Root cause:** Hybrid Search/RRF không phân biệt hiệu lực pháp lý của hai đoạn có lexical match tương đương.
- **Suggested fix:** Index metadata version và status; dùng metadata filter trước RRF, đồng thời đưa header/version vào enriched text.

### #4 — Bảo hiểm PVI cho nhân viên thử việc

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Không. Nhân viên thử việc chỉ tham gia bảo hiểm xã hội bắt buộc, chưa hưởng PVI.
- **Got / retrieved evidence:** Top-1 là chunk về thử việc/PVI nhưng phần replay bắt đầu giữa câu ở “được hưởng gói bảo hiểm sức khỏe PVI”, mất phủ định; top-3 cũng bị cắt ở “chưa…”.
- **Worst metric:** `faithfulness`; average score `0.5000`.
- **Câu trả lời có đúng không?** Có rủi ro sai cao: khi phủ định “không/chưa” bị tách khỏi phần còn lại, LLM có thể đảo nghĩa.
- **Context có chứa đáp án không?** Chỉ chứa đáp án không trọn vẹn; bằng chứng phủ định bị mất ở biên chunk.
- **Câu hỏi có cần viết lại không?** Không; câu hỏi yes/no rất rõ.
- **Error Tree:** Output dễ đảo nghĩa → context thiếu phủ định → query rõ → lỗi thuộc chunk boundary/parent-child expansion.
- **Root cause:** Pipeline index child chunk nhưng không mở rộng về parent context khi trả lời, nên điều kiện phủ định có thể nằm ngoài child được retrieve.
- **Suggested fix:** Sau khi retrieve child, map `parent_id` để gửi parent chunk đầy đủ cho reranker/LLM; thêm test regression cho câu hỏi phủ định và assertion rằng “không/chưa” không bị tách khỏi mệnh đề.

### #5 — Lương thử việc Junior cao nhất

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior tối đa 20.000.000 VNĐ/tháng; thử việc nhận 85%, nên kết quả là 17.000.000 VNĐ/tháng.
- **Got / retrieved evidence:** Top-1/2 nêu quy tắc 85%; top-3 chứa bảng lương Junior 12–20 triệu. Không có một chunk đơn lẻ chứa phép tính cuối cùng.
- **Worst metric:** `faithfulness`; average score `0.7269`.
- **Câu trả lời có đúng không?** Có thể đúng nếu LLM kết hợp hai chứng cứ và tính `20.000.000 × 85%`; RAGAS vẫn chỉ ra grounding chưa đủ mạnh.
- **Context có chứa đáp án không?** Có đủ hai thành phần, nhưng bị phân tán giữa nhiều context và top-1 không nêu cấp Junior.
- **Câu hỏi có cần viết lại không?** Không; đây là câu hỏi multi-hop hợp lệ.
- **Error Tree:** Output cần tính toán → evidence phân tán → query rõ → cần multi-hop synthesis/rerank cho bảng lương.
- **Root cause:** Reranking ưu tiên chunk quy tắc thử việc hơn bảng lương theo cấp, còn prompt synthesis không ép mô hình nêu phép tính và nguồn cho từng biến.
- **Suggested fix:** Thêm query decomposition (“Junior max salary” + “probation rate”), preserve table header/row trong chunk, và prompt trả lời theo công thức trước khi kết luận.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?

**Error Tree walkthrough:**

1. **Output đúng?** Chưa thể audit raw answer, nhưng faithfulness là metric thấp nhất.
2. **Context đúng?** Chưa trọn vẹn: top chunk bắt đầu giữa mệnh đề và mất phủ định; đây là lỗi nguy hiểm cho câu yes/no.
3. **Query rewrite OK?** Query đã rõ; không cần rewrite.
4. **Fix ở bước:** M1/M2: expand child sang parent bằng `parent_id`; M3/LLM: ưu tiên context chứa phủ định và trả lời `Có/Không` trước, rồi giải thích.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Ghi `answer`, `contexts`, source, version và per-metric score vào report cho mọi câu hỏi.
- Thêm metadata hiệu lực và regression tests cho 120/90 ngày, 12/8 ký tự, và câu phủ định PVI.
- Benchmark M5: score production giảm so với baseline, nên chạy A/B với raw chunks, contextual prepend, và full enrichment trước khi giữ enrichment mặc định.
