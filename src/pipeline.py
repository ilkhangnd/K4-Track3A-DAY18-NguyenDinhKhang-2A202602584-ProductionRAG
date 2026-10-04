from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


# Avoid retrying an unreachable LLM endpoint for every evaluation question.
_llm_synthesis_available = True


def _document_metadata(document: dict) -> dict:
    """Add lightweight lifecycle metadata used to suppress superseded policies."""
    metadata = dict(document["metadata"])
    source = metadata.get("source", "")
    header = document["text"].split("\n", 4)[:4]
    header_text = "\n".join(header).lower()
    source_lower = source.lower()

    # Old policies label their status in the header. The filename fallback covers
    # the supplied versioned policies whose body later mentions replacement.
    obsolete = "đã thay thế" in header_text or source_lower.endswith("_v1.md")
    metadata["status"] = "obsolete" if obsolete else "current"
    metadata["version"] = (
        "v2023" if "v2023" in source_lower else
        "v2024" if "v2024" in source_lower else
        "v1.0" if "_v1" in source_lower else
        "v2.0" if "_v2" in source_lower else ""
    )
    return metadata


def _parent_context(parent_text: str, metadata: dict) -> str:
    """Attach provenance so synthesis can prefer current, complete context."""
    labels = [f"nguồn: {metadata.get('source', 'unknown')}"]
    if metadata.get("status"):
        labels.append(f"trạng thái: {metadata['status']}")
    if metadata.get("version"):
        labels.append(f"phiên bản: {metadata['version']}")
    return f"[{'; '.join(labels)}]\n\n{parent_text}"


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        metadata = _document_metadata(doc)
        parents, children = chunk_hierarchical(doc["text"], metadata=metadata)
        parent_lookup = {parent.parent_id: parent.text for parent in parents}
        for child in children:
            local_parent_id = child.parent_id or "parent_0"
            global_parent_id = f"{metadata.get('source', 'document')}::{local_parent_id}"
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    **metadata,
                    "parent_id": global_parent_id,
                    "parent_text": _parent_context(parent_lookup[local_parent_id], metadata),
                },
            })
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    enrichment_mode = os.getenv("RAG_ENRICHMENT_MODE", "combined").lower()
    if enrichment_mode == "raw":
        print("\n[2/4] Enrichment disabled for raw-chunk A/B run.", flush=True)
    else:
        print(f"\n[2/4] Enriching {len(all_chunks)} chunks (mode={enrichment_mode})...", flush=True)
        methods = None if enrichment_mode == "combined" else [enrichment_mode]
        enriched = enrich_chunks(all_chunks, methods=methods)
        if enriched:
            all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
            print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
        else:
            print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    global _llm_synthesis_available

    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rank precise child chunks, then return their full parent sections to the
    # LLM. Retrieve more candidates to retain three distinct parent contexts.
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K * 3)
    ranked_docs = reranked if reranked else results
    contexts, seen_parents = [], set()
    for result in ranked_docs:
        metadata = result.metadata
        parent_id = metadata.get("parent_id")
        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)
        contexts.append(metadata.get("parent_text", result.text))
        if len(contexts) == RERANK_TOP_K:
            break

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts and _llm_synthesis_available:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu các đoạn mâu thuẫn, ưu tiên đoạn có trạng thái current và phiên bản mới hơn. Với câu hỏi số liệu, nêu phép tính trước khi kết luận. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            _llm_synthesis_available = False
            print(f"  ⚠️  LLM generation failed; using retrieved context for remaining queries: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
