from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


@lru_cache(maxsize=1)
def _semantic_encoder():
    """Load MiniLM once per process; it is reused across every document."""
    from sentence_transformers import SentenceTransformer

    # The lab setup pre-downloads this model. Local-only loading avoids a network
    # round trip for every pytest/production run and gives a clear cache error if
    # the setup step was skipped.
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    # Keep the delimiters' preceding punctuation, while treating blank lines as a
    # semantic boundary candidate as well.  Empty strings otherwise produce an
    # invalid embedding batch for short/whitespace-only documents.
    sentences = [sentence.strip() for sentence in re.split(
        r"(?<=[.!?])\s+|\n\n", text
    ) if sentence.strip()]
    if not sentences:
        return []

    from numpy import dot
    from numpy.linalg import norm
    model = _semantic_encoder()
    embeddings = model.encode(sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for index in range(1, len(sentences)):
        previous, current = embeddings[index - 1], embeddings[index]
        similarity = float(dot(previous, current) / (norm(previous) * norm(current) + 1e-9))
        if similarity < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    metadata = metadata or {}

    def split_to_limit(value: str, limit: int) -> list[str]:
        """Split on paragraph/word boundaries where possible, never exceed limit."""
        paragraphs = [paragraph.strip() for paragraph in value.split("\n\n") if paragraph.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            # A long unbroken paragraph still has to obey the configured limit.
            while len(paragraph) > limit:
                cut = paragraph.rfind(" ", 0, limit + 1)
                if cut <= 0:
                    cut = limit
                units.append(paragraph[:cut].strip())
                paragraph = paragraph[cut:].strip()
            if paragraph:
                units.append(paragraph)

        chunks: list[str] = []
        current = ""
        for unit in units:
            separator = "\n\n" if current else ""
            if current and len(current) + len(separator) + len(unit) > limit:
                chunks.append(current)
                current = unit
            else:
                current += separator + unit
        if current:
            chunks.append(current)
        return chunks

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_index, parent_text in enumerate(split_to_limit(text, parent_size)):
        parent_id = f"parent_{parent_index}"
        parent_metadata = {
            **metadata,
            "chunk_type": "parent",
            "parent_id": parent_id,
            "chunk_index": parent_index,
        }
        parents.append(Chunk(text=parent_text, metadata=parent_metadata, parent_id=parent_id))

        for child_index, child_text in enumerate(split_to_limit(parent_text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "chunk_index": child_index,
                },
                parent_id=parent_id,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    # re.split retains every header in an alternating list, which makes it easy to
    # attach its complete body (including tables and lists) without line splitting.
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []

    preamble = parts[0].strip()
    if preamble:
        chunks.append(Chunk(
            text=preamble,
            metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0},
        ))

    for index in range(1, len(parts), 2):
        header = parts[index].strip()
        body = parts[index + 1].strip() if index + 1 < len(parts) else ""
        section = re.sub(r"^#{1,3}\s+", "", header).strip()
        chunk_text = f"{header}\n\n{body}".strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "section": section,
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            ))

    # Documents without Markdown headers remain usable as one logical section.
    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0},
        ))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
