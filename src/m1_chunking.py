"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
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

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]
    if not sentences:
        return []

    def sentence_terms(sentence: str) -> set[str]:
        return set(re.findall(r"\w+", sentence.lower()))

    def cosine_like(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        return overlap / sqrt(len(left) * len(right))

    chunks = []
    current_group = [sentences[0]]
    prev_terms = sentence_terms(sentences[0])

    for sentence in sentences[1:]:
        current_terms = sentence_terms(sentence)
        similarity = cosine_like(prev_terms, current_terms)
        if similarity < threshold:
            chunks.append(
                Chunk(
                    text=" ".join(current_group).strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                )
            )
            current_group = []
        current_group.append(sentence)
        prev_terms = current_terms

    if current_group:
        chunks.append(
            Chunk(
                text=" ".join(current_group).strip(),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            )
        )
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents = []
    children = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > parent_size and current:
            pid = f"parent_{len(parents)}"
            parent_text = current.strip()
            parents.append(
                Chunk(
                    text=parent_text,
                    metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                )
            )
            current = paragraph
        else:
            current = candidate

    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(
            Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            )
        )

    for parent in parents:
        pid = parent.metadata["parent_id"]
        start = 0
        while start < len(parent.text):
            child_text = parent.text[start:start + child_size].strip()
            if child_text:
                children.append(
                    Chunk(
                        text=child_text,
                        metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                        parent_id=pid,
                    )
                )
            start += child_size

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_content = ""

    for part in parts:
        if not part:
            continue
        if re.match(r"^#{1,3}\s+.+$", part.strip()):
            if current_header or current_content.strip():
                section_title = current_header or "Preamble"
                section_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(
                    Chunk(
                        text=section_text,
                        metadata={**metadata, "section": section_title, "strategy": "structure"},
                    )
                )
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_header or current_content.strip():
        section_title = current_header or "Preamble"
        section_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        if section_text:
            chunks.append(
                Chunk(
                    text=section_text,
                    metadata={**metadata, "section": section_title, "strategy": "structure"},
                )
            )

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def summarize(chunks: list[Chunk]) -> dict:
        lengths = [len(chunk.text) for chunk in chunks]
        if not lengths:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic_chunks = []
    semantic_chunks = []
    parent_chunks = []
    child_chunks = []
    structure_chunks = []

    for doc in documents:
        doc_text = doc["text"]
        doc_meta = doc.get("metadata", {})
        basic_chunks.extend(chunk_basic(doc_text, metadata=doc_meta))
        semantic_chunks.extend(chunk_semantic(doc_text, metadata=doc_meta))
        parents, children = chunk_hierarchical(doc_text, metadata=doc_meta)
        parent_chunks.extend(parents)
        child_chunks.extend(children)
        structure_chunks.extend(chunk_structure_aware(doc_text, metadata=doc_meta))

    results = {
        "basic": summarize(basic_chunks),
        "semantic": summarize(semantic_chunks),
        "hierarchical": {
            "num_parents": len(parent_chunks),
            "num_children": len(child_chunks),
            **summarize(child_chunks),
        },
        "structure": summarize(structure_chunks),
    }
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
