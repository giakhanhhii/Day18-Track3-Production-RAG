"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass
from statistics import mean
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if os.getenv("ENABLE_REAL_MODELS", "0") != "1":
                self._model = False
            else:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name, local_files_only=True)
                except Exception:
                    self._model = False
        return self._model

    def _fallback_score(self, query: str, text: str) -> float:
        query_terms = set(re.findall(r"\w+", query.lower()))
        text_terms = set(re.findall(r"\w+", text.lower()))
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        if model is False:
            scores = [self._fallback_score(query, doc["text"]) for doc in documents]
        else:
            pairs = [(query, doc["text"]) for doc in documents]
            try:
                scores = model.predict(pairs)
            except Exception:
                self._model = False
                scores = [self._fallback_score(query, doc["text"]) for doc in documents]
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        from flashrank import Ranker, RerankRequest
        if self._model is None:
            self._model = Ranker()
        passages = [{"text": d["text"]} for d in documents]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        return [
            RerankResult(
                text=r["text"],
                original_score=documents[i].get("score", 0.0),
                rerank_score=float(r["score"]),
                metadata=documents[i].get("metadata", {}),
                rank=i + 1,
            )
            for i, r in enumerate(results[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": mean(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
