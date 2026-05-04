"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass
import re

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from underthesea import word_tokenize
except Exception:
    word_tokenize = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        if word_tokenize is not None:
            return word_tokenize(text, format="text")
    except Exception:
        pass
    return " ".join(re.findall(r"\w+", text.lower()))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        tokenized_query = segment_vietnamese(query).split()
        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenized_query)
        else:
            query_terms = set(tokenized_query)
            scores = [
                float(len(query_terms & set(tokens)))
                for tokens in self.corpus_tokens
            ]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=scores[i],
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._documents = []
        self._segmented_documents = []
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("ENABLE_REAL_MODELS", "0") != "1":
                self._encoder = False
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
                except Exception:
                    self._encoder = False
        return self._encoder

    def _fallback_score(self, query: str, text: str) -> float:
        query_terms = set(segment_vietnamese(query).split())
        text_terms = set(segment_vietnamese(text).split())
        if not query_terms or not text_terms:
            return 0.0
        overlap = len(query_terms & text_terms)
        return overlap / max(len(query_terms), 1)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._documents = chunks
        self._segmented_documents = [segment_vietnamese(chunk["text"]) for chunk in chunks]
        encoder = self._get_encoder()
        if self.client is None or encoder is False:
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        try:
            self.client.recreate_collection(
                collection_name=collection,
                vector_params=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [chunk["text"] for chunk in chunks]
            vectors = encoder.encode(texts, show_progress_bar=True)
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
                )
                for i, vector in enumerate(vectors)
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            self.client = None

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        if self.client is None or encoder is False:
            scored = sorted(
                self._documents,
                key=lambda doc: self._fallback_score(query, doc["text"]),
                reverse=True,
            )[:top_k]
            return [
                SearchResult(
                    text=doc["text"],
                    score=self._fallback_score(query, doc["text"]),
                    metadata=doc.get("metadata", {}),
                    method="dense",
                )
                for doc in scored
            ]

        try:
            query_vector = encoder.encode(query).tolist()
            hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
            return [
                SearchResult(
                    text=hit.payload["text"],
                    score=hit.score,
                    metadata=hit.payload,
                    method="dense",
                )
                for hit in hits
            ]
        except Exception:
            self.client = None
            return self.search(query, top_k=top_k, collection=collection)


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in sorted_results[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
