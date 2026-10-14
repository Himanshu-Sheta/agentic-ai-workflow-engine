import math
import numpy as np
from typing import List, Dict, Any, Tuple
from src.rag.embeddings import VectorEmbeddingEngine

class HybridBM25DenseRetriever:
    """
    Production Hybrid Information Retrieval system.
    Combines:
    1. Sparse Lexical Search (BM25 Okapi Algorithm)
    2. Dense Semantic Search (Vector Cosine Similarity)
    Fused using Reciprocal Rank Fusion (RRF) for optimal recall and precision.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.embedder = VectorEmbeddingEngine()
        self.documents: List[Dict[str, Any]] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf_cache: Dict[str, float] = {}

    def index_documents(self, docs: List[Dict[str, Any]]):
        self.documents = docs
        self.doc_lengths = [len(d["text"].split()) for d in docs]
        self.avg_doc_len = sum(self.doc_lengths) / max(len(docs), 1)
        
        # Build IDF cache
        N = len(docs)
        df_counts: Dict[str, int] = {}
        for d in docs:
            unique_terms = set(d["text"].lower().split())
            for t in unique_terms:
                df_counts[t] = df_counts.get(t, 0) + 1
                
        for t, count in df_counts.items():
            self.idf_cache[t] = math.log(1.0 + (N - count + 0.5) / (count + 0.5))
            
        # Dense index
        texts = [d["text"] for d in docs]
        self.doc_embeddings = self.embedder.batch_embed(texts)

    def _score_bm25(self, query_terms: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_words = self.documents[doc_idx]["text"].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = {}
        for w in doc_words:
            word_counts[w] = word_counts.get(w, 0) + 1

        for term in query_terms:
            tf = word_counts.get(term, 0)
            if tf > 0:
                idf = self.idf_cache.get(term, 0.1)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.documents:
            return []
            
        query_terms = query.lower().split()
        query_vec = self.embedder.embed_text(query)
        
        # 1. Sparse BM25 scoring & ranking
        bm25_scores = [self._score_bm25(query_terms, i) for i in range(len(self.documents))]
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        # 2. Dense Cosine similarity scoring & ranking
        dense_scores = [
            VectorEmbeddingEngine.cosine_similarity(query_vec, self.doc_embeddings[i])
            for i in range(len(self.documents))
        ]
        dense_ranked = sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True)
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {i: 0.0 for i in range(len(self.documents))}
        for rank, doc_idx in enumerate(bm25_ranked):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)
        for rank, doc_idx in enumerate(dense_ranked):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)
            
        top_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            res = dict(self.documents[idx])
            res["rrf_score"] = round(rrf_scores[idx], 5)
            res["bm25_score"] = round(bm25_scores[idx], 3)
            res["dense_similarity"] = round(dense_scores[idx], 3)
            results.append(res)
            
        return results
