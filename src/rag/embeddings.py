import numpy as np
from typing import List, Optional

class VectorEmbeddingEngine:
    """
    Dense semantic embedding generator using vectorized cosine distance matrices.
    Provides deterministic normalized embeddings for fast in-memory similarity computation.
    """
    def __init__(self, dimension: int = 128, seed: int = 42):
        self.dim = dimension
        np.random.seed(seed)

    def embed_text(self, text: str) -> np.ndarray:
        # Deterministic hashing-based feature representation (portable & zero-dependency)
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        if not words:
            return vec
            
        for idx, word in enumerate(words):
            char_hash = sum(ord(c) * (31 ** i) for i, c in enumerate(word[:12]))
            pos = char_hash % self.dim
            vec[pos] += 1.0 / (1.0 + np.log(idx + 1.0))
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.embed_text(t) for t in texts])

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
