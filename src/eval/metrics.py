import numpy as np
from typing import Dict, Any, List

class RAGTriadEvaluator:
    """
    Evaluates Agentic RAG performance along three essential dimensions:
    1. Context Relevance: Did the retrieval engine pull relevant text?
    2. Groundedness (Faithfulness): Is the answer fully derived from context?
    3. Answer Relevance: Does the output directly satisfy the user's intent?
    """
    @staticmethod
    def evaluate(query: str, retrieved_contexts: List[str], answer: str) -> Dict[str, float]:
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(retrieved_contexts).lower().split()) if retrieved_contexts else set()
        
        # 1. Context Relevance
        if query_words and context_words:
            ctx_overlap = len(query_words.intersection(context_words)) / len(query_words)
            context_relevance = min(1.0, ctx_overlap * 1.5)
        else:
            context_relevance = 0.0
            
        # 2. Groundedness / Faithfulness
        if answer_words and context_words:
            grounded_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
            groundedness = min(1.0, grounded_overlap * 1.3)
        else:
            groundedness = 0.0
            
        # 3. Answer Relevance
        if query_words and answer_words:
            ans_overlap = len(query_words.intersection(answer_words)) / len(query_words)
            answer_relevance = min(1.0, ans_overlap * 1.8)
        else:
            answer_relevance = 0.0

        composite_rag_score = (context_relevance + groundedness + answer_relevance) / 3.0

        return {
            "context_relevance": round(context_relevance, 2),
            "faithfulness": round(groundedness, 2),
            "answer_relevance": round(answer_relevance, 2),
            "composite_rag_score": round(composite_rag_score, 2)
        }
