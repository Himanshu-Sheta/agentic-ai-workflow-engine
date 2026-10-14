import re
from typing import List, Dict, Any, Tuple

class HallucinationDetector:
    """
    Evaluates generated model outputs against source retrieved context.
    Detects unsubstantiated claims and calculates Groundedness / Faithfulness.
    """
    def __init__(self, groundedness_threshold: float = 0.70):
        self.threshold = groundedness_threshold

    def evaluate_groundedness(self, answer: str, contexts: List[str]) -> Tuple[float, List[str]]:
        if not contexts or not answer.strip():
            return 0.0, ["Missing context or empty answer."]
            
        # Combine all contexts
        full_context = " ".join(contexts).lower()
        context_words = set(re.findall(r'\b[a-z]{3,}\b', full_context))
        
        # Split answer into declarative statements / clauses
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        unsupported_sentences = []
        supported_count = 0
        
        for sent in sentences:
            sent_words = set(re.findall(r'\b[a-z]{3,}\b', sent.lower()))
            if not sent_words:
                continue
            # Overlap coefficient
            overlap = len(sent_words.intersection(context_words)) / len(sent_words)
            if overlap >= 0.45:
                supported_count += 1
            else:
                unsupported_sentences.append(sent)

        groundedness_score = round(supported_count / max(len(sentences), 1), 2)
        return groundedness_score, unsupported_sentences
