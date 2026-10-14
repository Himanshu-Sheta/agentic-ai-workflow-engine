import re
from typing import Dict, Any, Tuple
from src.core.state import AgentRole

class QueryIntentRouter:
    """
    Classifies incoming user prompts to determine the optimal execution path:
    - KNOWLEDGE_RETRIEVAL (RAG path)
    - CODE_SYNTHESIS (Code generation & analysis)
    - STRUCTURED_DATA_ANALYSIS (Tabular/SQL reasoning)
    - DIRECT_REASONING (General knowledge synthesis)
    """
    INTENT_KEYWORDS = {
        "KNOWLEDGE_RETRIEVAL": [
            r"\b(what is|explain|define|how does|architecture|policy|documentation|guide)\b",
            r"\b(regulation|whitepaper|research|specification|according to)\b"
        ],
        "CODE_SYNTHESIS": [
            r"\b(code|function|class|algorithm|implement|debug|refactor|script|api|endpoint)\b",
            r"\b(python|typescript|javascript|sql|bug|stack trace|fastapi|react)\b"
        ],
        "STRUCTURED_DATA_ANALYSIS": [
            r"\b(dataset|average|sum|aggregate|metrics|table|csv|dataframe|trend|statistics)\b",
            r"\b(distribution|percentile|outlier|pandas|correlation)\b"
        ]
    }

    def route_query(self, query: str) -> Tuple[str, float]:
        clean_query = query.lower().strip()
        matched_scores: Dict[str, int] = {k: 0 for k in self.INTENT_KEYWORDS}
        
        for intent, patterns in self.INTENT_KEYWORDS.items():
            for pat in patterns:
                matches = re.findall(pat, clean_query)
                matched_scores[intent] += len(matches)

        best_intent = max(matched_scores, key=matched_scores.get)
        confidence = 0.92 if matched_scores[best_intent] > 0 else 0.70
        
        if matched_scores[best_intent] == 0:
            return "DIRECT_REASONING", 0.75
            
        return best_intent, confidence
