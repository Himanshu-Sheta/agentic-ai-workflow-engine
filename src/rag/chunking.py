import re
from typing import List, Dict, Any

class SemanticTextChunker:
    """
    Splits continuous documentation and unstructured data into semantic chunks
    with sliding-window overlap and sentence-boundary preservation.
    """
    def __init__(self, chunk_size_words: int = 150, overlap_words: int = 30):
        self.chunk_size = chunk_size_words
        self.overlap = overlap_words

    def chunk_document(self, text: str, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        meta = source_metadata or {"source": "unnamed_document"}
        # Split on sentence boundaries (.!?)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk_words: List[str] = []
        chunk_index = 0
        
        for sentence in sentences:
            words = sentence.split()
            if len(current_chunk_words) + len(words) > self.chunk_size and current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append({
                    "chunk_id": f"{meta.get('source', 'doc')}_{chunk_index}",
                    "text": chunk_text,
                    "word_count": len(current_chunk_words),
                    "metadata": meta
                })
                chunk_index += 1
                # Sliding window overlap
                current_chunk_words = current_chunk_words[-self.overlap:] + words
            else:
                current_chunk_words.extend(words)
                
        if current_chunk_words:
            chunks.append({
                "chunk_id": f"{meta.get('source', 'doc')}_{chunk_index}",
                "text": " ".join(current_chunk_words),
                "word_count": len(current_chunk_words),
                "metadata": meta
            })
            
        return chunks
