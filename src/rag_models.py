# rag_models.py

from typing import List
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    """
    Represents a document chunk found by search.
    """
    content: str
    source: str
    chunk_id: int
    similarity_score: float


@dataclass
class RAGResponse:
    """
    The final response from the RAG system.
    """
    answer: str
    sources: List[RetrievedDocument]
    confidence: float