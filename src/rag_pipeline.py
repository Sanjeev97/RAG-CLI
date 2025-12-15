# rag_pipeline.py

import logging
from typing import List

# Import core components
from rag_models import RetrievedDocument, RAGResponse
from document_store import DocumentStore
from local_llm import LocalLLM

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    The complete RAG system pipeline.
    """
    
    # GUARDRAILS: Topics we refuse to help with
    BLOCKED_KEYWORDS = [
        "hack", "exploit", "malware", "virus",
        "bomb", "weapon", "illegal", "kill"
    ]
    
    def __init__(
        self,
        document_store: DocumentStore,
        llm: LocalLLM
    ):
        self.doc_store = document_store
        self.llm = llm
        
        # System prompt tells the AI how to behave
        self.system_prompt = """You are a helpful AI assistant.
Answer the question based ONLY on the context provided below.
If the context doesn't contain the answer, say "I don't have that information."
Always mention which source document you used."""
    
    def _check_guardrails(self, query: str) -> tuple[bool, str]:
        # ... (implementation from original file, unchanged)
        query_lower = query.lower()
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in query_lower:
                return False, f"Query blocked: contains '{keyword}'"
        return True, ""
    
    def _build_prompt(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> str:
        # ... (implementation from original file, unchanged)
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}: {doc.source}]\n{doc.content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""{self.system_prompt}

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ===
"""
        return prompt
    
    def query(self, user_query: str, top_k: int = 3) -> RAGResponse:
        # ... (implementation from original file, unchanged)
        is_allowed, reason = self._check_guardrails(user_query)
        if not is_allowed:
            return RAGResponse(
                answer=f"I cannot help with that. {reason}",
                sources=[],
                confidence=0.0
            )
        
        documents = self.doc_store.search(user_query, top_k=top_k)
        
        if not documents:
            return RAGResponse(
                answer="I couldn't find any relevant information. Try rephrasing your question.",
                sources=[],
                confidence=0.0
            )
        
        prompt = self._build_prompt(user_query, documents)
        
        logger.info("Generating answer...")
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return RAGResponse(
                answer="Sorry, I encountered an error. Please try again.",
                sources=documents,
                confidence=0.0
            )
        
        avg_score = sum(d.similarity_score for d in documents) / len(documents)
        confidence = min(avg_score, 1.0)
        
        return RAGResponse(
            answer=answer,
            sources=documents,
            confidence=confidence
        )