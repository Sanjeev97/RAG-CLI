# rag_pipeline.py

import logging
from typing import List, Optional, Tuple

# Import core components
from rag_models import RetrievedDocument, RAGResponse
from document_store import DocumentStore
from local_llm import LocalLLM

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    The complete RAG system pipeline with query reformulation.
    """
    
    # GUARDRAILS: Topics we refuse to help with
    BLOCKED_KEYWORDS = [
        "hack", "exploit", "malware", "virus",
        "bomb", "weapon", "illegal", "kill"
    ]
    
    # AMBIGUOUS QUERY PATTERNS: Queries that need clarification
    AMBIGUOUS_PATTERNS = [
        "it", "that", "this", "they", "them",  # Pronouns without context
        "tell me more", "explain", "what about",  # Vague requests
    ]
    
    # TOPICS AVAILABLE: For suggesting alternatives
    AVAILABLE_TOPICS = [
        "artificial intelligence",
        "machine learning", 
        "neural networks",
        "deep learning",
        "RAG",
        "space exploration",
        "moon landing",
        "Apollo 11",
        "International Space Station",
        "ISS",
        "Mars exploration"
    ]
    
    def __init__(
        self,
        document_store: DocumentStore,
        llm: LocalLLM,
        enable_reformulation: bool = True
    ):
        self.doc_store = document_store
        self.llm = llm
        self.enable_reformulation = enable_reformulation
        
        # Track conversation history for context
        self.conversation_history = []
        
        # System prompt tells the AI how to behave
        self.system_prompt = """You are a helpful AI assistant.
Answer the question based ONLY on the context provided below.
If the context doesn't contain the answer, say "I don't have that information."
Always mention which source document you used."""
    
    def _check_guardrails(self, query: str) -> tuple[bool, str]:
        """Check if query violates content guardrails."""
        query_lower = query.lower()
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in query_lower:
                return False, f"Query blocked: contains '{keyword}'"
        return True, ""
    
    def _is_ambiguous(self, query: str) -> Tuple[bool, str]:
        """
        Detect if a query is too ambiguous and needs clarification.
        
        Returns:
            (is_ambiguous, reason)
        """
        query_lower = query.lower().strip()
        
        # Check for very short queries (likely incomplete)
        if len(query.split()) <= 2:
            # Check if it's just pronouns or vague words
            for pattern in self.AMBIGUOUS_PATTERNS:
                if pattern in query_lower and len(query.split()) <= 3:
                    return True, f"Query too vague: '{query}'"
        
        # Check for queries without specific topics
        query_words = query_lower.split()
        
        # If query starts with vague phrases without topic
        vague_starters = ["tell me about", "what about", "explain", "describe"]
        for starter in vague_starters:
            if query_lower.startswith(starter):
                # Check if followed by pronoun
                remaining = query_lower[len(starter):].strip()
                if remaining in ["it", "that", "this", "them", "those"]:
                    return True, f"Unclear reference: '{remaining}'"
        
        return False, ""
    
    def _generate_clarification(self, query: str, reason: str) -> str:
        """
        Generate a clarifying question for the user.
        """
        # Get list of available topics
        topics_str = ", ".join([f"'{t}'" for t in self.AVAILABLE_TOPICS[:5]])
        
        clarification = f"""I need more information to help you effectively.

Your query: "{query}"

Could you please clarify what you'd like to know about? For example, I have information on topics like:
{topics_str}, and more.

Please rephrase your question with more specific details."""
        
        return clarification
    
    def _suggest_query_expansion(self, query: str, documents: List[RetrievedDocument]) -> Optional[str]:
        """
        If retrieval returns low-confidence results, suggest query expansion.
        """
        if not documents:
            return None
        
        # If all documents have low similarity scores
        avg_score = sum(d.similarity_score for d in documents) / len(documents)
        
        if avg_score < 0.3:  # Low confidence threshold
            # Extract potential topics from retrieved documents
            sources = list(set(d.source for d in documents))
            
            suggestion = f"""I found some potentially relevant information, but I'm not very confident it answers your question.

Retrieved from: {', '.join(sources)}
Confidence: {avg_score:.0%}

Would you like to:
1. Rephrase your question with more specific terms?
2. Ask about a different topic?
3. See what I found anyway?

Type your choice or a new question."""
            
            return suggestion
        
        return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract key terms from query for better retrieval.
        """
        # Simple keyword extraction (could be enhanced with NLP)
        stop_words = {"what", "is", "are", "the", "a", "an", "how", "when", "where", "who", "why", "tell", "me", "about"}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _reformulate_query(self, query: str) -> str:
        """
        Automatically expand query with synonyms or related terms.
        """
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Add related terms (simple expansion)
        expansions = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nn": "neural network",
            "dl": "deep learning",
        }
        
        expanded_query = query
        for abbrev, full_term in expansions.items():
            if abbrev in query.lower():
                expanded_query += f" {full_term}"
        
        logger.info(f"Query reformulation: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    def _build_prompt(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> str:
        """Build the prompt for the LLM."""
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
    
    def query(
        self, 
        user_query: str, 
        top_k: int = 3,
        skip_reformulation: bool = False
    ) -> RAGResponse:
        """
        Process a user query with intelligent reformulation.
        
        Args:
            user_query: The user's question
            top_k: Number of documents to retrieve
            skip_reformulation: Skip ambiguity checks (for follow-ups)
        """
        # Step 1: Check guardrails
        is_allowed, reason = self._check_guardrails(user_query)
        if not is_allowed:
            return RAGResponse(
                answer=f"I cannot help with that. {reason}",
                sources=[],
                confidence=0.0
            )
        
        # Step 2: Check for ambiguity (if reformulation enabled)
        if self.enable_reformulation and not skip_reformulation:
            is_ambiguous, ambiguity_reason = self._is_ambiguous(user_query)
            
            if is_ambiguous:
                clarification = self._generate_clarification(user_query, ambiguity_reason)
                return RAGResponse(
                    answer=clarification,
                    sources=[],
                    confidence=0.0
                )
        
        # Step 3: Automatic query expansion
        expanded_query = self._reformulate_query(user_query)
        
        # Step 4: Retrieve documents
        documents = self.doc_store.search(expanded_query, top_k=top_k)
        
        if not documents:
            # No documents found - suggest alternatives
            suggestion = f"""I couldn't find any relevant information for: "{user_query}"

I have information about these topics:
{', '.join(self.AVAILABLE_TOPICS[:8])}

Could you try asking about one of these topics, or rephrase your question?"""
            
            return RAGResponse(
                answer=suggestion,
                sources=[],
                confidence=0.0
            )
        
        # Step 5: Check if we should suggest query expansion
        if self.enable_reformulation and not skip_reformulation:
            suggestion = self._suggest_query_expansion(user_query, documents)
            
            if suggestion:
                return RAGResponse(
                    answer=suggestion,
                    sources=documents,
                    confidence=sum(d.similarity_score for d in documents) / len(documents)
                )
        
        # Step 6: Generate answer
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
        
        # Step 7: Calculate confidence
        avg_score = sum(d.similarity_score for d in documents) / len(documents)
        confidence = min(avg_score, 1.0)
        
        # Track in conversation history
        self.conversation_history.append({
            "query": user_query,
            "answer": answer[:100],
            "confidence": confidence
        })
        
        return RAGResponse(
            answer=answer,
            sources=documents,
            confidence=confidence
        )
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation so far.
        """
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = "Conversation Summary:\n"
        for i, item in enumerate(self.conversation_history[-5:], 1):  # Last 5 items
            summary += f"{i}. Q: {item['query'][:50]}... (confidence: {item['confidence']:.0%})\n"
        
        return summary