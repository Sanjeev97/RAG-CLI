# document_store.py

import os
import json
import logging
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Import data class
from rag_models import RetrievedDocument

logger = logging.getLogger(__name__)

# ============================================================
# DOCUMENT STORE - Stores and searches documents
# ============================================================

class DocumentStore:
    """
    Stores documents and finds relevant ones for a query.
    
    How it works:
    1. Takes your documents and splits them into chunks
    2. Converts each chunk into numbers (embeddings)
    3. When you search, converts your query to numbers
    4. Finds chunks with similar numbers = relevant documents!
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document store.
        
        Args:
            embedding_model: Which model to use for converting text to numbers.
                           "all-MiniLM-L6-v2" is small, fast, and good quality.
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        
        # Load the model that converts text to numbers
        # This downloads the model the first time (~90MB)
        self.encoder = SentenceTransformer(embedding_model)
        
        # Get the size of embeddings (384 numbers for MiniLM)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Storage for our documents (list of dictionaries)
        self.documents: List[Dict] = []
        
        # FAISS index for fast searching
        # IndexFlatIP = uses Inner Product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        logger.info(f"Document store ready! Embedding size: {self.embedding_dim}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks.
        
        Why chunk?
        - AI models have limited memory (context window)
        - Smaller chunks = more precise search results
        
        Args:
            text: The full document text
            chunk_size: Maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Split by paragraphs (double newline)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph keeps us under the limit, add it
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk and start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        # Don't forget the last chunk!
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no chunks were created, just use the whole text
        if not chunks:
            chunks = [text[:chunk_size]]
        
        return chunks
    
    def add_documents(self, texts: List[str], sources: List[str]) -> int:
        """
        Add documents to the store.
        
        Args:
            texts: List of document contents (the actual text)
            sources: List of source names (like filenames)
        
        Returns:
            Number of chunks added
        """
        all_chunks = []
        all_metadata = []
        
        # Process each document
        for text, source in zip(texts, sources):
            logger.info(f"Processing: {source}")
            
            # Split into chunks
            chunks = self._chunk_text(text)
            
            # Store each chunk with its metadata
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "content": chunk,
                    "source": source,
                    "chunk_id": len(self.documents) + len(all_metadata)
                })
        
        if not all_chunks:
            logger.warning("No chunks to add!")
            return 0
        
        # Convert all chunks to embeddings (numbers)
        logger.info(f"Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.encoder.encode(
            all_chunks,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=True
        )
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store the metadata
        self.documents.extend(all_metadata)
        
        logger.info(f"Added {len(all_chunks)} chunks. Total documents: {len(self.documents)}")
        return len(all_chunks)
    
    def search(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """
        Search for documents relevant to a query.
        
        Args:
            query: The user's question
            top_k: How many results to return
        
        Returns:
            List of relevant documents, sorted by relevance
        """
        if not self.documents:
            logger.warning("No documents to search!")
            return []
        
        # Convert query to embedding (numbers)
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search FAISS index for similar embeddings
        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents))
        )
        
        # Build results list
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for no results
                continue
            
            doc = self.documents[idx]
            results.append(RetrievedDocument(
                content=doc["content"],
                source=doc["source"],
                chunk_id=doc["chunk_id"],
                similarity_score=float(score)
            ))
        
        return results
    
    def save(self, path: str):
        """Save the index to disk so we don't have to rebuild it."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.json", "w") as f:
            json.dump(self.documents, f)
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load a previously saved index."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.json", "r") as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} documents from {path}")