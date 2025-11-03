"""
Text chunking strategies for breaking documents into smaller pieces
"""
import re
from typing import List, Dict, Any
import logging

#from config import RAGConfig

class TextChunker:
    """Split text into chunks using various strategies"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.logger = logging.getLogger(__name__)
        
    def chunk_document(self, document: Dict[str, Any], strategy: str = "recursive") -> List[Dict[str, Any]]:
        """
        Chunk a document using the specified strategy
        
        Args:
            document: Document dictionary with content and metadata
            strategy: Chunking strategy ('recursive', 'sentence', 'paragraph')
            
        Returns:
            List of chunk dictionaries
        """
        content = document["content"]
        metadata = document["metadata"]
        
        if strategy == "recursive":
            chunks = self._recursive_chunk(content)
        elif strategy == "sentence":
            chunks = self._sentence_chunk(content)
        elif strategy == "paragraph":
            chunks = self._paragraph_chunk(content)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Add metadata to each chunk
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })
            
            chunk_docs.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
            
        self.logger.info(f"Created {len(chunk_docs)} chunks from {metadata['filename']}")
        return chunk_docs
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Recursively split text by different separators
        
        This is the most commonly used chunking strategy in RAG systems.
        It tries to keep semantically related content together.
        """
        # Define separators in order of preference
        separators = [
            "\n\n",    # Double newline (paragraphs)
            "\n",      # Single newline
            ". ",      # Sentence endings
            "! ",      # Exclamation
            "? ",      # Question
            "; ",      # Semicolon
            ", ",      # Comma
            " ",       # Space
            ""         # Character level (last resort)
        ]
        
        chunks = [text]
        
        for separator in separators:
            new_chunks = []
            
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    # Split by current separator
                    split_chunks = self._split_by_separator(chunk, separator)
                    new_chunks.extend(split_chunks)
            
            chunks = new_chunks
            
            # Check if all chunks are within size limit
            if all(len(chunk) <= self.chunk_size for chunk in chunks):
                break
        
        # Remove empty chunks and apply overlap
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        chunks = self._apply_overlap(chunks)
        
        return chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator and recombine to meet chunk size requirements"""
        if not separator:
            # Character-level splitting as last resort
            return [text[i:i + self.chunk_size] 
                   for i in range(0, len(text), self.chunk_size)]
        
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Check if adding this part would exceed chunk size
            test_chunk = current_chunk + separator + part if current_chunk else part
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with current part
                if len(part) <= self.chunk_size:
                    current_chunk = part
                else:
                    # Part is too large, need to split further
                    chunks.append(part)  # Will be handled in next iteration
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _sentence_chunk(self, text: str) -> List[str]:
        """Split text by sentences"""
        # Simple sentence splitting using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return self._apply_overlap(chunks)
    
    def _paragraph_chunk(self, text: str) -> List[str]:
        """Split text by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + "\n\n" + paragraph) <= self.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return self._apply_overlap(chunks)
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks"""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                
                # Find a good break point for overlap (try to break at word boundary)
                words = overlap_text.split()
                if len(words) > 1:
                    overlap_text = " ".join(words)
                
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk["content"]) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "total_characters": sum(chunk_sizes)
        }
