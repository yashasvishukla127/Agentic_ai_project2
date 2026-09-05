from typing import List, Dict

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class SemanticChunker:
    """
    Semantic chunker that respects paragraph boundaries.
    
    Splits text on double newlines (paragraph boundaries) and merges
    short paragraphs to create more meaningful chunks. This approach
    preserves semantic coherence better than fixed-size chunking.
    """
    
    MIN_PARAGRAPH_LENGTH = 100  # Minimum characters for a paragraph to stand alone
    TARGET_CHUNK_LENGTH = 512   # Target length for merged chunks
    
    def __init__(self, min_paragraph_length: int = 100, target_chunk_length: int = 512):
        """
        Initialize the semantic chunker.
        
        Args:
            min_paragraph_length: Minimum characters for a paragraph to stand alone
            target_chunk_length: Target length for merged chunks
        """
        self.MIN_PARAGRAPH_LENGTH = min_paragraph_length
        self.TARGET_CHUNK_LENGTH = target_chunk_length
        
        log.info(
            "Semantic chunker initialized",
            extra={
                "min_paragraph_length": min_paragraph_length,
                "target_chunk_length": target_chunk_length
            }
        )
    
    def chunk(self, text: str, source_file: str) -> List[Dict[str, any]]:
        """
        Split text into semantic chunks based on paragraph boundaries.
        
        Args:
            text: The text to chunk
            source_file: The source file name for metadata
            
        Returns:
            List of chunk dictionaries with 'content', 'source_file', and 'chunk_index' keys
            
        Raises:
            ValueError: If text is empty or None
        """
        if not text:
            raise ValueError("Cannot chunk empty text")
        
        # Split on double newlines to get paragraphs
        paragraphs = text.split('\n\n')
        
        # Clean up paragraphs (remove extra whitespace)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            log.warning("No paragraphs found after splitting", extra={"source_file": source_file})
            return []
        
        # Merge short paragraphs with neighbors
        merged_paragraphs = self._merge_short_paragraphs(paragraphs)
        
        # Create chunks with metadata
        chunks = []
        for i, paragraph in enumerate(merged_paragraphs):
            chunks.append({
                "content": paragraph,
                "source_file": source_file,
                "chunk_index": i
            })
        
        log.info(
            "Text chunked with semantic strategy",
            extra={
                "source_file": source_file,
                "total_chunks": len(chunks),
                "original_paragraphs": len(paragraphs),
                "merged_paragraphs": len(merged_paragraphs),
                "text_length": len(text)
            }
        )
        
        return chunks
    
    def _merge_short_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Merge short paragraphs with neighboring paragraphs.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            List of merged paragraphs
        """
        if not paragraphs:
            return []
        
        merged = []
        current_chunk = paragraphs[0]
        
        for paragraph in paragraphs[1:]:
            # If current chunk is long enough, start a new chunk
            if len(current_chunk) >= self.MIN_PARAGRAPH_LENGTH:
                merged.append(current_chunk)
                current_chunk = paragraph
            else:
                # Merge with current chunk
                current_chunk = current_chunk + "\n\n" + paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, any]]:
        """
        Chunk a document loaded by DocumentLoader.
        
        Args:
            document: Dictionary with 'content' and 'source_file' keys
            
        Returns:
            List of chunk dictionaries with 'content', 'source_file', and 'chunk_index' keys
            
        Raises:
            ValueError: If document is missing required keys or content is empty
        """
        if 'content' not in document or 'source_file' not in document:
            raise ValueError("Document must contain 'content' and 'source_file' keys")
        
        return self.chunk(document['content'], document['source_file'])
