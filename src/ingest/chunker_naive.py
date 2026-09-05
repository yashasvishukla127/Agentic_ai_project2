from typing import List, Dict

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class NaiveChunker:
    """
    Fixed-size chunker that splits text into 512-character chunks.
    
    This is a simple chunking strategy that doesn't respect semantic boundaries.
    It's useful as a baseline for comparison with more sophisticated chunking methods.
    """
    
    CHUNK_SIZE = 512
    
    def __init__(self):
        """Initialize the naive chunker."""
        log.info("Naive chunker initialized", extra={"chunk_size": self.CHUNK_SIZE})
    
    def chunk(self, text: str, source_file: str) -> List[Dict[str, any]]:
        """
        Split text into fixed-size chunks.
        
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
        
        chunks = []
        total_chunks = (len(text) + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        
        for i in range(total_chunks):
            start = i * self.CHUNK_SIZE
            end = start + self.CHUNK_SIZE
            chunk_content = text[start:end]
            
            chunks.append({
                "content": chunk_content,
                "source_file": source_file,
                "chunk_index": i
            })
        
        log.info(
            "Text chunked with naive strategy",
            extra={
                "source_file": source_file,
                "total_chunks": len(chunks),
                "chunk_size": self.CHUNK_SIZE,
                "text_length": len(text)
            }
        )
        
        return chunks
    
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
