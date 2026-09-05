import os
from pathlib import Path
from typing import Dict, Optional

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class DocumentLoader:
    """
    Load markdown documents from the data directory.
    
    Reads sales_psychology.md and mortgage_domain.md files and returns
    their content along with metadata about the source file.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize document loader.
        
        Args:
            data_dir: Path to data directory (defaults to ../data relative to this file)
        """
        if data_dir is None:
            # Default to data directory relative to project root
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        log.info("Document loader initialized", extra={"data_dir": str(self.data_dir)})
    
    def load_document(self, filename: str) -> Dict[str, str]:
        """
        Load a single markdown document.
        
        Args:
            filename: Name of the file to load (e.g., 'sales_psychology.md')
            
        Returns:
            Dictionary with 'content' and 'source_file' keys
            
        Raises:
            FileNotFoundError: If the file does not exist
            IOError: If the file cannot be read
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            log.info(
                "Document loaded successfully",
                extra={
                    # "filename" is a reserved LogRecord attribute.
                    "source_filename": filename,
                    "file_path": str(file_path),
                    "content_length": len(content)
                }
            )
            
            return {
                "content": content,
                "source_file": filename
            }
        except IOError as e:
            log.error(
                "Failed to read document file",
                extra={"source_filename": filename, "file_path": str(file_path), "error": str(e)}
            )
            raise
    
    def load_sales_psychology(self) -> Dict[str, str]:
        """
        Load the sales psychology document.
        
        Returns:
            Dictionary with 'content' and 'source_file' keys
            
        Raises:
            FileNotFoundError: If sales_psychology.md does not exist
            IOError: If the file cannot be read
        """
        return self.load_document("sales_psychology.md")
    
    def load_mortgage_domain(self) -> Dict[str, str]:
        """
        Load the mortgage domain document.
        
        Returns:
            Dictionary with 'content' and 'source_file' keys
            
        Raises:
            FileNotFoundError: If mortgage_domain.md does not exist
            IOError: If the file cannot be read
        """
        return self.load_document("mortgage_domain.md")
    
    def load_all_documents(self) -> Dict[str, Dict[str, str]]:
        """
        Load all available documents from the data directory.
        
        Returns:
            Dictionary mapping collection names to document data:
            {
                'sales_psychology': {'content': str, 'source_file': str},
                'mortgage_domain': {'content': str, 'source_file': str}
            }
            
        Raises:
            FileNotFoundError: If any document file does not exist
            IOError: If any file cannot be read
        """
        documents = {}
        
        try:
            documents['sales_psychology'] = self.load_sales_psychology()
        except FileNotFoundError:
            log.warning("sales_psychology.md not found, skipping")
        
        try:
            documents['mortgage_domain'] = self.load_mortgage_domain()
        except FileNotFoundError:
            log.warning("mortgage_domain.md not found, skipping")
        
        if not documents:
            raise ValueError("No documents could be loaded from data directory")
        
        log.info(
            "All documents loaded",
            extra={"document_count": len(documents), "collections": list(documents.keys())}
        )
        
        return documents
