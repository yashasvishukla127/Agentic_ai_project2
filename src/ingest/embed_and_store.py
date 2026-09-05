import argparse
import os
from typing import Any, Dict, List, Optional, Protocol
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI, OpenAIError, APITimeoutError, RateLimitError
from psycopg import sql
from psycopg.errors import Error as PsycopgError

from src.db.postgres_client import PostgresClient
from src.ingest.chunker_naive import NaiveChunker
from src.ingest.chunker_semantic import SemanticChunker
from src.ingest.loader import DocumentLoader
from src.observability.logging_config import get_logger

log = get_logger(__name__)


class EmbeddingCostTracker(Protocol):
    """Interface for optional embedding-cost instrumentation."""

    def track_embedding_cost(self, text_length: int) -> None:
        """Record the cost of embedding a text payload."""


class EmbedderAndStore:
    """
    Embed text chunks using OpenAI and store them in PostgreSQL with pgvector.
    
    Uses OpenAI's embedding models (default: text-embedding-3-small) to generate 
    embeddings and stores them in the document_chunks table with associated metadata.
    Includes retry logic, error handling, and cost tracking.
    
    Supported models:
        - text-embedding-3-small (1536 dimensions) - default
        - text-embedding-3-large (3072 dimensions)
        - text-embedding-ada-002 (1536 dimensions)
    
    Model can be set via:
        - embedding_model parameter in __init__
        - OPENAI_EMBEDDING_MODEL environment variable
        - Defaults to text-embedding-3-small
    """
    
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    EXPECTED_EMBEDDING_DIM = 1536
    MAX_RETRIES = 3
    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        postgres_client: PostgresClient,
        openai_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        cost_tracker: Optional[EmbeddingCostTracker] = None
    ):
        """
        Initialize embedder and store.
        
        Args:
            postgres_client: PostgreSQL client with connection pool
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            embedding_model: OpenAI embedding model to use (defaults to text-embedding-3-small)
                           Supported: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
            cost_tracker: Optional cost tracker for monitoring API costs
            
        Raises:
            ValueError: If OpenAI API key is not provided or embedding model is invalid
        """
        self.postgres_client = postgres_client
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Set embedding model with validation
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", self.DEFAULT_EMBEDDING_MODEL)
        
        if self.embedding_model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {self.embedding_model}. "F
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        # Validate dimension matches expected for the model
        expected_dim = self.SUPPORTED_MODELS[self.embedding_model]
        if expected_dim != self.EXPECTED_EMBEDDING_DIM:
            log.warning(
                "Embedding dimension mismatch",
                extra={
                    "model": self.embedding_model,
                    "model_expected_dim": expected_dim,
                    "configured_expected_dim": self.EXPECTED_EMBEDDING_DIM
                }
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.cost_tracker = cost_tracker
        
        log.info(
            "Embedder and store initialized",
            extra={
                "model": self.embedding_model,
                "expected_dim": self.EXPECTED_EMBEDDING_DIM,
                "model_expected_dim": self.SUPPORTED_MODELS[self.embedding_model],
                "api_key_present": bool(self.api_key),
                "cost_tracker_enabled": cost_tracker is not None
            }
        )
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current embedding model configuration.
        
        Returns:
            Dictionary with model information including name, dimensions, and cost per 1K tokens
        """
        model_costs = {
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.00010,
        }
        
        return {
            "model": self.embedding_model,
            "dimensions": self.SUPPORTED_MODELS[self.embedding_model],
            "cost_per_1k_tokens": model_costs.get(self.embedding_model, 0.0),
            "supported_models": list(self.SUPPORTED_MODELS.keys()),
        }
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        reraise=True
    )
    def _embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string using OpenAI API with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float embedding values
            
        Raises:
            OpenAIError: If embedding API call fails after retries
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Track cost if cost tracker is available
            if self.cost_tracker:
                self.cost_tracker.track_embedding_cost(len(text))
            
            log.debug(
                "Text embedded successfully",
                extra={
                    "model": self.embedding_model,
                    "text_length": len(text),
                    "embedding_dim": len(embedding)
                }
            )
            
            return embedding
            
        except APITimeoutError as e:
            log.error("OpenAI API timeout during embedding", extra={"error": str(e)})
            raise
        except RateLimitError as e:
            log.error("OpenAI API rate limit exceeded during embedding", extra={"error": str(e)})
            raise
        except OpenAIError as e:
            log.error("OpenAI API error during embedding", extra={"error": str(e)})
            raise
    
    def _validate_embedding_dimension(self, embedding: List[float]) -> None:
        """
        Validate that embedding has the expected dimension.
        
        Args:
            embedding: Embedding vector to validate
            
        Raises:
            ValueError: If embedding dimension is not exactly 1536
        """
        if len(embedding) != self.EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.EXPECTED_EMBEDDING_DIM}, "
                f"got {len(embedding)}. This indicates a problem with the OpenAI API response."
            )
    
    def _store_chunk(
        self,
        chunk: Dict[str, Any],
        embedding: List[float],
        collection: str,
        chunking_strategy: str
    ) -> None:
        """
        Store a single chunk with its embedding in the database.
        
        Args:
            chunk: Chunk dictionary with 'content', 'source_file', 'chunk_index'
            embedding: Embedding vector
            collection: Collection name ('sales_psychology' or 'mortgage_domain')
            chunking_strategy: Chunking strategy ('naive', 'semantic', or 'hyde')
            
        Raises:
            PsycopgError: If database operation fails
            ValueError: If chunk is missing required fields
        """
        required_fields = ['content', 'source_file', 'chunk_index']
        for field in required_fields:
            if field not in chunk:
                raise ValueError(f"Chunk missing required field: {field}")
        
        try:
            conn = self.postgres_client.get_connection()
            try:
                with conn.cursor() as cur:
                    query = sql.SQL("""
                        INSERT INTO document_chunks 
                        (collection, content, embedding, source_file, chunk_index, chunking_strategy)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """)
                    
                    cur.execute(
                        query,
                        (
                            collection,
                            chunk['content'],
                            embedding,
                            chunk['source_file'],
                            chunk['chunk_index'],
                            chunking_strategy
                        )
                    )
                    
                    conn.commit()
                    
                    log.debug(
                        "Chunk stored successfully",
                        extra={
                            "collection": collection,
                            "source_file": chunk['source_file'],
                            "chunk_index": chunk['chunk_index'],
                            "chunking_strategy": chunking_strategy
                        }
                    )
            finally:
                self.postgres_client.return_connection(conn)
                
        except PsycopgError as e:
            log.error(
                "Failed to store chunk in database",
                extra={
                    "error": str(e),
                    "collection": collection,
                    "source_file": chunk.get('source_file'),
                    "chunk_index": chunk.get('chunk_index')
                }
            )
            raise
    
    def embed_and_store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection: str,
        chunking_strategy: str,
        validate_first_embedding: bool = True
    ) -> int:
        """
        Embed and store chunks in the database.
        
        Args:
            chunks: List of chunk dictionaries with 'content', 'source_file', 'chunk_index'
            collection: Collection name ('sales_psychology' or 'mortgage_domain')
            chunking_strategy: Chunking strategy ('naive', 'semantic', or 'hyde')
            validate_first_embedding: Whether to validate first embedding dimension
            
        Returns:
            Number of chunks successfully stored
            
        Raises:
            ValueError: If chunks is empty, collection invalid, or embedding dimension wrong
            OpenAIError: If embedding API call fails
            PsycopgError: If database operation fails
        """
        if not chunks:
            raise ValueError("Cannot embed and store empty chunks list")
        
        if collection not in ['sales_psychology', 'mortgage_domain']:
            raise ValueError(f"Invalid collection: {collection}. Must be 'sales_psychology' or 'mortgage_domain'")
        
        if chunking_strategy not in ['naive', 'semantic', 'hyde']:
            raise ValueError(f"Invalid chunking strategy: {chunking_strategy}. Must be 'naive', 'semantic', or 'hyde'")
        
        log.info(
            "Starting embed and store process",
            extra={
                "collection": collection,
                "chunking_strategy": chunking_strategy,
                "total_chunks": len(chunks)
            }
        )
        
        stored_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Embed the chunk
                embedding = self._embed_text(chunk['content'])
                
                # Validate first embedding dimension if requested
                if validate_first_embedding and i == 0:
                    self._validate_embedding_dimension(embedding)
                    log.info(
                        "First embedding dimension validated",
                        extra={"dimension": len(embedding)}
                    )
                
                # Store the chunk
                self._store_chunk(chunk, embedding, collection, chunking_strategy)
                stored_count += 1
                
            except (ValueError, OpenAIError, PsycopgError) as e:
                log.error(
                    "Failed to process chunk",
                    extra={
                        "error": str(e),
                        "chunk_index": i,
                        "source_file": chunk.get('source_file'),
                        "collection": collection
                    }
                )
                raise
        
        log.info(
            "Embed and store process completed",
            extra={
                "collection": collection,
                "chunking_strategy": chunking_strategy,
                "stored_count": stored_count,
                "total_chunks": len(chunks)
            }
        )
        
        return stored_count


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Chunk, embed, and store a knowledge-base document."
    )
    parser.add_argument(
        "--collection",
        required=True,
        choices=("sales_psychology", "mortgage_domain"),
        help="Collection and matching markdown source document to ingest.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("naive", "semantic"),
        help="Chunking strategy to use before embedding.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing the source markdown files (defaults to ./data).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the end-to-end ingestion workflow from the command line."""
    args = _build_argument_parser().parse_args(argv)

    postgres_client: Optional[PostgresClient] = None
    try:
        loader = DocumentLoader(data_dir=args.data_dir)
        document = loader.load_document(f"{args.collection}.md")
        chunker = SemanticChunker() if args.strategy == "semantic" else NaiveChunker()
        chunks = chunker.chunk_document(document)

        postgres_client = PostgresClient()
        postgres_client.check_extension_health()

        stored_count = EmbedderAndStore(postgres_client).embed_and_store_chunks(
            chunks=chunks,
            collection=args.collection,
            chunking_strategy=args.strategy,
        )
        log.info(
            "Document ingestion completed",
            extra={
                "collection": args.collection,
                "chunking_strategy": args.strategy,
                "stored_count": stored_count,
            },
        )
        return 0
    except (OSError, ValueError, OpenAIError, PsycopgError, RuntimeError) as error:
        log.error("Document ingestion failed", extra={"error": str(error)})
        return 1
    finally:
        if postgres_client is not None:
            postgres_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
