"""
Main ingestion script to load, chunk, embed, and store documents.

This script orchestrates the entire ingestion pipeline:
1. Load documents from data directory
2. Chunk documents using specified strategy
3. Embed chunks using OpenAI API
4. Store chunks in PostgreSQL with pgvector

Usage:
    python -m src.ingest.run_ingestion --strategy naive
    python -m src.ingest.run_ingestion --strategy semantic
    python -m src.ingest.run_ingestion --strategy naive --model text-embedding-3-large
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.postgres_client import PostgresClient
from src.ingest.loader import DocumentLoader
from src.ingest.chunker_naive import NaiveChunker
from src.ingest.chunker_semantic import SemanticChunker
from src.ingest.embed_and_store import EmbedderAndStore
from src.observability.logging_config import get_logger
from src.observability.cost_tracker import CostTracker

log = get_logger(__name__)


def run_ingestion(chunking_strategy: str = "naive", embedding_model: str = None):
    """
    Run the complete ingestion pipeline.
    
    Args:
        chunking_strategy: Chunking strategy to use ('naive' or 'semantic')
        embedding_model: OpenAI embedding model to use (optional, defaults to env var or text-embedding-3-small)
        
    Raises:
        ValueError: If chunking strategy is invalid
        Exception: If any step in the pipeline fails
    """
    if chunking_strategy not in ['naive', 'semantic']:
        raise ValueError(f"Invalid chunking strategy: {chunking_strategy}. Must be 'naive' or 'semantic'")
    
    log.info("Starting ingestion pipeline", extra={"chunking_strategy": chunking_strategy, "embedding_model": embedding_model})
    
    try:
        # Initialize PostgreSQL client
        log.info("Initializing PostgreSQL client")
        postgres_client = PostgresClient()
        
        # Check pgvector extension health
        log.info("Checking pgvector extension health")
        postgres_client.check_extension_health()
        
        # Initialize cost tracker
        cost_tracker = CostTracker(embedding_model)
        
        # Initialize embedder and store
        log.info("Initializing embedder and store")
        embedder_store = EmbedderAndStore(
            postgres_client=postgres_client,
            embedding_model=embedding_model,
            cost_tracker=cost_tracker
        )
        
        # Log model information
        model_info = embedder_store.get_model_info()
        log.info("Embedding model configuration", extra=model_info)
        
        # Load documents
        log.info("Loading documents")
        loader = DocumentLoader()
        documents = loader.load_all_documents()
        
        # Select chunker based on strategy
        if chunking_strategy == "naive":
            chunker = NaiveChunker()
        else:
            chunker = SemanticChunker()
        
        # Process each document
        total_chunks_stored = 0
        
        for collection, document in documents.items():
            log.info(
                f"Processing document: {collection}",
                extra={"collection": collection, "source_file": document['source_file']}
            )
            
            # Chunk the document
            chunks = chunker.chunk_document(document)
            log.info(
                f"Document chunked: {collection}",
                extra={"collection": collection, "chunk_count": len(chunks)}
            )
            
            # Embed and store chunks
            stored_count = embedder_store.embed_and_store_chunks(
                chunks=chunks,
                collection=collection,
                chunking_strategy=chunking_strategy,
                validate_first_embedding=True
            )
            
            total_chunks_stored += stored_count
            log.info(
                f"Chunks stored: {collection}",
                extra={"collection": collection, "stored_count": stored_count}
            )
        
        # Log final cost summary
        cost_summary = cost_tracker.get_summary()
        log.info(
            "Ingestion pipeline completed successfully",
            extra={
                "total_chunks_stored": total_chunks_stored,
                "chunking_strategy": chunking_strategy,
                "total_cost_usd": cost_summary['total_cost_usd'],
                "total_tokens": cost_summary['total_tokens']
            }
        )
        
        print(f"\n✅ Ingestion completed successfully!")
        print(f"   Strategy: {chunking_strategy}")
        print(f"   Embedding model: {embedder_store.embedding_model}")
        print(f"   Total chunks stored: {total_chunks_stored}")
        print(f"   Total cost: ${cost_summary['total_cost_usd']:.4f}")
        print(f"   Total tokens: {cost_summary['total_tokens']}")
        
        return total_chunks_stored
        
    except Exception as e:
        log.error("Ingestion pipeline failed", extra={"error": str(e)})
        raise


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="Run document ingestion pipeline")
    parser.add_argument(
        "--strategy",
        choices=["naive", "semantic"],
        default="naive",
        help="Chunking strategy to use (default: naive)"
    )
    parser.add_argument(
        "--model",
        choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        default=None,
        help="OpenAI embedding model to use (default: text-embedding-3-small or OPENAI_EMBEDDING_MODEL env var)"
    )
    
    args = parser.parse_args()
    
    try:
        run_ingestion(chunking_strategy=args.strategy, embedding_model=args.model)
    except Exception as e:
        print(f"\n❌ Ingestion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
