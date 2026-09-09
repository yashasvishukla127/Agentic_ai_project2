"""
Test ingestion idempotency and crash recovery.

This test verifies that:
1. Ingestion can be resumed after a crash
2. Already-ingested chunks are skipped on re-run
3. OpenAI embedding API is only called for new chunks
4. Final state has exactly 10 chunks (no duplicates)
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.db.postgres_client import PostgresClient
from src.ingest.embed_and_store import EmbedderAndStore


@pytest.fixture
def postgres_client():
    """
    Create a PostgreSQL client for testing.
    
    Skips test if required environment variables are not set.
    """
    required_vars = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        pytest.skip(
            f"Skipping test: missing required environment variables: {', '.join(missing_vars)}. "
            "Set these in your .env file or environment to run this test."
        )
    
    try:
        client = PostgresClient(max_pool_size=5)
    except ValueError as e:
        pytest.skip(f"Skipping test: {str(e)}")
    except Exception as e:
        pytest.skip(f"Skipping test: Failed to initialize PostgreSQL client: {str(e)}")
    
    yield client
    client.close()


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """
    Create 10 sample chunks for testing.
    
    Returns:
        List of 10 chunk dictionaries with 'content', 'source_file', 'chunk_index'
    """
    chunks = []
    for i in range(10):
        chunks.append({
            'content': f'This is chunk {i} with some sample content for testing.',
            'source_file': 'test_document.txt',
            'chunk_index': i
        })
    return chunks


@pytest.fixture
def mock_embedding():
    """
    Create a mock embedding vector of the expected dimension.
    
    Returns:
        List of 1536 float values (expected embedding dimension)
    """
    return [0.1] * 1536


def test_ingestion_crash_recovery(postgres_client, sample_chunks, mock_embedding):
    """
    Test that ingestion can recover from a crash and only processes new chunks.
    
    Test scenario:
    1. First run: ingest 10 chunks, but simulate crash after chunk 6
    2. Second run: rerun ingestion on the same 10 chunks
    3. Assert exactly 10 rows exist (not 16 - no duplicates)
    4. Assert OpenAI embedding API was called exactly 4 times on second run (chunks 7-10 only)
    """
    # Clean up any existing test data
    cleanup_query = "DELETE FROM document_chunks WHERE source_file = %s"
    conn = postgres_client.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(cleanup_query, ('test_document.txt',))
            conn.commit()
    finally:
        postgres_client.return_connection(conn)
    
    # Create embedder and store with mocked OpenAI client
    with patch('src.ingest.embed_and_store.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock the embeddings.create response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data[0].embedding = mock_embedding
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Set a fake API key to bypass validation
        os.environ['OPENAI_API_KEY'] = 'test_key'
        
        embedder_store = EmbedderAndStore(
            postgres_client=postgres_client,
            openai_api_key='test_key',
            embedding_model='text-embedding-3-small'
        )
        
        # Mock ingestion run tracking methods to avoid dependency on ingestion_runs table
        with patch.object(postgres_client, 'get_prior_ingestion_run', return_value=None):
            with patch.object(postgres_client, 'create_ingestion_run', return_value='test-run-id'):
                with patch.object(postgres_client, 'update_ingestion_run'):
                    # Track call count for the mock
                    embedding_call_count = {'count': 0}
                    
                    def track_embedding_calls(*args, **kwargs):
                        embedding_call_count['count'] += 1
                        # Simulate crash after chunk 6 (i.e., on the 7th call)
                        if embedding_call_count['count'] == 7:
                            raise RuntimeError("Simulated crash after chunk 6")
                        return mock_embedding_response
                    
                    mock_client.embeddings.create.side_effect = track_embedding_calls
                    
                    # First run: should crash after chunk 6
                    with pytest.raises(RuntimeError, match="Simulated crash after chunk 6"):
                        embedder_store.embed_and_store_chunks(
                            chunks=sample_chunks,
                            collection='sales_psychology',
                            chunking_strategy='naive',
                            validate_first_embedding=False
                        )
                    
                    # Verify that 6 chunks were stored before crash
                    count_query = "SELECT COUNT(*) FROM document_chunks WHERE source_file = %s"
                    conn = postgres_client.get_connection()
                    try:
                        with conn.cursor() as cur:
                            cur.execute(count_query, ('test_document.txt',))
                            chunk_count = cur.fetchone()[0]
                    finally:
                        postgres_client.return_connection(conn)
                    
                    assert chunk_count == 6, f"Expected 6 chunks after crash, got {chunk_count}"
                    
                    # Reset the mock for the second run
                    mock_client.embeddings.create.side_effect = None
                    mock_client.embeddings.create.reset_mock()
                    embedding_call_count['count'] = 0
                    
                    # Second run: should only process chunks 7-10 (4 chunks)
                    result = embedder_store.embed_and_store_chunks(
                        chunks=sample_chunks,
                        collection='sales_psychology',
                        chunking_strategy='naive',
                        validate_first_embedding=False
                    )
                    
                    # Verify final state: exactly 10 chunks (no duplicates)
                    conn = postgres_client.get_connection()
                    try:
                        with conn.cursor() as cur:
                            cur.execute(count_query, ('test_document.txt',))
                            final_chunk_count = cur.fetchone()[0]
                    finally:
                        postgres_client.return_connection(conn)
                    
                    assert final_chunk_count == 10, (
                        f"Expected exactly 10 chunks after recovery, got {final_chunk_count}. "
                        "This suggests duplicates were created."
                    )
                    
                    # Verify OpenAI embedding API was called exactly 4 times (chunks 7-10 only)
                    assert mock_client.embeddings.create.call_count == 4, (
                        f"Expected OpenAI embedding API to be called 4 times on second run, "
                        f"but it was called {mock_client.embeddings.create.call_count} times. "
                        "This suggests already-ingested chunks were re-embedded."
                    )
                    
                    # Verify the result indicates 4 inserted and 6 skipped
                    assert result['inserted'] == 4, (
                        f"Expected 4 chunks to be inserted on second run, got {result['inserted']}"
                    )
                    assert result['skipped'] == 6, (
                        f"Expected 6 chunks to be skipped on second run, got {result['skipped']}"
                    )
                    
                    print(f"\nIngestion crash recovery test passed:")
                    print(f"   - 6 chunks stored before crash")
                    print(f"   - 4 chunks inserted on recovery (chunks 7-10)")
                    print(f"   - 6 chunks skipped (already ingested)")
                    print(f"   - Final total: 10 chunks (no duplicates)")
                    print(f"   - OpenAI API calls on second run: 4 (optimal)")
