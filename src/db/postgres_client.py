import os
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
import psycopg2
from psycopg2 import pool, sql, DatabaseError
from psycopg2.extras import register_uuid
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

from src.observability.logging_config import get_logger

logger = get_logger(__name__)

# Register UUID adapter for psycopg2
register_uuid()


class PostgresClient:
    """PostgreSQL client with connection pooling for eval_runs operations."""
    
    _pool: Optional[pool.ThreadedConnectionPool] = None
    
    @classmethod
    def initialize_pool(cls, min_conn: int = 1, max_conn: int = 10) -> None:
        """Initialize the connection pool."""
        if cls._pool is not None:
            logger.warning("Connection pool already initialized")
            return
        
        try:
            cls._pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                dbname=os.getenv("POSTGRES_DB", "postgres"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                connect_timeout=5
            )
            logger.info("PostgreSQL connection pool initialized", extra={"min_conn": min_conn, "max_conn": max_conn})
        except DatabaseError as e:
            logger.error("Failed to initialize PostgreSQL connection pool", extra={"error": str(e)})
            raise
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.initialize_pool()
        return cls._pool.getconn()
    
    @classmethod
    def return_connection(cls, conn):
        """Return a connection to the pool."""
        if cls._pool is not None:
            cls._pool.putconn(conn)
    
    @classmethod
    def close_pool(cls) -> None:
        """Close all connections in the pool."""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
            logger.info("PostgreSQL connection pool closed")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(DatabaseError),
    reraise=True
)
def insert_eval_run(
    query_id: str,
    strategy: str,
    collection: str,
    faithfulness: Optional[float] = None,
    answer_relevancy: Optional[float] = None,
    context_precision: Optional[float] = None,
    latency_ms: Optional[int] = None,
    cost_usd: Optional[float] = None,
    timestamp: Optional[datetime] = None
) -> uuid.UUID:
    """
    Insert an evaluation run into the eval_runs table.
    
    Args:
        query_id: Identifier for the query being evaluated
        strategy: Retrieval strategy used
        collection: Collection name queried
        faithfulness: Faithfulness score (0-1)
        answer_relevancy: Answer relevancy score (0-1)
        context_precision: Context precision score (0-1)
        latency_ms: Latency in milliseconds
        cost_usd: Cost in USD
        timestamp: Timestamp of the run (defaults to current time)
    
    Returns:
        UUID of the inserted run
    
    Raises:
        DatabaseError: If insertion fails after retries
        ValueError: If required fields are empty or invalid
    """
    # Validate required fields
    if not query_id or not isinstance(query_id, str):
        raise ValueError("query_id must be a non-empty string")
    if not strategy or not isinstance(strategy, str):
        raise ValueError("strategy must be a non-empty string")
    if not collection or not isinstance(collection, str):
        raise ValueError("collection must be a non-empty string")
    
    # Validate score ranges
    for score_name, score_value in [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision)
    ]:
        if score_value is not None and (not isinstance(score_value, (int, float)) or score_value < 0 or score_value > 1):
            logger.warning(f"Invalid {score_name} score, must be between 0 and 1", extra={score_name: score_value})
    
    run_id = uuid.uuid4()
    conn = None
    
    try:
        conn = PostgresClient.get_connection()
        cursor = conn.cursor()
        
        insert_query = sql.SQL("""
            INSERT INTO eval_runs (run_id, query_id, strategy, collection, faithfulness, 
                                  answer_relevancy, context_precision, latency_ms, cost_usd, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """)
        
        cursor.execute(insert_query, (
            run_id,
            query_id,
            strategy,
            collection,
            faithfulness,
            answer_relevancy,
            context_precision,
            latency_ms,
            cost_usd,
            timestamp or datetime.utcnow()
        ))
        
        conn.commit()
        logger.info("Eval run inserted successfully", extra={
            "run_id": str(run_id),
            "query_id": query_id,
            "strategy": strategy,
            "collection": collection,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        })
        
        return run_id
        
    except DatabaseError as e:
        if conn:
            conn.rollback()
        logger.error("Failed to insert eval run", extra={
            "query_id": query_id,
            "strategy": strategy,
            "error": str(e)
        })
        raise
    finally:
        if conn:
            PostgresClient.return_connection(conn)
