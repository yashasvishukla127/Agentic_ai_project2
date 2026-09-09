import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from psycopg import Connection, sql
from psycopg_pool import ConnectionPool, PoolTimeout
from psycopg.errors import Error as PsycopgError

from src.observability.logging_config import get_logger

log = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()


class PostgresClient:
    """
    PostgreSQL client with connection pooling for pgvector-backed RAG system.
    
    Manages database connections via psycopg_pool.ConnectionPool and provides
    health checking for required extensions (pgvector). Credentials are loaded
    from environment variables via python-dotenv.
    
    Features:
        - Connection pooling with configurable min/max sizes
        - Strict 3-second timeout for connection acquisition
        - Critical-level logging on pool timeout for monitoring visibility
        - Extension health checking for pgvector
    
    Environment variables required:
        POSTGRES_HOST: Database host address
        POSTGRES_PORT: Database port (default: 5432)
        POSTGRES_DB: Database name
        POSTGRES_USER: Database user
        POSTGRES_PASSWORD: Database password
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ):
        """
        Initialize PostgreSQL connection pool.
        
        Args:
            host: Database host (defaults to POSTGRES_HOST env var)
            port: Database port (defaults to POSTGRES_PORT env var or 5432)
            dbname: Database name (defaults to POSTGRES_DB env var)
            user: Database user (defaults to POSTGRES_USER env var)
            password: Database password (defaults to POSTGRES_PASSWORD env var)
            min_pool_size: Minimum number of connections in pool
            max_pool_size: Maximum number of connections in pool
            
        Raises:
            ValueError: If required environment variables are not set
            PsycopgError: If connection pool initialization fails
            
        Note:
            Connection pool has a strict 3-second timeout configured. If a connection
            cannot be acquired within this time, a PoolTimeout exception is raised
            and a critical-level log is emitted for monitoring visibility.
        """
        self.host = host or os.getenv("POSTGRES_HOST")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.dbname = dbname or os.getenv("POSTGRES_DB")
        self.user = user or os.getenv("POSTGRES_USER")
        self.password = password or os.getenv("POSTGRES_PASSWORD")
        
        if not all([self.host, self.dbname, self.user, self.password]):
            missing = []
            if not self.host:
                missing.append("POSTGRES_HOST")
            if not self.dbname:
                missing.append("POSTGRES_DB")
            if not self.user:
                missing.append("POSTGRES_USER")
            if not self.password:
                missing.append("POSTGRES_PASSWORD")
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set these in your .env file or environment."
            )
        
        self.conninfo = (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )
        
        try:
            self.pool = ConnectionPool(
                self.conninfo,
                min_size=min_pool_size,
                max_size=max_pool_size,
                open=True,
                timeout=3.0,
            )
            log.info(
                "PostgreSQL connection pool initialized",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "dbname": self.dbname,
                    "user": self.user,
                    "min_pool_size": min_pool_size,
                    "max_pool_size": max_pool_size,
                }
            )
        except PsycopgError as e:
            log.error(
                "Failed to initialize PostgreSQL connection pool",
                extra={"error": str(e), "host": self.host, "port": self.port}
            )
            raise
    
    def get_connection(self) -> Connection:
        """
        Get a connection from the pool with strict 3-second timeout.
        
        If a connection cannot be acquired within 3 seconds, a PoolTimeout
        exception is raised immediately and a critical-level log is emitted
        for monitoring visibility.
        
        Returns:
            Connection: Active database connection from pool
            
        Raises:
            PoolTimeout: If connection cannot be acquired within 3 seconds
            PsycopgError: If connection cannot be obtained from pool
        """
        try:
            conn = self.pool.getconn()
            log.debug("Connection acquired from pool")
            return conn
        except PoolTimeout as e:
            log.critical(
                "Connection pool timeout - unable to acquire connection within 3 seconds",
                extra={
                    "error": str(e),
                    "timeout_seconds": 3.0,
                    "host": self.host,
                    "port": self.port,
                    "dbname": self.dbname,
                }
            )
            raise
        except PsycopgError as e:
            log.error("Failed to acquire connection from pool", extra={"error": str(e)})
            raise
    
    def return_connection(self, conn: Connection) -> None:
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return to pool
            
        Raises:
            PsycopgError: If connection cannot be returned to pool
        """
        try:
            self.pool.putconn(conn)
            log.debug("Connection returned to pool")
        except PsycopgError as e:
            log.error("Failed to return connection to pool", extra={"error": str(e)})
            raise
    
    def check_extension_health(self) -> None:
        """
        Check if pgvector extension is installed and enabled.
        
        Runs SELECT * FROM pg_extension WHERE extname = 'vector' to verify
        the pgvector extension is available. Raises a clear exception if missing.
        
        Raises:
            RuntimeError: If pgvector extension is not found in database
            PsycopgError: If query execution fails
        """
        query = sql.SQL("SELECT * FROM pg_extension WHERE extname = %s")
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, ("vector",))
                    result = cur.fetchone()
                    
                    if result is None:
                        raise RuntimeError(
                            "CRITICAL: pgvector extension is not installed in the database. "
                            "Please run 'CREATE EXTENSION vector;' in your PostgreSQL database "
                            "before using this RAG system. The pgvector extension is required "
                            "for vector similarity search functionality."
                        )
                    
                    log.info("pgvector extension health check passed")
            finally:
                self.return_connection(conn)
                
        except RuntimeError:
            raise
        except PsycopgError as e:
            log.error(
                "Failed to check pgvector extension health",
                extra={"error": str(e)}
            )
            raise
    
    def chunk_already_ingested(self, chunk_id: str) -> bool:
        """
        Check if a chunk already exists in the database by primary key.
        
        Args:
            chunk_id: UUID string of the chunk to check
            
        Returns:
            True if chunk exists, False otherwise
            
        Raises:
            PsycopgError: If query execution fails
        """
        query = sql.SQL("SELECT 1 FROM document_chunks WHERE id = %s")
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (chunk_id,))
                    result = cur.fetchone()
                    exists = result is not None
                    
                    log.debug(
                        "Chunk existence check completed",
                        extra={"chunk_id": chunk_id, "exists": exists}
                    )
                    
                    return exists
            finally:
                self.return_connection(conn)
                
        except PsycopgError as e:
            log.error(
                "Failed to check chunk existence",
                extra={"error": str(e), "chunk_id": chunk_id}
            )
            raise
    
    def insert_chunk(
        self,
        chunk_id: str,
        collection: str,
        content: str,
        embedding: List[float],
        source_file: str,
        chunk_index: int,
        chunking_strategy: str
    ) -> bool:
        """
        Insert a chunk with its embedding into the database.
        
        Uses ON CONFLICT (id) DO NOTHING to handle duplicate chunk IDs gracefully.
        If a chunk with the same ID already exists, the insert is skipped silently.
        
        Args:
            chunk_id: Deterministic UUID string for the chunk
            collection: Collection name ('sales_psychology' or 'mortgage_domain')
            content: Chunk content text
            embedding: Embedding vector
            source_file: Source file name
            chunk_index: Index of the chunk in the document
            chunking_strategy: Chunking strategy used
            
        Returns:
            True if chunk was inserted, False if it already existed
            
        Raises:
            PsycopgError: If database operation fails
            ValueError: If any required parameter is invalid
        """
        if collection not in ['sales_psychology', 'mortgage_domain']:
            raise ValueError(f"Invalid collection: {collection}")
        
        if chunking_strategy not in ['naive', 'semantic', 'hyde']:
            raise ValueError(f"Invalid chunking strategy: {chunking_strategy}")
        
        query = sql.SQL("""
            INSERT INTO document_chunks 
            (id, collection, content, embedding, source_file, chunk_index, chunking_strategy)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """)
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            chunk_id,
                            collection,
                            content,
                            embedding,
                            source_file,
                            chunk_index,
                            chunking_strategy
                        )
                    )
                    conn.commit()
                    
                    # Check if the insert actually happened
                    inserted = cur.rowcount > 0
                    
                    log.debug(
                        "Chunk insert completed",
                        extra={
                            "chunk_id": chunk_id,
                            "collection": collection,
                            "source_file": source_file,
                            "chunk_index": chunk_index,
                            "chunking_strategy": chunking_strategy,
                            "inserted": inserted,
                            "skipped": not inserted
                        }
                    )
                    
                    return inserted
            finally:
                self.return_connection(conn)
                
        except PsycopgError as e:
            log.error(
                "Failed to insert chunk",
                extra={
                    "error": str(e),
                    "chunk_id": chunk_id,
                    "collection": collection,
                    "source_file": source_file
                }
            )
            raise
    
    def create_ingestion_run(
        self,
        source_file: str,
        chunking_strategy: str,
        total_chunks: int
    ) -> str:
        """
        Create a new ingestion run record.
        
        Args:
            source_file: Source file name
            chunking_strategy: Chunking strategy used
            total_chunks: Total number of chunks to process
            
        Returns:
            UUID string of the created ingestion run
            
        Raises:
            PsycopgError: If database operation fails
        """
        query = sql.SQL("""
            INSERT INTO ingestion_runs 
            (source_file, chunking_strategy, total_chunks, chunks_completed, chunks_skipped, chunks_added, status)
            VALUES (%s, %s, %s, 0, 0, 0, 'running')
            RETURNING id
        """)
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (source_file, chunking_strategy, total_chunks))
                    run_id = cur.fetchone()[0]
                    conn.commit()
                    
                    log.info(
                        "Ingestion run created",
                        extra={
                            "run_id": run_id,
                            "source_file": source_file,
                            "chunking_strategy": chunking_strategy,
                            "total_chunks": total_chunks
                        }
                    )
                    
                    return run_id
            finally:
                self.return_connection(conn)
                
        except PsycopgError as e:
            log.error(
                "Failed to create ingestion run",
                extra={"error": str(e), "source_file": source_file, "chunking_strategy": chunking_strategy}
            )
            raise
    
    def get_prior_ingestion_run(
        self,
        source_file: str,
        chunking_strategy: str
    ) -> Optional[Dict[str, any]]:
        """
        Get the most recent prior ingestion run for the same source file and strategy.
        
        Args:
            source_file: Source file name
            chunking_strategy: Chunking strategy used
            
        Returns:
            Dictionary with run info if found, None otherwise
            
        Raises:
            PsycopgError: If query execution fails
        """
        query = sql.SQL("""
            SELECT id, source_file, chunking_strategy, total_chunks, 
                   chunks_completed, chunks_skipped, chunks_added, status, started_at, updated_at
            FROM ingestion_runs
            WHERE source_file = %s AND chunking_strategy = %s
            ORDER BY started_at DESC
            LIMIT 1
        """)
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (source_file, chunking_strategy))
                    result = cur.fetchone()
                    
                    if result:
                        return {
                            "id": result[0],
                            "source_file": result[1],
                            "chunking_strategy": result[2],
                            "total_chunks": result[3],
                            "chunks_completed": result[4],
                            "chunks_skipped": result[5],
                            "chunks_added": result[6],
                            "status": result[7],
                            "started_at": result[8],
                            "updated_at": result[9]
                        }
                    return None
            finally:
                self.return_connection(conn)
                
        except PsycopgError as e:
            log.error(
                "Failed to get prior ingestion run",
                extra={"error": str(e), "source_file": source_file, "chunking_strategy": chunking_strategy}
            )
            raise
    
    def update_ingestion_run(
        self,
        run_id: str,
        chunks_completed: Optional[int] = None,
        chunks_skipped: Optional[int] = None,
        chunks_added: Optional[int] = None,
        status: Optional[str] = None
    ) -> None:
        """
        Update an ingestion run record.
        
        Args:
            run_id: UUID string of the ingestion run
            chunks_completed: New chunks completed count (optional)
            chunks_skipped: New chunks skipped count (optional)
            chunks_added: New chunks added count (optional)
            status: New status (optional)
            
        Raises:
            PsycopgError: If database operation fails
        """
        updates = []
        params = []
        
        if chunks_completed is not None:
            updates.append(sql.SQL("chunks_completed = %s"))
            params.append(chunks_completed)
        
        if chunks_skipped is not None:
            updates.append(sql.SQL("chunks_skipped = %s"))
            params.append(chunks_skipped)
        
        if chunks_added is not None:
            updates.append(sql.SQL("chunks_added = %s"))
            params.append(chunks_added)
        
        if status is not None:
            updates.append(sql.SQL("status = %s"))
            params.append(status)
        
        updates.append(sql.SQL("updated_at = NOW()"))
        params.append(run_id)
        
        query = sql.SQL("UPDATE ingestion_runs SET {} WHERE id = %s").format(
            sql.SQL(", ").join(updates)
        )
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    conn.commit()
                    
                    log.debug(
                        "Ingestion run updated",
                        extra={
                            "run_id": run_id,
                            "chunks_completed": chunks_completed,
                            "status": status
                        }
                    )
            finally:
                self.return_connection(conn)
                
        except PsycopgError as e:
            log.error(
                "Failed to update ingestion run",
                extra={"error": str(e), "run_id": run_id}
            )
            raise
    
    def close(self) -> None:
        """
        Close the connection pool and release all resources.
        
        Should be called when shutting down the application.
        """
        try:
            self.pool.close()
            log.info("PostgreSQL connection pool closed")
        except PsycopgError as e:
            log.error("Error closing connection pool", extra={"error": str(e)})
            raise
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures pool is closed."""
        self.close()
        return False


def get_postgres_client() -> PostgresClient:
    """
    Factory function to create a PostgresClient instance from environment variables.
    
    Returns:
        PostgresClient: Configured client instance
        
    Raises:
        ValueError: If required environment variables are not set
        PsycopgError: If connection pool initialization fails
    """
    return PostgresClient()
