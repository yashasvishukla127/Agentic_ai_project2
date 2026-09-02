import os
from typing import Optional
from dotenv import load_dotenv
from psycopg import Connection, sql
from psycopg_pool import ConnectionPool
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
        Get a connection from the pool.
        
        Returns:
            Connection: Active database connection from pool
            
        Raises:
            PsycopgError: If connection cannot be obtained from pool
        """
        try:
            conn = self.pool.getconn()
            log.debug("Connection acquired from pool")
            return conn
        except PsycopgError as e:
            log.error("Failed to acquire connection from pool", extra={"error": str(e)})
            raise
    
    def return_connection(self, conn: Connection) -> None:
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return to pool
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
