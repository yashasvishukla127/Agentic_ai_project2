"""
Setup script to initialize the database schema.

This script applies the schema.sql to create the required tables and indexes
for the RAG system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.postgres_client import PostgresClient
from src.observability.logging_config import get_logger

log = get_logger(__name__)


def setup_database():
    """
    Apply the database schema from schema.sql.
    
    Raises:
        Exception: If schema application fails
    """
    log.info("Starting database setup")
    
    try:
        # Initialize PostgreSQL client
        postgres_client = PostgresClient()
        
        # Check pgvector extension health
        log.info("Checking pgvector extension health")
        postgres_client.check_extension_health()
        
        # Read the schema file
        schema_path = Path(__file__).parent.parent / "src" / "db" / "schema.sql"
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        log.info("Applying database schema")
        
        # Execute the schema
        conn = postgres_client.get_connection()
        try:
            with conn.cursor() as cur:
                # Split by semicolon to handle multiple statements
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        cur.execute(statement)
                
                conn.commit()
                log.info("Database schema applied successfully")
                
        finally:
            postgres_client.return_connection(conn)
        
        postgres_client.close()
        
        print("✅ Database setup completed successfully!")
        print("   Tables created: document_chunks, eval_runs")
        print("   Indexes created: hnsw on embedding, btree on (collection, chunking_strategy)")
        
    except Exception as e:
        log.error("Database setup failed", extra={"error": str(e)})
        print(f"❌ Database setup failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        setup_database()
    except Exception as e:
        sys.exit(1)
