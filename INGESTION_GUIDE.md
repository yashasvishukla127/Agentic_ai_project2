# Ingestion Pipeline Setup Guide

## Prerequisites

1. **PostgreSQL with pgvector extension**
   - Ensure PostgreSQL is running with pgvector extension installed
   - Run `CREATE EXTENSION vector;` in your database if not already done

2. **OpenAI API Key**
   - Add your OpenAI API key to `.env` file:
     ```
     OPENAI_API_KEY=sk-your-actual-api-key-here
     ```

3. **Environment Variables**
   - Ensure `.env` file contains all required variables:
     ```
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432
     POSTGRES_DB=p2_rag
     POSTGRES_USER=p2_user
     POSTGRES_PASSWORD=root
     OPENAI_API_KEY=your_openai_api_key_here
     OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Optional, defaults to text-embedding-3-small
     ```

4. ** Need to setup vector in postgres


## Setup Steps

### 1. Initialize Database Schema

```bash
python scripts/setup_database.py
```

This will create:
- `document_chunks` table with HNSW index on embeddings
- `eval_runs` table for evaluation metrics
- Required pgvector extension check

### 2. Run Ingestion Pipeline

#### Using Naive Chunking (512-character fixed chunks):
```bash
python -m src.ingest.run_ingestion --strategy naive
```

#### Using Semantic Chunking (paragraph-boundary aware):
```bash
python -m src.ingest.run_ingestion --strategy semantic
```

#### Using Different Embedding Models:
```bash
# Use text-embedding-3-large (3072 dimensions)
python -m src.ingest.run_ingestion --strategy naive --model text-embedding-3-large

# Use text-embedding-ada-002 (1536 dimensions)
python -m src.ingest.run_ingestion --strategy semantic --model text-embedding-ada-002

# Set via environment variable
export OPENAI_EMBEDDING_MODEL=text-embedding-3-large
python -m src.ingest.run_ingestion --strategy naive
```

## What Happens During Ingestion

1. **Document Loading**: Reads `data/sales_psychology.md` and `data/mortgage_domain.md`
2. **Chunking**: Splits documents using the specified strategy
3. **Embedding**: Calls OpenAI embedding model for each chunk
4. **Validation**: Ensures embeddings match expected dimensions for the model
5. **Storage**: Inserts chunks with embeddings into `document_chunks` table
6. **Cost Tracking**: Logs total API cost and token usage

## Supported OpenAI Embedding Models

| Model | Dimensions | Cost per 1K tokens | Use Case |
|-------|-----------|-------------------|----------|
| `text-embedding-3-small` | 1536 | $0.00002 | Default, good balance of cost/performance |
| `text-embedding-3-large` | 3072 | $0.00013 | Higher quality, more expensive |
| `text-embedding-ada-002` | 1536 | $0.00010 | Legacy model, generally superseded by v3 |

**Note**: If you use `text-embedding-3-large` (3072 dimensions), you'll need to update your database schema to use `vector(3072)` instead of `vector(1536)`.

## Troubleshooting

### No embeddings appearing in database

1. **Check OpenAI API Key**: Ensure `OPENAI_API_KEY` is set correctly in `.env`
2. **Check Database Connection**: Verify PostgreSQL is running and credentials are correct
3. **Check Schema**: Run `python scripts/setup_database.py` to ensure tables exist
4. **Check Logs**: Review log output for any error messages

### pgvector extension errors

```sql
-- Connect to your database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Connection pool timeout

If you see pool timeout errors, the pool may be exhausted. Check:
- Pool size configuration in `postgres_client.py`
- Number of concurrent operations
- Database connection limits

## Verification

Check if embeddings were created successfully:

```sql
-- Check total chunks
SELECT COUNT(*) FROM document_chunks;

-- Check by collection
SELECT collection, COUNT(*) FROM document_chunks GROUP BY collection;

-- Check by chunking strategy
SELECT chunking_strategy, COUNT(*) FROM document_chunks GROUP BY chunking_strategy;

-- Sample data
SELECT id, collection, source_file, chunk_index, chunking_strategy, 
       LENGTH(content) as content_length, 
       array_length(embedding, 1) as embedding_dim
FROM document_chunks 
LIMIT 5;
```

## Cost Estimates

- OpenAI `text-embedding-3-small`: ~$0.00002 / 1K tokens
- Typical document: ~5-10K tokens
- Estimated cost per document: $0.0001 - $0.0002
- Total for both documents: ~$0.0002 - $0.0004

## Next Steps

After ingestion, you can:
1. Query the embeddings using the retrieval system
2. Run evaluation queries to test retrieval quality
3. Compare different chunking strategies
