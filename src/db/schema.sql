-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- document_chunks table for storing text chunks with embeddings
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection TEXT NOT NULL CHECK (collection IN ('sales_psychology', 'mortgage_domain')),
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunking_strategy TEXT NOT NULL CHECK (chunking_strategy IN ('naive', 'semantic', 'hyde')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- eval_runs table for storing evaluation metrics
CREATE TABLE eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    collection TEXT NOT NULL,
    faithfulness DOUBLE PRECISION,
    answer_relevancy DOUBLE PRECISION,
    context_precision DOUBLE PRECISION,
    latency_ms INTEGER,
    cost_usd NUMERIC,
    answered BOOLEAN,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index on document_chunks.embedding using cosine distance
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- B-tree index on document_chunks(collection, chunking_strategy) for filtering
CREATE INDEX ON document_chunks (collection, chunking_strategy);
