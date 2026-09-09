-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- document_chunks table for storing text chunks with embeddings
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
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

-- ingestion_runs table for tracking ingestion job status
CREATE TABLE ingestion_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL CHECK (chunking_strategy IN ('naive', 'semantic', 'hyde')),
    total_chunks INTEGER NOT NULL,
    chunks_completed INTEGER NOT NULL DEFAULT 0,
    chunks_skipped INTEGER NOT NULL DEFAULT 0,
    chunks_added INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- B-tree index on ingestion_runs(source_file, chunking_strategy) for run lookup
CREATE INDEX ON ingestion_runs (source_file, chunking_strategy);

-- HNSW index on document_chunks.embedding using cosine distance
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- B-tree index on document_chunks(collection, chunking_strategy) for filtering
CREATE INDEX ON document_chunks (collection, chunking_strategy);
