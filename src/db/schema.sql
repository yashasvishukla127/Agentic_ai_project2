CREATE TABLE IF NOT EXISTS eval_runs (
    run_id UUID PRIMARY KEY,
    query_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    collection TEXT NOT NULL,
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    latency_ms INTEGER,
    cost_usd FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_query_id ON eval_runs(query_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_strategy ON eval_runs(strategy);
CREATE INDEX IF NOT EXISTS idx_eval_runs_collection ON eval_runs(collection);
CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs(timestamp);
