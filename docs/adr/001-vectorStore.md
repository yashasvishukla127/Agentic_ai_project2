## Context

This project needs to store document chunk embeddings and retrieve them by vector similarity. The system also stores eval results (RAGAS metrics) per run. I need to decide where embeddings live.  

## Decision
i am choosing pgvector because it would be one database for leads , email logs , followup state and embeddings - no sync issues 
there would be transaction consistency ( deleting a lead -> embeddings will also get deleted)
also i would run pg vector on docker 
## Alternatives Considered
chromaDb - easiest Python setup, no SQL needed. 
Rejected: separate process from Postgres,still requires two different databases , not good when deployed for real work 

Qdrant -  best vector search performance and filtering at scale. 
Rejected: requires a separate Docker container, adds operational complexity not justified at this project scale.

pinecone -  fully managed, zero ops burden. 
Rejected: adds external cost ($), adds another API failure point, separate process from Postgres, so eval results and vectors are in two different databases. Harder to join and compare.

## Tradeoffs
Gains: single database for both vectors and eval results — simpler operations , easy SQL joins between eval runs and retrieved chunks. 
Costs: pgvector is slower than Qdrant at high vector counts (>1M). Postgres must be running for any part of the system to work.
 - will revisit this decision if knowledge_chunks grows more

## At 10x scale
At 10x document volume (>500k chunks) or multi-tenant usage, I'd migrate vectors to a dedicated Qdrant cluster for performance, keep Postgres only for eval metadata and user data. The schema would stay the same — only the vector query path changes.
because at this large scale pg vector would start leading contrain on RAM

╭──────────────────────────────────────────────╮
│ 🏗️ Architecture Decision Record #001       │
├──────────────────────────────────────────────┤
│ Problem                                      │
│ Store embeddings + metadata efficiently      │
├──────────────────────────────────────────────┤
│ ✅ Decision                                  │
│ PostgreSQL + pgvector                        │
├──────────────────────────────────────────────┤
│ Why?                                         │
│ • One database                               │
│ • ACID transactions                          │
│ • No synchronization issues                  │
│ • SQL joins for evaluation                   │
├──────────────────────────────────────────────┤
│ Tradeoffs                                    │
│ ✔ Simpler architecture                       │
│ ✖ Slower than dedicated vector DBs           │
├──────────────────────────────────────────────┤
│ Future                                       │
│ >500k chunks → migrate to Qdrant             │
╰──────────────────────────────────────────────╯