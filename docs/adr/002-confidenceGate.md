





┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│       Similarity Score Distribution          │    │               Production RAG                 │
├──────────────────────────────────────────────┤    ├──────────────────────────────────────────────┤
│                                              │    │                                              │
│ Similarity Score →                           │    │                 User Query                   │
│ 0.4      0.6      0.8      1.0               │    │                     │                        │
│                                              │    │                     ▼                        │
│ Sales Collection                             │    │             Embedding Search                 │
│      ▁▂▃▅███████▆▃                           │    │                     │                        │
│           ▲                                  │    │                     ▼                        │
│      Threshold = 0.72                        │    │             Similarity Score                 │
│                                              │    │                     │                        │
│ Mortgage Collection                          │    │         ┌───────────┴───────────┐            │
│                     ▁▂▃▅███████▆▃            │    │         │                       │            │
│                          ▲                   │    │  Score ≥ Threshold     Score < Threshold    │
│                     Threshold = 0.84         │    │         │                       │            │
│                                              │    │         ▼                       ▼            │
│ Same embedding model.                        │    │  Generate Answer    "Insufficient            │
│ Different score distributions.               │    │                     Information"             │
│                                              │    │                                              │
│ → One threshold doesn't fit every domain.    │    │ Engineering Principle                       │
│                                              │    │                                              │
│                                              │    │ Better to admit uncertainty                 │
│                                              │    │ than answer confidently                     │
│                                              │    │ with weak evidence.                         │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘