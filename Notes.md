.\.venv\Scripts\Activate.ps1
psql -U postgres -d p2_rag -h localhost -p 5432

python -m pytest tests/test_ingestion_idempotency.py::test_ingestion_crash_recovery -v -s

(to run tests)