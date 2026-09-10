.\.venv\Scripts\Activate.ps1
psql -U postgres -d p2_rag -h localhost -p 5432

python -m pytest tests/test_ingestion_idempotency.py::test_ingestion_crash_recovery -v -s

(to run tests)

# Run this specific file with the -v (verbose) flag for detailed output
pytest tests/failure_modes/test_db_unreachable.py -v