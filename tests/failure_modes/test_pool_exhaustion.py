import pytest
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg_pool import PoolTimeout

from src.db.postgres_client import PostgresClient


DEFAULT_POOL_SIZE = 10
EXTRA_CONNECTIONS = 5
TOTAL_CONNECTIONS = DEFAULT_POOL_SIZE + EXTRA_CONNECTIONS
TIMEOUT_THRESHOLD = 4.0  # Allow slightly more than 3 seconds for individual connection timeout


@pytest.fixture
def postgres_client():
    """
    Create a PostgreSQL client with default pool size for testing.
    
    Uses environment variables for connection details.
    Skips test if required environment variables are not set.
    """
    required_vars = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        pytest.skip(
            f"Skipping test: missing required environment variables: {', '.join(missing_vars)}. "
            "Set these in your .env file or environment to run this test."
        )
    
    try:
        client = PostgresClient(max_pool_size=DEFAULT_POOL_SIZE)
    except ValueError as e:
        pytest.skip(f"Skipping test: {str(e)}")
    except Exception as e:
        pytest.skip(f"Skipping test: Failed to initialize PostgreSQL client: {str(e)}")
    
    yield client
    client.close()


def test_pool_exhaustion_timeout(postgres_client):
    """
    Test that connection pool exhaustion fails fast with timeout error.
    """
    import threading

    connection_results = []
    connection_times = []
    pool_filled_event = threading.Event()

    HOLD_TIME = 5.0  # must be longer than the pool's 3s timeout

    def hold_connection(conn_id):
        """Acquire a connection and hold it for HOLD_TIME seconds before releasing."""
        start_time = time.time()
        try:
            conn = postgres_client.get_connection()
            elapsed = time.time() - start_time
            connection_times.append((conn_id, elapsed, "success"))

            # Signal once all "holder" threads presumably have their connections
            time.sleep(HOLD_TIME)

            postgres_client.return_connection(conn)
            return (conn_id, "success", elapsed)
        except PoolTimeout:
            elapsed = time.time() - start_time
            connection_times.append((conn_id, elapsed, "timeout"))
            return (conn_id, "timeout", elapsed)
        except Exception as e:
            elapsed = time.time() - start_time
            connection_times.append((conn_id, elapsed, f"error: {type(e).__name__}"))
            return (conn_id, f"error: {type(e).__name__}", elapsed)

    def acquire_and_release_fast(conn_id):
        """Extra threads: try to acquire, expect timeout since pool is held."""
        start_time = time.time()
        try:
            conn = postgres_client.get_connection()
            elapsed = time.time() - start_time
            connection_times.append((conn_id, elapsed, "success"))
            postgres_client.return_connection(conn)
            return (conn_id, "success", elapsed)
        except PoolTimeout:
            elapsed = time.time() - start_time
            connection_times.append((conn_id, elapsed, "timeout"))
            return (conn_id, "timeout", elapsed)
        except Exception as e:
            elapsed = time.time() - start_time
            connection_times.append((conn_id, elapsed, f"error: {type(e).__name__}"))
            return (conn_id, f"error: {type(e).__name__}", elapsed)

    test_start_time = time.time()

    with ThreadPoolExecutor(max_workers=TOTAL_CONNECTIONS) as executor:
        # Start the 10 "holder" threads first — they'll occupy the whole pool
        holder_futures = {
            executor.submit(hold_connection, i): i
            for i in range(DEFAULT_POOL_SIZE)
        }

        # Give holders a brief head start to actually grab their connections
        time.sleep(0.5)

        # Now fire the extra threads — pool should be fully occupied
        extra_futures = {
            executor.submit(acquire_and_release_fast, i): i
            for i in range(DEFAULT_POOL_SIZE, TOTAL_CONNECTIONS)
        }

        all_futures = {**holder_futures, **extra_futures}
        for future in as_completed(all_futures):
            connection_results.append(future.result())

    total_test_time = time.time() - test_start_time

    successful_connections = [r for r in connection_results if r[1] == "success"]
    timeout_connections = [r for r in connection_results if r[1] == "timeout"]
    error_connections = [r for r in connection_results if r[1].startswith("error")]

    assert len(successful_connections) == DEFAULT_POOL_SIZE, (
        f"Expected {DEFAULT_POOL_SIZE} successful connections, got {len(successful_connections)}"
    )
    assert len(timeout_connections) == EXTRA_CONNECTIONS, (
        f"Expected {EXTRA_CONNECTIONS} timeout connections, got {len(timeout_connections)}"
    )
    assert len(error_connections) == 0, f"Expected no errors, got: {error_connections}"

    for conn_id, status, elapsed in timeout_connections:
        assert 2.0 <= elapsed <= TIMEOUT_THRESHOLD, (
            f"Connection {conn_id} timed out after {elapsed:.2f}s, expected 2.0–{TIMEOUT_THRESHOLD}s"
        )

    print(f"\nTotal test time: {total_test_time:.3f}s")
    for conn_id, elapsed, status in sorted(connection_times):
        print(f"  Connection {conn_id}: {elapsed:.3f}s - {status}")