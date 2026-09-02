several workers could try to write to costs.csv concurrently.

That can cause concurrent write contention or potentially malformed/interleaved writes, depending on how the application is deployed and how the OS/processes interact.

So this implementation is fine for a simple/single-process setup, but for a high-concurrency application, we would generally want safer sotrage mechanism such as:

SQLite
PostgreSQL
a dedicated logging/metrics system

or explicit file locking if CSV storage is a requirement.