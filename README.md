# govgis_nov2023-slim-spatial-server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
![Known Vulnerabilities](https://snyk.io/test/github/joshuasundance-swca/govgis_nov2023-slim-spatial-server/badge.svg)

# govgis_nov2023-slim-spatial-server

🤖 This README was written by GPT-4. 🤖

## UPDATES (10/01/2025)
- Migrated asyncpg & psycopg2 to psycopg3 (async) using AsyncConnectionPool
- Implemented pooled connections for FastAPI lifespan
- Rewrote data loader to use batched psycopg async executemany with pgvector + PostGIS
- Removed custom asyncpg codecs and copy logic; simplified vector + geometry insertion
- Added configurable eager pool warm-up + partial warm targeting and embedding model warm-up (see "Performance & Pooling")
- Added vector similarity index creation (ivfflat or hnsw) + tuning env vars
- Added high-volume (~1M rows) bulk load tuning (commit interval + session tweaks)
- Introduced COPY-based high-performance ingestion (default) via `LOADER_METHOD=copy` with deferred index creation until after all rows are loaded

## UPDATES (09/30/2025)
- moved from python 3.11 to python 3.13
- added an mcp server with `fastmcp`
- `agent.ipynb` shows the use of `deepagents` with the mcp server
- (Completed) remove asyncpg dependency and use psycopg3

## Introduction

`govgis_nov2023-slim-spatial-server` is a Dockerized project combining `PostGIS` and `pgvector` extensions to process and serve a comprehensive geospatial dataset, `govgis_nov2023`. This project aims to provide a robust framework for efficient handling and querying of geospatial data along with high-dimensional vector similarity search capabilities, leveraging the power of PostgreSQL.

- https://huggingface.co/datasets/joshuasundance/govgis_nov2023
- https://huggingface.co/datasets/joshuasundance/govgis_nov2023-slim-spatial

## Components

- `PostGIS`: Spatial indexing & geometry functions.
- `pgvector`: Approximate / exact vector similarity search (supports ivfflat & hnsw indexes here).
- `govgis_nov2023`: Metadata + embeddings derived from public GIS endpoints (Nov 2023 snapshot).

### Docker Composition

The `docker-compose.yml` file in this project defines multiple services:

1. **postgres**: Utilizes the `joshuasundance/postgis_pgvector:1.0.0` image, incorporating both PostGIS and pgvector.
2. **postgres-init**: A service to initialize and load nearly a million rows from the geoparquet file.
3. **pgadmin**: Provides a web interface for database management using `dpage/pgadmin4:7.8`.

## Usage

1. **Setup**: Clone the repository and navigate to the directory containing the `docker-compose.yml` file.
2. **Configuration**: Copy `.env-example` to `.env` and adjust paths & secrets.
3. **Download Dataset**: Place the `govgis_nov2023_slim_spatial_embs.geoparquet` file into `./govgis-nov2023`.
4. **Build and Run**: `docker compose up --build` (first run will ingest data, taking several minutes depending on hardware / IO & index strategy).
5. **Access pgAdmin**: Open `http://localhost:80`.

## Database Initialization

The `postgres-init` service loads (up to) ~1,000,000 rows, transforming embeddings + polygon geometries, enforcing dimensionality, and then (after the data load completes) creates:
- A spatial GiST index on `geom`
- A vector similarity index (configurable: ivfflat or hnsw) on `embeddings`
- A read-only role with `SELECT` privileges for application access.

Index creation is intentionally deferred until all rows are inserted to maximize ingestion throughput and avoid incremental index maintenance overhead.

## Vector Similarity Indexing

Environment variables control how the embeddings index is built:
- `VECTOR_INDEX_TYPE`: `ivfflat` (default) or `hnsw`.
- `VECTOR_METRIC`: `l2`, `cosine`, or `ip` (inner product). Mapped to the proper operator class automatically.
- `VECTOR_INDEX_NAME`: Name of the index (default `layer_embedding_index`).
- `PGVECTOR_DIM`: Embedding dimension (must match dataset; enforced per row).

Type-specific parameters:
- ivfflat: `VECTOR_IVFFLAT_LISTS` (default 100). Rough tuning heuristic: choose lists near √N or N / 1000. For ~1M rows consider 1000–2000 for improved recall. Larger `lists` increases build-time RAM use.
- hnsw: `VECTOR_HNSW_M` (graph degree, default 16) and `VECTOR_HNSW_EF_CONSTRUCTION` (construction search breadth, default 64). For higher recall at cost of build time, consider M=32, EF=200.

Memory autotuning (ivfflat only):
- `VECTOR_MAINTENANCE_WORK_MEM_MB` (default 512): Session value requested for `maintenance_work_mem` before build.
- `VECTOR_MAINTENANCE_WORK_MEM_MAX_MB` (default 2048): Upper bound the autotuner may raise to if the server reports a higher requirement.
- `VECTOR_AUTOTUNE_INDEX_MEMORY` (default true): When true, if the server error contains a line like `memory required is 525 MB, maintenance_work_mem is 64 MB`, the loader will attempt (once) to raise `maintenance_work_mem` (up to the max) and retry. If still insufficient, it will reduce `VECTOR_IVFFLAT_LISTS` by 25% iteratively (not dropping below 50) before giving up.

If you see a memory error:
1. Increase `VECTOR_MAINTENANCE_WORK_MEM_MB` (and possibly the MAX) OR reduce `VECTOR_IVFFLAT_LISTS`.
2. Ensure the underlying Postgres container allows that value (the session `SET` can fail if global limits are lower).
3. Re-run the init container (data reload is skipped; only index build runs). If the index already partially exists, you may need to `DROP INDEX layer_embedding_index;` first.

Example fast recall (higher memory):
```
VECTOR_IVFFLAT_LISTS=1800
VECTOR_MAINTENANCE_WORK_MEM_MB=1024
VECTOR_MAINTENANCE_WORK_MEM_MAX_MB=2048
```
Memory conservation build:
```
VECTOR_IVFFLAT_LISTS=400
VECTOR_MAINTENANCE_WORK_MEM_MB=256
```

Example query (L2 distance):
```
SELECT id, name, embeddings <-> '[0.12,0.34,...]' AS dist
FROM layers
ORDER BY embeddings <-> '[0.12,0.34,...]'
LIMIT 10;
```
Cosine (requires `VECTOR_METRIC=cosine` at index build):
```
SELECT id, name, embeddings <=> '[0.12,0.34,...]' AS cosine_distance
FROM layers
ORDER BY embeddings <=> '[0.12,0.34,...]'
LIMIT 10;
```

## High-Volume Load Tuning (~1M Rows)

The loader provides knobs to balance speed, WAL size, and crash safety.

Loader method:
- `LOADER_METHOD=copy` (default): Uses a temporary staging table + PostgreSQL COPY for fastest ingestion, then bulk INSERT-select transform into the target table.
- `LOADER_METHOD=copy_direct`: Pre-vectorizes all data in Python (WKB + embeddings) then uses COPY with staging. Best for trusted data with `ENFORCE_EMBED_DIM=false`.
- `LOADER_METHOD=executemany`: Falls back to batched parameter inserts (slower, but simpler; useful for debugging or if COPY issues arise).

Core variables:
- `INIT_LOADER_BATCH_SIZE` (executemany only; default 500): Rows per batch when not using COPY.
- `LOADER_COMMIT_INTERVAL` (executemany only; default 100000): Rows between commits. Ignored in COPY mode (COPY path commits once at end).
- `LOADER_PERFORMANCE_TWEAKS` (default true): Applies session changes:
  - `synchronous_commit` (default off for speed – set `LOADER_SYNCHRONOUS_COMMIT=on` for durability)
  - `work_mem`
  - `temp_buffers`

Performance optimization flags:
- `ENFORCE_EMBED_DIM` (default true): When false, skips per-row dimension validation for ~10x faster ingestion. Only safe if data is pre-validated.
- `FAST_DEV_SKIP_INDEX` (default false): When true, skips vector index creation during data load. Useful for iterative development cycles.

Additional tuning:
- For COPY, `LOADER_COMMIT_INTERVAL` is intentionally ignored since COPY already streams efficiently and one final commit is fastest.
- Ensure dataset file resides on fast local storage (container volume on SSD preferable).

Recommended scenarios:
- Fast initial load: `LOADER_METHOD=copy_direct`, `ENFORCE_EMBED_DIM=false`, tweaks on (defaults suffice).
- Production load: `LOADER_METHOD=copy` (default), `ENFORCE_EMBED_DIM=true`.
- Diagnostic / controlled: `LOADER_METHOD=executemany`, `LOADER_COMMIT_INTERVAL=50000`, synchronous commit on.

Progress Logs:
- COPY: Logs every 100k streamed rows, plus detailed timing for read/vectorize/copy/transform phases.
- executemany: Logs every 50 batches and at each configured commit boundary.

## Performance & Pooling

(Existing section retained, now complemented by loader tuning.)

To minimize first-request latency and control connection behavior, several environment variables are available:

Core variables (existing):
- `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: Superuser / bootstrap credentials.
- `READ_ONLY_POSTGRES_USER`, `READ_ONLY_POSTGRES_USER_PASSWORD`: A read-only role is auto-created / updated on startup if needed.
- `POSTGRES_SCHEMA` (default `public`), `POSTGRES_TABLE` (default `layers`).

Pooling / warm-up variables:
- `PG_POOL_MAX_SIZE`, `PG_POOL_MIN_SIZE`, `PG_POOL_EAGER`, `PG_POOL_WARM_TARGET`, `PG_POOL_TIMEOUT`, `EMBEDDING_WARMUP`.

Query performance tuning:
- `IVFFLAT_PROBES` (optional): Set at query time to control recall vs speed tradeoff. Higher values = better recall but slower queries. Default is `lists/10`. For a 1200-list index, try `IVFFLAT_PROBES=200` for better recall.
- `EMBED_CACHE_SIZE` (default 256): LRU cache size for embedding generation. Reduces latency on repeated queries. Set to 0 to disable.

Native pgvector adapter:
- When `pgvector` Python package is installed, the connection pool automatically registers the native adapter for efficient `list[float]` → vector conversion.
- Falls back to string literal format if adapter is unavailable.
- Removes `::vector` cast overhead and improves query planning.

## Environment Variable Reference (Summary)

Data & table:
- `GEOPARQUET_PATH`, `POSTGRES_SCHEMA`, `POSTGRES_TABLE`, `PGVECTOR_DIM`

Vector indexing:
- `VECTOR_INDEX_TYPE`, `VECTOR_METRIC`, `VECTOR_INDEX_NAME`, `VECTOR_IVFFLAT_LISTS`, `VECTOR_HNSW_M`, `VECTOR_HNSW_EF_CONSTRUCTION`, `VECTOR_MAINTENANCE_WORK_MEM_MB`, `VECTOR_MAINTENANCE_WORK_MEM_MAX_MB`, `VECTOR_AUTOTUNE_INDEX_MEMORY`

Loader performance & method:
- `LOADER_METHOD`, `INIT_LOADER_BATCH_SIZE` (executemany only), `LOADER_COMMIT_INTERVAL` (executemany only), `LOADER_PERFORMANCE_TWEAKS`, `LOADER_WORK_MEM`, `LOADER_TEMP_BUFFERS`, `LOADER_SYNCHRONOUS_COMMIT`, `ENFORCE_EMBED_DIM`, `FAST_DEV_SKIP_INDEX`

Query performance:
- `IVFFLAT_PROBES`, `EMBED_CACHE_SIZE`

Spatial indexing:
- `SPATIAL_INDEX_NAME`

Security / roles:
- `READ_ONLY_POSTGRES_USER`, `READ_ONLY_POSTGRES_USER_PASSWORD`

Connection pooling:
- `PG_POOL_MAX_SIZE`, `PG_POOL_MIN_SIZE`, `PG_POOL_EAGER`, `PG_POOL_WARM_TARGET`, `PG_POOL_TIMEOUT`

Misc:
- `LOG_LEVEL`, `EMBEDDING_WARMUP`

## Quick Start Example (.env Snippet)
```
GEOPARQUET_PATH=/data/govgis_nov2023_slim_spatial_embs.geoparquet
POSTGRES_HOST=postgres
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
READ_ONLY_POSTGRES_USER=appuser
READ_ONLY_POSTGRES_USER_PASSWORD=secret
PGVECTOR_DIM=1024
VECTOR_INDEX_TYPE=ivfflat
VECTOR_METRIC=cosine
VECTOR_IVFFLAT_LISTS=1200
LOADER_METHOD=copy
LOADER_PERFORMANCE_TWEAKS=true
SPATIAL_INDEX_NAME=layer_extent_index
```

## Customization

- **Environment Variables**: Modify the `.env` file to set values like `POSTGRES_PASSWORD`, `POSTGRES_USER`, etc.
- **Data Source**: Swap in another geoparquet with same column schema (id,name,type,description,url,metadata_text,embeddings,geometry) + matching embedding dimension.

## Notes

- First full ingestion + index build for ~1M rows can take several minutes; repeat runs skip ingestion if table already populated.
- If switching metrics (e.g. l2 -> cosine) you must drop the vector index before re-running to rebuild with the new operator class.
- For production, review durability trade-offs if using `synchronous_commit=off` during load (executemany or COPY session tweaks).

## Contributing

Contributions welcome. Please use the standard fork & PR workflow and keep changes focused.

## License

`govgis_nov2023-slim-spatial-server` is licensed under the MIT License.
