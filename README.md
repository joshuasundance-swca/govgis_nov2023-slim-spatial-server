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

- `PostGIS`: An extension of PostgreSQL, enabling it to store and manipulate spatial data.
- `pgvector`: A PostgreSQL extension for efficient similarity searches in high-dimensional vector spaces.
- `govgis_nov2023`: A rich dataset encapsulating metadata from various government GIS servers as of November 2023.

### Docker Composition

The `docker-compose.yml` file in this project defines multiple services:

1. **postgres**: Utilizes the `joshuasundance/postgis_pgvector:1.0.0` image, incorporating both PostGIS and pgvector.
2. **postgres-init**: A service to initialize the database with data from the `govgis_nov2023` dataset.
3. **pgadmin**: Provides a web interface for database management using `dpage/pgadmin4:7.8`.

## Usage

1. **Setup**: Clone the repository and navigate to the directory containing the `docker-compose.yml` file.
2. **Configuration**: Adjust the `.env` file to set necessary environment variables.
3. **Build and Run**: Execute `docker compose up` to build and start the services.
4. **Access pgAdmin**: Open `http://localhost:80` in a web browser for database management.

## Database Initialization

The `postgres-init` service is responsible for loading data into the database. It processes the `govgis_nov2023` dataset, transforming it into a suitable format for PostgreSQL, and then populates the database.

Download [the dataset](https://huggingface.co/datasets/joshuasundance/govgis_nov2023-slim-spatial/blob/main/govgis_nov2023_slim_spatial_embs.geoparquet) and
place it in `./govgis-nov2023` (according to the `docker-compose.yml` volume mapping).

## Performance & Pooling

To minimize first-request latency and control connection behavior, several environment variables are available:

Core variables (existing):
- `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: Superuser / bootstrap credentials.
- `READ_ONLY_POSTGRES_USER`, `READ_ONLY_POSTGRES_USER_PASSWORD`: A read-only role is auto-created / updated on startup if needed.
- `POSTGRES_SCHEMA` (default `public`), `POSTGRES_TABLE` (default `layers`).

Pooling / warm-up variables (new):
- `PG_POOL_MAX_SIZE` (default `10`): Upper bound of concurrent DB connections in the async pool.
- `PG_POOL_MIN_SIZE` (default `1`): Minimum number of kept-open connections (ignored if eager mode is on).
- `PG_POOL_EAGER` (default `false`): When true, sets `min_size = max_size` and warms all connections immediately.
- `PG_POOL_WARM_TARGET` (optional int): If not eager, pre-initialize up to this many connections (capped by `PG_POOL_MAX_SIZE`).
- `PG_POOL_TIMEOUT` (default `30` seconds): How long a request waits for a free connection before raising an error.
- `EMBEDDING_WARMUP` (default `true`): Pre-computes a single embedding at startup so the model is JIT-loaded before traffic.

Behavior precedence:
1. If `PG_POOL_EAGER=true`, the pool will attempt to fully open and validate `PG_POOL_MAX_SIZE` connections; `PG_POOL_WARM_TARGET` is ignored.
2. If `PG_POOL_EAGER=false` and `PG_POOL_WARM_TARGET` is set, that many connections are warmed (minimum of configured target and `PG_POOL_MAX_SIZE`).
3. If neither eager nor warm target is set, only `PG_POOL_MIN_SIZE` connections are opened initially; additional are created lazily on demand.

Recommended setups:
- Local development (fast startup, ok with a small first-hit penalty):
  - `PG_POOL_EAGER=false`, `PG_POOL_MIN_SIZE=1`, omit warm target.
- Low-latency demo / production small instance (predictable concurrency < 10):
  - `PG_POOL_EAGER=true`, adjust `PG_POOL_MAX_SIZE` to expected peak.
- Burst traffic with moderate DB budget:
  - `PG_POOL_EAGER=false`, `PG_POOL_WARM_TARGET=4` (or similar), `PG_POOL_MAX_SIZE=16` (connections above warm target are created lazily).

Example docker-compose service override:
```
  backend:
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_DB: govgis
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: example
      READ_ONLY_POSTGRES_USER: ro
      READ_ONLY_POSTGRES_USER_PASSWORD: ro_pw
      PG_POOL_MAX_SIZE: "12"
      PG_POOL_MIN_SIZE: "2"
      PG_POOL_WARM_TARGET: "6"   # partial pre-warm
      PG_POOL_TIMEOUT: "20"
      EMBEDDING_WARMUP: "true"
      # Set PG_POOL_EAGER: "true" to fully pre-warm all 12 connections instead
```

Windows CMD one-off run (fully eager):
```
set PG_POOL_EAGER=true
set PG_POOL_MAX_SIZE=10
set PG_POOL_MIN_SIZE=1
set EMBEDDING_WARMUP=true
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Latency measurement tips:
- First request cold latency: `curl -w "First byte: %{time_starttransfer}\n" -o NUL -s http://localhost:8000/docs`
- Load test (example using `hey`): `hey -z 30s -c 10 http://localhost:8000/health` (add a lightweight health endpoint if desired).

## Customization

- **Environment Variables**: Modify the `.env` file to set values like `POSTGRES_PASSWORD`, `POSTGRES_USER`, etc.
- **Data Source**: You can change the source of the `govgis_nov2023` dataset by modifying the `load_data.py` script.

## Notes

- Docker and Docker Compose are prerequisites.
- The project is intended for development and testing purposes.
- Secure your database and pgAdmin for production use.

## Contributing

Contributions to enhance the project are welcome. Please use the standard fork and pull request workflow for contributions.

## License

`govgis_nov2023-slim-spatial-server` is licensed under the MIT License.
