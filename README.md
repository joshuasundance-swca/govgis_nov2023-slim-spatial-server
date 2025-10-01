# govgis_nov2023-slim-spatial-server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
![Known Vulnerabilities](https://snyk.io/test/github/joshuasundance-swca/govgis_nov2023-slim-spatial-server/badge.svg)

# govgis_nov2023-slim-spatial-server

🤖 This README was written by GPT-4. 🤖

# UPDATE: added an mcp server with `fastmcp`.
# UPDATE: moved from python 3.11 to python 3.13
# TODO: remove asyncpg dependency and use psycopg3

# UPDATE: [`./agent.ipynb`](./agent.ipynb) shows the use of `deepagents` with the mcp server

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
