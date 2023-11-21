import logging
import os
from typing import Iterable

import geopandas as gpd
import psycopg2
from psycopg2.extras import execute_values

from utils import transform_emb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

make_table_sql = """CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS layers
(
    id text,
    name text,
    type text,
    description text,
    url text,
    metadata_text text,
    embeddings vector(1024),
    geom geometry(Polygon,4326)
)
TABLESPACE pg_default;"""


def make_table(conn) -> None:
    """Create the table if it doesn't already exist"""
    with conn.cursor() as cur:
        cur.execute(make_table_sql)
        conn.commit()
        cur.close()


make_index_sql = """CREATE INDEX IF NOT EXISTS layer_extent_index
    ON layers USING gist
    (geom)
    TABLESPACE pg_default;"""


def make_index(conn) -> None:
    """Create the spatial index if it doesn't already exist"""
    with conn.cursor() as cur:
        cur.execute(make_index_sql)
        conn.commit()
        cur.close()


invalid_env_values = {"", None}


def check_env_vars():
    """Check that the required environment variables are set"""
    for env_var in (
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_DB",
    ):
        if os.environ.get(env_var, None) in invalid_env_values:
            raise ValueError(f"{env_var} environment variable must be set")


def check_geoparquet_path() -> str:
    """Check that the GEOPARQUET_PATH environment variable is set and points to a valid path"""
    geoparquet_path = os.environ.get("GEOPARQUET_PATH", "")
    if geoparquet_path == "":
        raise ValueError("GEOPARQUET_PATH environment variable must be set")

    if not os.path.exists(geoparquet_path):
        raise FileNotFoundError(
            f"GEOPARQUET_PATH environment variable must be set to a valid path, but {geoparquet_path} does not exist",
        )

    return geoparquet_path


def table_is_empty(conn, table_name: str = "layers") -> bool:
    """Check if the table is empty"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")  # nosec
        count = cur.fetchone()[0]
    logging.info(f"Table has {count} rows")
    return count == 0


def get_connection():
    """Get a connection to the database"""
    return psycopg2.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        database=os.environ["POSTGRES_DB"],
    )


def data_generator(
    geoparquet_path: str,
) -> Iterable[tuple[str, str, str, str, str, str, str, str]]:
    """Generate data to be inserted into the database"""
    return (
        gpd.read_parquet(geoparquet_path)
        .drop_duplicates(subset=["id", "metadata_text"])
        .assign(
            embeddings=lambda df: df.embeddings.apply(transform_emb),
            geom=lambda df: df.geometry.apply(lambda g: g.wkt),
        )[
            [
                "id",
                "name",
                "type",
                "description",
                "url",
                "metadata_text",
                "embeddings",
                "geom",
            ]
        ]
        .itertuples(index=False)
    )


load_data_sql = "INSERT INTO layers (id, name, type, description, url, metadata_text, embeddings, geom) VALUES %s"


def load_data(conn, geoparquet_path: str) -> None:
    """Load data into the database"""
    with conn.cursor() as cur:
        execute_values(cur, load_data_sql, data_generator(geoparquet_path))
        conn.commit()
        cur.close()


if __name__ == "__main__":
    check_env_vars()
    logging.info("Environment variables are valid")

    conn = get_connection()
    logging.info("Connected to database")

    logging.info("Checking table")
    make_table(conn)

    if table_is_empty(conn):
        logging.info("Table is empty")

        geoparquet_path = check_geoparquet_path()
        logging.info("Geoparquet path is valid")

        logging.info("Loading data")
        load_data(conn, geoparquet_path)
        logging.info("Data loaded")

        logging.info("Making spatial index")
        make_index(conn)

    logging.info("All done")
    conn.close()
