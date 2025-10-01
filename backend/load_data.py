import asyncio
import logging
import os
from typing import Iterable, Tuple

import geopandas as gpd
import shapely.wkb
import psycopg
from psycopg import AsyncConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

table_columns = [
    "id",
    "name",
    "type",
    "description",
    "url",
    "metadata_text",
    "embeddings",
    "geom",
]

BATCH_SIZE = 500


def check_env_vars() -> None:
    """Check that the required environment variables are set"""
    required_env_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "GEOPARQUET_PATH",
        "READ_ONLY_POSTGRES_USER",
        "READ_ONLY_POSTGRES_USER_PASSWORD",
    ]

    optional_env_vars_with_defaults = {
        "POSTGRES_SCHEMA": "public",
        "POSTGRES_TABLE": "layers",
        "SPATIAL_INDEX_NAME": "layer_extent_index",
    }

    for env_var in required_env_vars:
        if not os.environ.get(env_var):
            raise ValueError(f"{env_var} environment variable must be set")

    for env_var, default_value in optional_env_vars_with_defaults.items():
        if not os.environ.get(env_var):
            os.environ[env_var] = default_value
            logging.info(f"Setting default for {env_var}: {default_value}")


def check_geoparquet_path() -> None:
    """Check that the GEOPARQUET_PATH environment variable points to a valid path"""
    geoparquet_path = os.environ.get("GEOPARQUET_PATH")
    if not geoparquet_path or not os.path.exists(geoparquet_path):
        raise FileNotFoundError(
            f"GEOPARQUET_PATH environment variable must be set to a valid path, but {geoparquet_path} does not exist",
        )


make_table_sql = """CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table}
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


async def make_table(conn: AsyncConnection) -> None:
    """Create the table if it doesn't already exist"""
    async with conn.cursor() as cur:
        await cur.execute(make_table_sql.format(table=os.environ["POSTGRES_TABLE"]))


make_index_sql = """CREATE INDEX IF NOT EXISTS {index_name}
    ON {table} USING gist
    (geom)
    TABLESPACE pg_default;"""


async def make_index(conn: AsyncConnection) -> None:
    """Create the spatial index if it doesn't already exist"""
    async with conn.cursor() as cur:
        await cur.execute(
            make_index_sql.format(
                index_name=os.environ["SPATIAL_INDEX_NAME"],
                table=os.environ["POSTGRES_TABLE"],
            ),
        )


make_user_sql = """CREATE ROLE {read_only_user} WITH LOGIN PASSWORD '{read_only_user_password}';
GRANT CONNECT ON DATABASE {database} TO {read_only_user};
GRANT USAGE ON SCHEMA {schema} TO {read_only_user};
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} FROM {read_only_user};
GRANT SELECT ON {table} TO {read_only_user};"""


async def make_user(conn: AsyncConnection) -> None:
    """Create the user if it doesn't already exist"""
    user_kwargs = {
        "read_only_user": os.environ["READ_ONLY_POSTGRES_USER"],
        "read_only_user_password": os.environ["READ_ONLY_POSTGRES_USER_PASSWORD"],
        "database": os.environ["POSTGRES_DB"],
        "schema": os.environ["POSTGRES_SCHEMA"],
        "table": os.environ["POSTGRES_TABLE"],
    }

    for k, v in user_kwargs.items():
        if k == "read_only_user_password":
            logging.info(f"{k} = {'*' * len(v)}")
        else:
            logging.info(f"{k} = {v}")

    async with conn.cursor() as cur:
        try:
            await cur.execute(make_user_sql.format(**user_kwargs))
        except Exception as e:
            # Likely role exists; log and continue
            logging.info(f"User creation skipped or partially applied: {e}")


async def check_table_length(conn: AsyncConnection) -> bool:
    """Check if the table is empty"""
    async with conn.cursor() as cur:
        await cur.execute(
            f"SELECT COUNT(*) FROM {os.environ['POSTGRES_TABLE']}",  # nosec
        )
        row = await cur.fetchone()
    count = row[0] if row else 0
    logging.info(f"Table has {count} rows")
    return count == 0


async def get_connection() -> AsyncConnection:
    """Get an async psycopg connection to the database"""
    conn = await psycopg.AsyncConnection.connect(
        host=os.environ["POSTGRES_HOST"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    return conn


def vector_literal(val) -> str:
    if isinstance(val, str):  # already a literal
        return val
    if isinstance(val, (list, tuple)):
        return "[" + ",".join(format(float(x), "g") for x in val) + "]"
    raise TypeError("Unsupported embedding value type for vector literal")


def record_generator() -> Iterable[Tuple[str, str, str, str, str, str, str, bytes]]:
    """Generate transformed records ready for insertion.

    Embeddings are converted to pgvector literals; geom converted to WKB bytes.
    """
    gdf = (
        gpd.read_parquet(os.environ["GEOPARQUET_PATH"])
        .drop_duplicates(subset=["id", "metadata_text"])
        .rename(columns={"geometry": "geom"})
    )
    for row in gdf.itertuples(index=False):
        # row order matches table_columns
        (id_, name, type_, description, url, metadata_text, embeddings, geom) = row
        emb_lit = vector_literal(embeddings)
        geom_wkb = shapely.wkb.dumps(geom)
        yield (
            id_,
            name,
            type_,
            description,
            url,
            metadata_text,
            emb_lit,
            geom_wkb,
        )


def batched(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


INSERT_SQL = (
    "INSERT INTO {table} (id,name,type,description,url,metadata_text,embeddings,geom) "
    "VALUES (%s,%s,%s,%s,%s,%s,%s::vector,ST_GeomFromWKB(%s,4326))"
)


async def load_data(conn: AsyncConnection) -> None:
    """Load data into the database using batched executemany."""
    sql = INSERT_SQL.format(table=os.environ["POSTGRES_TABLE"])
    async with conn.transaction():
        async with conn.cursor() as cur:
            total = 0
            for chunk in batched(record_generator(), BATCH_SIZE):
                await cur.executemany(sql, chunk)
                total += len(chunk)
                logging.info(f"Inserted {total} records ...")
    logging.info("Data load complete")


async def amain() -> None:
    check_env_vars()
    logging.info("Environment variables are valid")

    conn = await get_connection()
    logging.info("Connected to database")

    logging.info("Checking / creating table")
    await make_table(conn)

    len_zero = await check_table_length(conn)

    if len_zero:
        check_geoparquet_path()
        logging.info("Geoparquet path is valid")

        logging.info("Loading data")
        await load_data(conn)
        logging.info("Data loaded")

        logging.info("Making spatial index")
        await make_index(conn)

        logging.info("Adding read-only user")
        await make_user(conn)

    logging.info("All done")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(amain())
