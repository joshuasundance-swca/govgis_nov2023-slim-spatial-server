import asyncio
import logging
import os
from typing import Iterable
from asyncpg.connection import Connection
import asyncpg
import geopandas as gpd
import shapely.wkb
from pgvector.asyncpg import register_vector

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

    # Check required environment variables
    for env_var in required_env_vars:
        if not os.environ.get(env_var):
            raise ValueError(f"{env_var} environment variable must be set")

    # Check optional environment variables and set defaults if not present
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


async def make_table(conn: Connection) -> None:
    """Create the table if it doesn't already exist"""
    await conn.execute(make_table_sql.format(table=os.environ["POSTGRES_TABLE"]))


make_index_sql = """CREATE INDEX IF NOT EXISTS {index_name}
    ON layers USING gist
    (geom)
    TABLESPACE pg_default;"""


async def make_index(conn: Connection) -> None:
    """Create the spatial index if it doesn't already exist"""
    await conn.execute(
        make_index_sql.format(index_name=os.environ["SPATIAL_INDEX_NAME"]),
    )


make_user_sql = """CREATE ROLE {read_only_user} WITH LOGIN PASSWORD '{read_only_user_password}';
GRANT CONNECT ON DATABASE {database} TO {read_only_user};
GRANT USAGE ON SCHEMA {schema} TO {read_only_user};
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} FROM {read_only_user};
GRANT SELECT ON {table} TO {read_only_user};"""


async def make_user(
    conn: Connection,
) -> None:
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

    await conn.execute(make_user_sql.format(**user_kwargs))


async def check_table_length(conn: Connection) -> bool:
    """Check if the table is empty"""
    count = await conn.fetchval(
        f"SELECT COUNT(*) FROM {os.environ['POSTGRES_TABLE']}",  # nosec
    )
    logging.info(f"Table has {count} rows")
    return count == 0


async def get_connection() -> Connection:
    """Get a connection to the database"""
    conn = await asyncpg.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        database=os.environ["POSTGRES_DB"],
    )

    # prepare connection for vectors
    await register_vector(conn)

    # prepare connection for geometries
    def encode_geometry(geometry):
        if not hasattr(geometry, "__geo_interface__"):
            raise TypeError(
                "{g} does not conform to " "the geo interface".format(g=geometry),
            )
        shape = shapely.geometry.shape(geometry)
        return shapely.wkb.dumps(shape)

    def decode_geometry(wkb):
        return shapely.wkb.loads(wkb)

    await conn.set_type_codec(
        "geometry",
        encoder=encode_geometry,
        decoder=decode_geometry,
        format="binary",
    )

    return conn


def data_generator() -> Iterable[tuple[str, str, str, str, str, str, str, str]]:
    """Generate data to be inserted into the database"""
    return (
        gpd.read_parquet(os.environ["GEOPARQUET_PATH"])
        .drop_duplicates(subset=["id", "metadata_text"])
        .rename(columns={"geometry": "geom"})[table_columns]
        .itertuples(index=False)
    )


async def load_data(conn: Connection) -> None:
    """Load data into the database"""
    async with conn.transaction():
        await conn.copy_records_to_table(
            "layers",
            records=data_generator(),
            columns=table_columns,
        )


async def amain() -> None:
    check_env_vars()
    logging.info("Environment variables are valid")

    conn = await get_connection()
    logging.info("Connected to database")

    logging.info("Checking table")
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
