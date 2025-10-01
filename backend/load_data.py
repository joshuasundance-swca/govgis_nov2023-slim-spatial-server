import array
import asyncio
import logging
import os
import re
from collections.abc import Iterable as _Iterable
from typing import Iterable, Tuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg
import shapely.wkb
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


def safe_identifier(name: str) -> str:
    """Validate a SQL identifier (very conservative) to mitigate injection risk.

    Allows only letters, digits, and underscore, and cannot start with a digit.
    """
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


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
        "PGVECTOR_DIM": "1024",
    }

    for env_var in required_env_vars:
        if not os.environ.get(env_var):
            raise ValueError(f"{env_var} environment variable must be set")

    for env_var, default_value in optional_env_vars_with_defaults.items():
        if not os.environ.get(env_var):
            os.environ[env_var] = default_value
            logging.info(f"Setting default for {env_var}: {default_value}")
    # Additional validation / sanitization
    os.environ["POSTGRES_TABLE"] = safe_identifier(os.environ["POSTGRES_TABLE"])
    os.environ["SPATIAL_INDEX_NAME"] = safe_identifier(os.environ["SPATIAL_INDEX_NAME"])
    os.environ["POSTGRES_SCHEMA"] = safe_identifier(os.environ["POSTGRES_SCHEMA"])
    os.environ["READ_ONLY_POSTGRES_USER"] = safe_identifier(
        os.environ["READ_ONLY_POSTGRES_USER"],
    )
    dim = os.environ.get("PGVECTOR_DIM", "1024")
    if not dim.isdigit():  # pragma: no cover (defensive)
        raise ValueError("PGVECTOR_DIM must be an integer string")
    os.environ["PGVECTOR_DIM"] = dim


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
    embeddings vector({dim}),
    geom geometry(Polygon,4326)
)
TABLESPACE pg_default;"""


async def make_table(conn: AsyncConnection) -> None:
    """Create the table if it doesn't already exist"""
    async with conn.cursor() as cur:
        await cur.execute(
            make_table_sql.format(
                table=os.environ["POSTGRES_TABLE"],
                dim=os.environ.get("PGVECTOR_DIM", "1024"),
            ),
        )


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
        "read_only_user_password": os.environ[
            "READ_ONLY_POSTGRES_USER_PASSWORD"
        ].replace(
            "'",
            "''",
        ),  # simple escape
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


def vector_literal(val) -> Optional[str]:
    """Return a pgvector literal string for the given embedding value.

    Supports:
    - already formatted string literals like "[0.1,0.2,...]"
    - list/tuple (including numpy scalar types)
    - numpy.ndarray
    - pandas Series
    - bytes / bytearray / memoryview containing 32-bit floats (length divisible by 4)
    - dicts with key 'embedding' or 'values' containing a numeric iterable
    - generic iterables of numeric values (array.array, map, etc.)
    Returns None unchanged (will insert NULL::vector).
    """
    if val is None:
        return None
    if isinstance(val, dict):  # unwrap common containers
        for key in ("embedding", "values", "data"):
            if key in val:
                val = val[key]
                break
    if isinstance(val, str):
        lit = val.strip()
        if lit.startswith("[") and lit.endswith("]"):
            return lit
        if "," in lit:
            parts = [p.strip() for p in lit.split(",")]
            try:
                _ = [float(p) for p in parts]
                return "[" + ",".join(parts) + "]"
            except ValueError:
                raise TypeError(
                    f"String embeddings contain non-numeric values: {lit[:80]}",
                )
        raise TypeError(f"String embeddings not in expected numeric format: {lit[:80]}")
    if isinstance(val, (bytes, bytearray, memoryview)):
        b = bytes(val)
        if len(b) % 4 != 0:
            raise TypeError(
                f"Byte embeddings length {len(b)} not divisible by 4 for float32 decoding",
            )
        arr = array.array("f")
        arr.frombytes(b)
        if arr.itemsize != 4:  # highly unlikely
            raise TypeError("Unexpected float itemsize in byte decoding")
        return "[" + ",".join(format(float(x), "g") for x in arr) + "]"
    # numpy array or scalar
    if isinstance(val, np.ndarray):
        if val.ndim != 1:
            raise TypeError(f"NumPy array must be 1-D, got shape {val.shape}")
        if val.dtype == object:
            raise TypeError("NumPy array has object dtype; cannot convert")
        val = val.tolist()
    elif isinstance(val, (np.floating, np.integer)):
        # Treat numpy scalar as single-dimension vector
        val = [float(val)]
    # pandas Series
    if isinstance(val, pd.Series):
        val = val.to_list()
    # list / tuple
    if isinstance(val, (list, tuple)):
        try:
            return "[" + ",".join(format(float(x), "g") for x in val) + "]"
        except Exception as e:
            raise TypeError(f"Failed to convert sequence to vector literal: {e}") from e
    # generic iterable
    if isinstance(val, _Iterable):
        seq = list(val)
        if not seq:
            raise TypeError("Empty iterable cannot form a vector literal")
        try:
            return "[" + ",".join(format(float(x), "g") for x in seq) + "]"
        except Exception as e:
            raise TypeError(
                "Iterable contained non-numeric values; types: "
                + ",".join(sorted({type(x).__name__ for x in seq[:5]})),
            ) from e
    raise TypeError(f"Unsupported embedding value type for vector literal: {type(val)}")


def record_generator() -> (
    Iterable[Tuple[str, str, str, str, str, str, Optional[str], bytes]]
):
    """Generate transformed records ready for insertion.

    Embeddings are converted to pgvector literals; geom converted to WKB bytes.
    Any rows with unconvertible embeddings are skipped with a warning.
    Enforces that embedding dimension matches PGVECTOR_DIM if provided.
    """
    # Read only required columns (plus geometry) to avoid accidental order issues
    required_cols = [c for c in table_columns if c != "geom"] + ["geometry"]
    gdf = (
        gpd.read_parquet(os.environ["GEOPARQUET_PATH"], columns=required_cols)
        .drop_duplicates(subset=["id", "metadata_text"])
        .rename(columns={"geometry": "geom"})
    )
    # Reorder columns exactly as table_columns specifies for deterministic tuple unpacking
    gdf = gdf[[c for c in table_columns]]
    skipped = 0
    dim_mismatch = 0
    first_type_logged = False
    expected_dim = int(os.environ.get("PGVECTOR_DIM", "1024"))
    for row in gdf.itertuples(index=False):
        (id_, name, type_, description, url, metadata_text, embeddings, geom) = row
        if not first_type_logged:
            logging.info("Embeddings sample python type: %s", type(embeddings))
            first_type_logged = True
        try:
            emb_lit = vector_literal(embeddings)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                logging.warning(
                    "Skipping row id=%s due to embedding conversion error: %s (type=%s)",
                    id_,
                    e,
                    type(embeddings),
                )
            continue
        # Dimension enforcement
        if emb_lit is not None:
            # count commas + 1 is dimension (fast, avoids splitting large lists fully)
            dim = emb_lit.count(",") + 1
            if dim != expected_dim:
                dim_mismatch += 1
                if dim_mismatch <= 10:
                    logging.warning(
                        "Skipping row id=%s: embedding dim %s != expected %s",
                        id_,
                        dim,
                        expected_dim,
                    )
                continue
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
    if skipped:
        logging.warning(
            "Total rows skipped due to embedding conversion issues: %s",
            skipped,
        )
    if dim_mismatch:
        logging.warning(
            "Total rows skipped due to embedding dimension mismatch: %s",
            dim_mismatch,
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
