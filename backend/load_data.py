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
from psycopg import AsyncConnection, sql

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

INIT_LOADER_BATCH_SIZE = int(os.environ.get("INIT_LOADER_BATCH_SIZE", "500"))


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
        # Vector index related defaults
        "VECTOR_INDEX_NAME": "layer_embedding_index",
        "VECTOR_INDEX_TYPE": "ivfflat",
        "VECTOR_METRIC": "l2",
        "VECTOR_IVFFLAT_LISTS": "100",
        "VECTOR_HNSW_M": "16",
        "VECTOR_HNSW_EF_CONSTRUCTION": "64",
        "PGVECTOR_DIM": "1024",
        # High-volume load tuning
        "LOADER_COMMIT_INTERVAL": "100000",
        "LOADER_PERFORMANCE_TWEAKS": "true",
        "LOADER_WORK_MEM": "128MB",
        "LOADER_TEMP_BUFFERS": "32MB",
        "LOADER_SYNCHRONOUS_COMMIT": "off",
        # Loader method: copy (default) or executemany
        "LOADER_METHOD": "copy",
        # New vector index memory tuning
        # Initial maintenance_work_mem to request for index build (MB)
        "VECTOR_MAINTENANCE_WORK_MEM_MB": "512",
        # Allow one retry raising maintenance_work_mem up to this cap if memory error encountered
        "VECTOR_MAINTENANCE_WORK_MEM_MAX_MB": "2048",
        # Enable automatic retry if memory error indicates higher requirement
        "VECTOR_AUTOTUNE_INDEX_MEMORY": "true",
        # Performance optimization flags
        "ENFORCE_EMBED_DIM": "true",
        "FAST_DEV_SKIP_INDEX": "false",
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
    os.environ["VECTOR_INDEX_NAME"] = safe_identifier(os.environ["VECTOR_INDEX_NAME"])

    dim = os.environ.get("PGVECTOR_DIM", "1024")
    if not dim.isdigit():
        raise ValueError("PGVECTOR_DIM must be an integer string")
    os.environ["PGVECTOR_DIM"] = dim

    index_type = os.environ.get("VECTOR_INDEX_TYPE", "ivfflat").lower()
    if index_type not in {"ivfflat", "hnsw"}:
        raise ValueError("VECTOR_INDEX_TYPE must be either 'ivfflat' or 'hnsw'")
    os.environ["VECTOR_INDEX_TYPE"] = index_type

    metric = os.environ.get("VECTOR_METRIC", "l2").lower()
    metric_map = {
        "l2": "vector_l2_ops",
        "cosine": "vector_cosine_ops",
        "ip": "vector_ip_ops",
    }
    if metric not in metric_map:
        raise ValueError("VECTOR_METRIC must be one of: l2, cosine, ip")
    os.environ["VECTOR_METRIC"] = metric
    os.environ["_VECTOR_OPS_CLASS"] = metric_map[metric]

    lists = os.environ.get("VECTOR_IVFFLAT_LISTS", "100")
    if not lists.isdigit() or int(lists) <= 0:
        raise ValueError("VECTOR_IVFFLAT_LISTS must be positive integer")
    m_val = os.environ.get("VECTOR_HNSW_M", "16")
    efc = os.environ.get("VECTOR_HNSW_EF_CONSTRUCTION", "64")
    if not (m_val.isdigit() and int(m_val) > 0):
        raise ValueError("VECTOR_HNSW_M must be positive integer")
    if not (efc.isdigit() and int(efc) > 0):
        raise ValueError("VECTOR_HNSW_EF_CONSTRUCTION must be positive integer")

    commit_interval = os.environ.get("LOADER_COMMIT_INTERVAL", "100000")
    if not (commit_interval.isdigit() and int(commit_interval) >= 0):
        raise ValueError("LOADER_COMMIT_INTERVAL must be >= 0 integer")

    loader_method = os.environ.get("LOADER_METHOD", "copy").lower()
    if loader_method not in {"copy", "copy_direct", "executemany"}:
        raise ValueError("LOADER_METHOD must be 'copy', 'copy_direct', or 'executemany'")
    os.environ["LOADER_METHOD"] = loader_method

    # maintenance work mem numbers
    base_mem = os.environ.get("VECTOR_MAINTENANCE_WORK_MEM_MB", "512")
    max_mem = os.environ.get("VECTOR_MAINTENANCE_WORK_MEM_MAX_MB", "2048")
    if not (base_mem.isdigit() and int(base_mem) > 0):
        raise ValueError("VECTOR_MAINTENANCE_WORK_MEM_MB must be positive integer (MB)")
    if not (max_mem.isdigit() and int(max_mem) >= int(base_mem)):
        raise ValueError(
            "VECTOR_MAINTENANCE_WORK_MEM_MAX_MB must be >= base mem and positive integer",
        )
    # store normalized
    os.environ["VECTOR_MAINTENANCE_WORK_MEM_MB"] = base_mem
    os.environ["VECTOR_MAINTENANCE_WORK_MEM_MAX_MB"] = max_mem
    auto_tune = os.environ.get("VECTOR_AUTOTUNE_INDEX_MEMORY", "true").lower()
    if auto_tune not in {"true", "false"}:
        raise ValueError("VECTOR_AUTOTUNE_INDEX_MEMORY must be true or false")
    os.environ["VECTOR_AUTOTUNE_INDEX_MEMORY"] = auto_tune

    # Validate new performance flags
    enforce_dim = os.environ.get("ENFORCE_EMBED_DIM", "true").lower()
    if enforce_dim not in {"true", "false"}:
        raise ValueError("ENFORCE_EMBED_DIM must be true or false")
    os.environ["ENFORCE_EMBED_DIM"] = enforce_dim

    skip_index = os.environ.get("FAST_DEV_SKIP_INDEX", "false").lower()
    if skip_index not in {"true", "false"}:
        raise ValueError("FAST_DEV_SKIP_INDEX must be true or false")
    os.environ["FAST_DEV_SKIP_INDEX"] = skip_index


def check_geoparquet_path() -> None:
    """Validate GEOPARQUET_PATH exists before attempting load.

    Raises FileNotFoundError if the path is missing. Split out so we can call
    only when we actually need to load data (avoids requiring the file when
    table already populated and we just want to ensure indexes / user)."""
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
    """Create the table if it doesn't already exist (idempotent)."""
    table_ident = sql.Identifier(os.environ["POSTGRES_TABLE"])  # validated earlier
    dim_literal = sql.Literal(int(os.environ.get("PGVECTOR_DIM", "1024")))
    create_table_stmt = sql.SQL(
        """CREATE TABLE IF NOT EXISTS {table} (
    id text,
    name text,
    type text,
    description text,
    url text,
    metadata_text text,
    embeddings vector({dim}),
    geom geometry(Polygon,4326)
) TABLESPACE pg_default;""",
    ).format(table=table_ident, dim=dim_literal)
    async with conn.cursor() as cur:
        # Extensions (no dynamic parts)
        await cur.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS postgis"))
        await cur.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
        await cur.execute(create_table_stmt)


make_index_sql = """CREATE INDEX IF NOT EXISTS {index_name}
    ON {table} USING gist
    (geom)
    TABLESPACE pg_default;"""


async def make_index(conn: AsyncConnection) -> None:
    """Create the spatial index if it doesn't already exist (idempotent)."""
    index_ident = sql.Identifier(os.environ["SPATIAL_INDEX_NAME"])  # validated
    table_ident = sql.Identifier(os.environ["POSTGRES_TABLE"])  # validated
    create_index_stmt = sql.SQL(
        "CREATE INDEX IF NOT EXISTS {index} ON {table} USING gist (geom) TABLESPACE pg_default",
    ).format(index=index_ident, table=table_ident)
    async with conn.cursor() as cur:
        await cur.execute(create_index_stmt)


async def make_vector_index(conn: AsyncConnection) -> None:
    """Create the vector similarity index on embeddings if it does not exist.

    Supports ivfflat (default) and hnsw index types.
    Parameters are controlled via environment variables set/validated in check_env_vars().
    - VECTOR_INDEX_NAME (identifier)
    - VECTOR_INDEX_TYPE (ivfflat | hnsw)
    - VECTOR_METRIC (l2 | cosine | ip) -> mapped to ops class
    - VECTOR_IVFFLAT_LISTS (only for ivfflat)
    - VECTOR_HNSW_M, VECTOR_HNSW_EF_CONSTRUCTION (only for hnsw)

    If creation fails (e.g., unsupported index type in installed pgvector version), logs a warning.
    """
    table_ident = sql.Identifier(os.environ["POSTGRES_TABLE"])
    index_ident = sql.Identifier(os.environ["VECTOR_INDEX_NAME"])
    ops_class = os.environ["_VECTOR_OPS_CLASS"]  # already validated mapping
    index_type = os.environ["VECTOR_INDEX_TYPE"]
    autotune = os.environ.get("VECTOR_AUTOTUNE_INDEX_MEMORY", "true") == "true"
    base_mem_mb = int(os.environ.get("VECTOR_MAINTENANCE_WORK_MEM_MB", "512"))
    max_mem_mb = int(os.environ.get("VECTOR_MAINTENANCE_WORK_MEM_MAX_MB", "2048"))

    async with conn.cursor() as cur:
        # Set initial maintenance_work_mem for index build (best-effort)
        if index_type == "ivfflat":
            try:
                await cur.execute(
                    sql.SQL("SET maintenance_work_mem TO {}MB").format(
                        sql.SQL(str(base_mem_mb)),
                    ),
                )
                logging.info(
                    "Set maintenance_work_mem to %sMB for vector index build",
                    base_mem_mb,
                )
            except Exception as e:  # pragma: no cover
                logging.info(
                    "Could not set maintenance_work_mem (%sMB): %s",
                    base_mem_mb,
                    e,
                )

        def build_stmt(lists_val: int | None = None):
            if index_type == "ivfflat":
                lists_local = (
                    lists_val
                    if lists_val is not None
                    else int(os.environ["VECTOR_IVFFLAT_LISTS"])
                )
                return sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} USING ivfflat (embeddings {ops}) WITH (lists = {lists})",
                ).format(
                    index=index_ident,
                    table=table_ident,
                    ops=sql.SQL(ops_class),
                    lists=sql.Literal(lists_local),
                )
            else:
                m_val = int(os.environ["VECTOR_HNSW_M"])
                efc = int(os.environ["VECTOR_HNSW_EF_CONSTRUCTION"])
                return sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} USING hnsw (embeddings {ops}) WITH (m = {m}, ef_construction = {efc})",
                ).format(
                    index=index_ident,
                    table=table_ident,
                    ops=sql.SQL(ops_class),
                    m=sql.Literal(m_val),
                    efc=sql.Literal(efc),
                )

        current_lists = (
            int(os.environ.get("VECTOR_IVFFLAT_LISTS", "100"))
            if index_type == "ivfflat"
            else None
        )
        tried_raise_mem = False
        while True:
            ddl = build_stmt(current_lists)
            try:
                await cur.execute(ddl)
                await cur.execute(sql.SQL("ANALYZE {table}").format(table=table_ident))
                logging.info(
                    "Vector index ensured (type=%s, metric=%s, name=%s, lists=%s)",
                    index_type,
                    os.environ["VECTOR_METRIC"],
                    os.environ["VECTOR_INDEX_NAME"],
                    current_lists if current_lists is not None else "-",
                )
                break
            except Exception as e:  # pragma: no cover
                msg = str(e)
                if (
                    index_type == "ivfflat"
                    and autotune
                    and "memory required is" in msg
                    and "maintenance_work_mem" in msg
                ):
                    # Parse required and current memory from error
                    m = re.search(
                        r"memory required is (\d+) MB, maintenance_work_mem is (\d+) MB",
                        msg,
                    )
                    if m:
                        required_mb = int(m.group(1))
                        current_mb = int(m.group(2))
                        logging.warning(
                            "Vector index build requires %sMB (current maintenance_work_mem=%sMB)",
                            required_mb,
                            current_mb,
                        )
                        # Try raising maintenance_work_mem once if not yet and within cap
                        if not tried_raise_mem and required_mb <= max_mem_mb:
                            try:
                                await cur.execute(
                                    sql.SQL("SET maintenance_work_mem TO {}MB").format(
                                        sql.SQL(str(required_mb)),
                                    ),
                                )
                                tried_raise_mem = True
                                logging.info(
                                    "Raised maintenance_work_mem to %sMB; retrying index build (lists=%s)",
                                    required_mb,
                                    current_lists,
                                )
                                continue
                            except Exception as se:
                                logging.info(
                                    "Failed to raise maintenance_work_mem to %sMB: %s",
                                    required_mb,
                                    se,
                                )
                        # If cannot raise memory, attempt to reduce lists heuristically
                        if current_lists and current_lists > 50:
                            new_lists = max(50, int(current_lists * 0.75))
                            if new_lists < current_lists:
                                logging.warning(
                                    "Reducing ivfflat lists from %s to %s and retrying due to memory pressure",
                                    current_lists,
                                    new_lists,
                                )
                                current_lists = new_lists
                                continue
                    logging.warning(
                        "Vector index creation failed after autotune attempts; proceeding without vector index: %s",
                        e,
                    )
                    break
                else:
                    logging.warning(
                        "Vector index creation failed (type=%s). Continuing without it: %s",
                        index_type,
                        e,
                    )
                    break


make_user_sql = """CREATE ROLE {read_only_user} WITH LOGIN PASSWORD '{read_only_user_password}';
GRANT CONNECT ON DATABASE {database} TO {read_only_user};
GRANT USAGE ON SCHEMA {schema} TO {read_only_user};
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} FROM {read_only_user};
GRANT SELECT ON {table} TO {read_only_user};"""


async def make_user(conn: AsyncConnection) -> None:
    """Create (if missing) and grant read-only privileges to the configured user."""
    user_ident = sql.Identifier(os.environ["READ_ONLY_POSTGRES_USER"])  # validated
    pwd_literal = sql.Literal(
        os.environ["READ_ONLY_POSTGRES_USER_PASSWORD"],
    )  # literal for safe quoting
    db_ident = sql.Identifier(os.environ["POSTGRES_DB"])  # database name
    schema_ident = sql.Identifier(os.environ["POSTGRES_SCHEMA"])  # validated
    table_ident = sql.Identifier(os.environ["POSTGRES_TABLE"])  # validated

    statements = [
        # Create role (will error if exists; we'll catch & continue)
        sql.SQL("CREATE ROLE {user} WITH LOGIN PASSWORD {pwd}").format(
            user=user_ident,
            pwd=pwd_literal,
        ),
        sql.SQL("GRANT CONNECT ON DATABASE {db} TO {user}").format(
            db=db_ident,
            user=user_ident,
        ),
        sql.SQL("GRANT USAGE ON SCHEMA {schema} TO {user}").format(
            schema=schema_ident,
            user=user_ident,
        ),
        sql.SQL(
            "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} FROM {user}",
        ).format(schema=schema_ident, user=user_ident),
        sql.SQL("GRANT SELECT ON {table} TO {user}").format(
            table=table_ident,
            user=user_ident,
        ),
    ]

    for stmt in statements:
        async with conn.cursor() as cur:
            try:
                await cur.execute(stmt)
            except Exception as e:
                # Non-fatal: role may already exist or grant already applied
                logging.info(f"User privilege statement skipped/partial: {e}")
                continue


async def check_table_length(conn: AsyncConnection) -> bool:
    """Return True if the target table is empty."""
    table_ident = sql.Identifier(os.environ["POSTGRES_TABLE"])
    query = sql.SQL("SELECT COUNT(*) FROM {table}").format(table=table_ident)
    async with conn.cursor() as cur:
        await cur.execute(query)
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
    enforce_dim = os.environ.get("ENFORCE_EMBED_DIM", "true").lower() == "true"
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
        # Dimension enforcement (can be disabled for performance)
        if enforce_dim and emb_lit is not None:
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


def prepare_dataframe():
    """Prepare the entire dataframe with vectorized operations for fast loading.
    
    Returns:
        tuple: (gdf, emb_literals_list, geom_wkb_list) where:
            - gdf: GeoDataFrame with all columns
            - emb_literals_list: list of embedding string literals
            - geom_wkb_list: list of WKB bytes for geometries
    """
    import time
    start_time = time.time()
    
    # Read data
    required_cols = [c for c in table_columns if c != "geom"] + ["geometry"]
    gdf = (
        gpd.read_parquet(os.environ["GEOPARQUET_PATH"], columns=required_cols)
        .drop_duplicates(subset=["id", "metadata_text"])
        .rename(columns={"geometry": "geom"})
    )
    gdf = gdf[[c for c in table_columns]]
    read_time = time.time() - start_time
    logging.info(f"Read {len(gdf)} rows in {read_time:.2f}s")
    
    # Vectorize WKB conversion
    vec_start = time.time()
    geom_wkb_list = [shapely.wkb.dumps(g, hex=False) for g in gdf["geom"]]
    wkb_time = time.time() - vec_start
    logging.info(f"Vectorized WKB conversion in {wkb_time:.2f}s")
    
    # Vectorize embedding literal conversion
    emb_start = time.time()
    enforce_dim = os.environ.get("ENFORCE_EMBED_DIM", "true").lower() == "true"
    expected_dim = int(os.environ.get("PGVECTOR_DIM", "1024"))
    
    emb_literals_list = []
    skipped = 0
    dim_mismatch = 0
    
    for i, emb_val in enumerate(gdf["embeddings"]):
        try:
            emb_lit = vector_literal(emb_val)
            
            # Dimension check (if enabled)
            if enforce_dim and emb_lit is not None:
                dim = emb_lit.count(",") + 1
                if dim != expected_dim:
                    dim_mismatch += 1
                    if dim_mismatch <= 10:
                        logging.warning(
                            f"Skipping row idx={i}: embedding dim {dim} != expected {expected_dim}",
                        )
                    emb_literals_list.append(None)
                    continue
            
            emb_literals_list.append(emb_lit)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                logging.warning(f"Embedding conversion error at idx={i}: {e}")
            emb_literals_list.append(None)
    
    emb_time = time.time() - emb_start
    logging.info(f"Vectorized embedding literals in {emb_time:.2f}s (skipped={skipped}, dim_mismatch={dim_mismatch})")
    
    total_time = time.time() - start_time
    logging.info(f"Total dataframe preparation: {total_time:.2f}s")
    
    return gdf, emb_literals_list, geom_wkb_list


def batched(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


INSERT_SQL_TEMPLATE = sql.SQL(
    "INSERT INTO {table} (id,name,type,description,url,metadata_text,embeddings,geom) VALUES (%s,%s,%s,%s,%s,%s,%s::vector,ST_GeomFromWKB(%s,4326))",
)


async def load_data(conn: AsyncConnection) -> None:
    """Load data using COPY (default), copy_direct (fastest), or batched executemany (fallback).

    COPY path:
      1. Create a temporary staging table with embeddings TEXT and geom_wkb BYTEA.
      2. Stream rows via PostgreSQL COPY.
      3. Insert-transform into target table casting embeddings::vector and ST_GeomFromWKB.
      4. Single final commit.

    COPY_DIRECT path (fastest):
      1. Pre-read and vectorize entire dataset (WKB + embedding literals).
      2. COPY directly to target table (no staging).
      3. Single final commit.

    Executemany fallback retains prior behavior (honors LOADER_COMMIT_INTERVAL).
    """
    import time
    total_start = time.time()
    
    method = os.environ.get("LOADER_METHOD", "copy")
    if method == "executemany":
        logging.info("LOADER_METHOD=executemany selected (legacy path)")
        await _load_data_executemany(conn)
        return
    
    if method == "copy_direct":
        logging.info("LOADER_METHOD=copy_direct selected (direct bulk load, no staging)")
        await _load_data_copy_direct(conn)
        total_time = time.time() - total_start
        logging.info(f"Total load_data time: {total_time:.2f}s")
        return
    
    logging.info("LOADER_METHOD=copy selected (bulk COPY path)")

    # Session tweaks (still valuable for COPY)
    tweaks = os.environ.get("LOADER_PERFORMANCE_TWEAKS", "true").lower() == "true"
    async with conn.cursor() as cur:
        if tweaks:
            settings = {
                "synchronous_commit": os.environ.get(
                    "LOADER_SYNCHRONOUS_COMMIT",
                    "off",
                ),
                "work_mem": os.environ.get("LOADER_WORK_MEM", "128MB"),
                "temp_buffers": os.environ.get("LOADER_TEMP_BUFFERS", "32MB"),
            }
            for k, v in settings.items():
                try:
                    await cur.execute(
                        sql.SQL("SET {} TO {}").format(  # type: ignore[arg-type]
                            sql.SQL(k),
                            sql.Literal(v),
                        ),
                    )
                except Exception as e:  # pragma: no cover
                    logging.info(f"Skipping load tweak {k}={v}: {e}")

        # 1. Temp staging table
        await cur.execute(
            """
            CREATE TEMP TABLE temp_layers_stage (
                id text,
                name text,
                type text,
                description text,
                url text,
                metadata_text text,
                embeddings text,
                geom_wkb bytea
            ) ON COMMIT DROP
            """,
        )
        logging.info("Created temporary staging table for COPY")

        # 2. COPY stream
        copy_stmt = "COPY temp_layers_stage (id,name,type,description,url,metadata_text,embeddings,geom_wkb) FROM STDIN"
        total = 0
        copy_start = time.time()
        async with cur.copy(copy_stmt) as copy:
            for rec in record_generator():
                (
                    id_,
                    name,
                    type_,
                    description,
                    url,
                    metadata_text,
                    emb_lit,
                    geom_wkb,
                ) = rec
                # emb_lit already like '[...]' or None
                # psycopg will adapt None -> NULL, bytes -> bytea hex
                await copy.write_row(
                    [
                        id_,
                        name,
                        type_,
                        description,
                        url,
                        metadata_text,
                        emb_lit,
                        psycopg.Binary(geom_wkb),
                    ],
                )
                total += 1
                if total % 100000 == 0:
                    logging.info("COPY streamed %d rows...", total)
        copy_time = time.time() - copy_start
        logging.info(f"Finished COPY streaming {total} rows in {copy_time:.2f}s")

        # 3. Insert-transform
        transform_start = time.time()
        target_table = sql.Identifier(os.environ["POSTGRES_TABLE"])
        insert_transform = sql.SQL(
            """
            INSERT INTO {target} (id,name,type,description,url,metadata_text,embeddings,geom)
            SELECT id,name,type,description,url,metadata_text,
                   CASE WHEN embeddings IS NULL THEN NULL ELSE embeddings::vector END,
                   ST_SetSRID(ST_GeomFromWKB(geom_wkb),4326)
            FROM temp_layers_stage
            """,
        ).format(target=target_table)
        await cur.execute(insert_transform)
        transform_time = time.time() - transform_start
        logging.info(f"Inserted {total} rows from staging into target table in {transform_time:.2f}s")
        await conn.commit()
        logging.info("Committed COPY load")

    total_time = time.time() - total_start
    logging.info(f"Data load complete (COPY path) - total time: {total_time:.2f}s")


async def _load_data_copy_direct(conn: AsyncConnection) -> None:
    """Fast direct COPY to target table using pre-vectorized data.
    
    This is the fastest loading method as it:
    1. Pre-reads entire dataset and vectorizes WKB/embeddings in Python
    2. Uses a staging approach but with all data pre-processed
    3. Single transform insert to target
    
    Best for: Known-good data where dimension checking overhead is unnecessary.
    """
    import time
    
    # Prepare all data upfront with vectorization
    prep_start = time.time()
    gdf, emb_literals_list, geom_wkb_list = prepare_dataframe()
    prep_time = time.time() - prep_start
    logging.info(f"Dataframe preparation complete in {prep_time:.2f}s")
    
    # Apply session tweaks
    tweaks = os.environ.get("LOADER_PERFORMANCE_TWEAKS", "true").lower() == "true"
    async with conn.cursor() as cur:
        if tweaks:
            settings = {
                "synchronous_commit": os.environ.get(
                    "LOADER_SYNCHRONOUS_COMMIT",
                    "off",
                ),
                "work_mem": os.environ.get("LOADER_WORK_MEM", "128MB"),
                "temp_buffers": os.environ.get("LOADER_TEMP_BUFFERS", "32MB"),
            }
            for k, v in settings.items():
                try:
                    await cur.execute(
                        sql.SQL("SET {} TO {}").format(  # type: ignore[arg-type]
                            sql.SQL(k),
                            sql.Literal(v),
                        ),
                    )
                except Exception as e:  # pragma: no cover
                    logging.info(f"Skipping load tweak {k}={v}: {e}")
        
        # Create temp staging table
        await cur.execute(
            """
            CREATE TEMP TABLE temp_layers_stage (
                id text,
                name text,
                type text,
                description text,
                url text,
                metadata_text text,
                embeddings text,
                geom_wkb bytea
            ) ON COMMIT DROP
            """,
        )
        
        # COPY to staging with pre-vectorized data
        copy_stmt = "COPY temp_layers_stage (id,name,type,description,url,metadata_text,embeddings,geom_wkb) FROM STDIN"
        
        copy_start = time.time()
        total = 0
        skipped = 0
        
        async with cur.copy(copy_stmt) as copy:
            for i, row in enumerate(gdf.itertuples(index=False)):
                emb_lit = emb_literals_list[i]
                geom_wkb = geom_wkb_list[i]
                
                # Skip rows with invalid embeddings (already logged during prep)
                if emb_lit is None:
                    skipped += 1
                    continue
                
                (id_, name, type_, description, url, metadata_text, _, _) = row
                
                await copy.write_row(
                    [
                        id_,
                        name,
                        type_,
                        description,
                        url,
                        metadata_text,
                        emb_lit,
                        psycopg.Binary(geom_wkb),
                    ],
                )
                total += 1
                if total % 100000 == 0:
                    logging.info(f"COPY direct streamed {total} rows...")
        
        copy_time = time.time() - copy_start
        logging.info(f"Finished COPY to staging {total} rows (skipped={skipped}) in {copy_time:.2f}s")
        
        # Transform and insert to target
        transform_start = time.time()
        target_table = sql.Identifier(os.environ["POSTGRES_TABLE"])
        insert_transform = sql.SQL(
            """
            INSERT INTO {target} (id,name,type,description,url,metadata_text,embeddings,geom)
            SELECT id,name,type,description,url,metadata_text,
                   CASE WHEN embeddings IS NULL THEN NULL ELSE embeddings::vector END,
                   ST_SetSRID(ST_GeomFromWKB(geom_wkb),4326)
            FROM temp_layers_stage
            """,
        ).format(target=target_table)
        await cur.execute(insert_transform)
        transform_time = time.time() - transform_start
        logging.info(f"Transformed and inserted {total} rows in {transform_time:.2f}s")
        
        await conn.commit()
        logging.info("Committed direct COPY load")
    
    logging.info("Data load complete (copy_direct path)")


# Legacy executemany path retained for fallback / comparison
async def _load_data_executemany(
    conn: AsyncConnection,
) -> None:  # pragma: no cover (optional path)
    insert_sql = INSERT_SQL_TEMPLATE.format(
        table=sql.Identifier(os.environ["POSTGRES_TABLE"]),
    )
    commit_interval = int(os.environ.get("LOADER_COMMIT_INTERVAL", "100000"))
    tweaks = os.environ.get("LOADER_PERFORMANCE_TWEAKS", "true").lower() == "true"

    async with conn.cursor() as cur:
        if tweaks:
            settings = {
                "synchronous_commit": os.environ.get(
                    "LOADER_SYNCHRONOUS_COMMIT",
                    "off",
                ),
                "work_mem": os.environ.get("LOADER_WORK_MEM", "128MB"),
                "temp_buffers": os.environ.get("LOADER_TEMP_BUFFERS", "32MB"),
            }
            for k, v in settings.items():
                try:
                    await cur.execute(
                        sql.SQL("SET {} TO {}").format(  # type: ignore[arg-type]
                            sql.SQL(k),
                            sql.Literal(v),
                        ),
                    )
                except Exception as e:
                    logging.info(f"Skipping load tweak {k}={v}: {e}")
        total = 0
        batch_no = 0
        if commit_interval == 0:
            async with conn.transaction():
                for chunk in batched(record_generator(), INIT_LOADER_BATCH_SIZE):
                    await cur.executemany(insert_sql, chunk)  # type: ignore[arg-type]
                    total += len(chunk)
                    batch_no += 1
                    if batch_no % 50 == 0:
                        logging.info(
                            "Inserted %d rows so far (single transaction mode)",
                            total,
                        )
        else:
            for chunk in batched(record_generator(), INIT_LOADER_BATCH_SIZE):
                await cur.executemany(insert_sql, chunk)  # type: ignore[arg-type]
                total += len(chunk)
                batch_no += 1
                if total % commit_interval == 0:
                    await conn.commit()
                    logging.info(
                        "Committed %d rows (commit interval %d)",
                        total,
                        commit_interval,
                    )
                elif batch_no % 50 == 0:
                    logging.info(
                        "Inserted %d rows so far (pending commit at %d)",
                        total,
                        commit_interval,
                    )
            await conn.commit()
            logging.info("Final commit after inserting %d rows", total)
    logging.info("Data load complete (executemany path)")


async def amain() -> None:
    check_env_vars()
    logging.info("Environment variables are valid")

    conn = await get_connection()
    logging.info("Connected to database")

    logging.info("Checking / creating table")
    await make_table(conn)
    # Explicit commit so DDL is persisted immediately (esp. if later logic early-exits)
    try:
        await conn.commit()
        logging.info("Committed table creation DDL")
    except Exception as e:  # pragma: no cover (defensive)
        logging.warning(f"Commit after table creation failed (continuing): {e}")

    len_zero = await check_table_length(conn)

    if len_zero:
        check_geoparquet_path()
        logging.info("Geoparquet path is valid")

        logging.info("Loading data")
        await load_data(conn)  # internal transaction handles its own commit
        logging.info("Data loaded")

        skip_index = os.environ.get("FAST_DEV_SKIP_INDEX", "false").lower() == "true"
        
        if skip_index:
            logging.info("FAST_DEV_SKIP_INDEX=true, skipping index creation")
        else:
            logging.info("Making spatial index")
            await make_index(conn)
            logging.info("Making vector index")
            await make_vector_index(conn)
            # Commit index creation
            try:
                await conn.commit()
                logging.info("Committed spatial & vector index creation")
            except Exception as e:  # pragma: no cover
                logging.warning(f"Commit after index creation failed (continuing): {e}")

        logging.info("Adding read-only user")
        await make_user(conn)
        # Commit role & grants so they persist (previously they were lost on close)
        try:
            await conn.commit()
            logging.info("Committed read-only user creation & grants")
        except Exception as e:  # pragma: no cover
            logging.warning(f"Commit after user creation failed (continuing): {e}")

    else:
        logging.info(
            "Table already had data; skipping load but ensuring indexes & read-only user exist",
        )
        await make_index(conn)  # safe idempotent, ensures spatial index
        await make_vector_index(conn)  # ensure vector index
        try:
            await conn.commit()
        except Exception as e:  # pragma: no cover
            logging.warning("Commit after ensuring indexes skipped or failed: %s", e)
        await make_user(conn)
        try:
            await conn.commit()
            logging.info("Committed (re)creation of read-only user & grants")
        except Exception as e:  # pragma: no cover
            logging.warning(f"Commit after user recreation failed (continuing): {e}")

    logging.info("All done")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(amain())
