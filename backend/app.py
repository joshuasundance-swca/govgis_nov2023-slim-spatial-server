import os
from contextlib import asynccontextmanager
import re
import logging

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from fastapi import FastAPI, Depends

try:  # graceful handling if dependency not installed at analysis time
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "langchain-huggingface package is required. Ensure it is installed (requirements.txt)",
    ) from e
import psycopg
from psycopg import sql
from models import SemanticSearchRequest, SearchResponse, LayerResult  # type: ignore[attr-defined]

__version__ = "0.0.1"

# Initialize a global async connection pool (opened in FastAPI lifespan)
pool: AsyncConnectionPool | None = None
logger = logging.getLogger(__name__)


def _safe_identifier(name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):  # conservative
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


async def ensure_read_only_user() -> None:
    """Ensure the configured read-only user/privileges exist.

    Attempts a connection with the read-only credentials; on failure (role missing / auth
    failure) it connects with the superuser credentials and (re)applies role + grants.
    Safe to run repeatedly. If the role already exists its password is updated to match
    the current environment configuration.
    """
    host = os.environ["POSTGRES_HOST"]
    db = os.environ["POSTGRES_DB"]
    ro_user = _safe_identifier(os.environ["READ_ONLY_POSTGRES_USER"])
    ro_pwd = os.environ["READ_ONLY_POSTGRES_USER_PASSWORD"]
    su_user = os.environ["POSTGRES_USER"]
    su_pwd = os.environ["POSTGRES_PASSWORD"]
    schema = _safe_identifier(os.environ.get("POSTGRES_SCHEMA", "public"))
    table = _safe_identifier(os.environ.get("POSTGRES_TABLE", "layers"))

    # First try connecting as the read-only user
    try:
        conn = await psycopg.AsyncConnection.connect(
            host=host,
            dbname=db,
            user=ro_user,
            password=ro_pwd,
        )
        await conn.close()
        logger.info("Read-only user '%s' already functional", ro_user)
        return
    except Exception as e:  # pragma: no cover (network/db dependent)
        logger.warning("Read-only user connection failed (%s); attempting creation", e)

    # Connect as superuser to (re)create role + grants
    async with await psycopg.AsyncConnection.connect(
        host=host,
        dbname=db,
        user=su_user,
        password=su_pwd,
    ) as su_conn:
        stmts = [
            # Try create; if it already exists we'll issue an ALTER below
            sql.SQL("CREATE ROLE {u} WITH LOGIN PASSWORD {p}").format(
                u=sql.Identifier(ro_user),
                p=sql.Literal(ro_pwd),
            ),
        ]
        for stmt in stmts:
            try:
                async with su_conn.cursor() as cur:
                    await cur.execute(stmt)
            except Exception as e:  # likely role exists
                if "already exists" in str(e).lower():
                    logger.info("Role %s exists; updating password", ro_user)
                    async with su_conn.cursor() as cur2:
                        await cur2.execute(
                            sql.SQL("ALTER ROLE {u} WITH LOGIN PASSWORD {p}").format(
                                u=sql.Identifier(ro_user),
                                p=sql.Literal(ro_pwd),
                            ),
                        )
                else:
                    logger.info("Role create skipped/partial: %s", e)
        # Grants (idempotent-ish)
        grant_stmts = [
            sql.SQL("GRANT CONNECT ON DATABASE {db} TO {u}").format(
                db=sql.Identifier(db),
                u=sql.Identifier(ro_user),
            ),
            sql.SQL("GRANT USAGE ON SCHEMA {s} TO {u}").format(
                s=sql.Identifier(schema),
                u=sql.Identifier(ro_user),
            ),
            sql.SQL(
                "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {s} FROM {u}",
            ).format(s=sql.Identifier(schema), u=sql.Identifier(ro_user)),
            sql.SQL("GRANT SELECT ON {t} TO {u}").format(
                t=sql.Identifier(table),
                u=sql.Identifier(ro_user),
            ),
        ]
        for stmt in grant_stmts:
            try:
                async with su_conn.cursor() as cur:
                    await cur.execute(stmt)
            except Exception as e:  # likely already granted
                logger.info("Grant statement skipped/partial: %s", e)
        try:
            await su_conn.commit()
        except Exception as e:  # pragma: no cover
            logger.warning("Commit after ensuring read-only user failed: %s", e)
    # Final sanity: try connecting again
    try:
        conn = await psycopg.AsyncConnection.connect(
            host=host,
            dbname=db,
            user=ro_user,
            password=ro_pwd,
        )
        await conn.close()
        logger.info("Read-only user '%s' verified after creation/update", ro_user)
    except Exception as e:  # pragma: no cover
        logger.error("Unable to verify read-only user after creation/update: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):  # FastAPI will call this on startup/shutdown
    global pool
    # Ensure user exists before opening pool (self-heals older deployments)
    await ensure_read_only_user()
    pool = AsyncConnectionPool(
        conninfo=(
            f"host={os.environ['POSTGRES_HOST']} "
            f"user={os.environ['READ_ONLY_POSTGRES_USER']} "
            f"password={os.environ['READ_ONLY_POSTGRES_USER_PASSWORD']} "
            f"dbname={os.environ['POSTGRES_DB']}"
        ),
        min_size=1,
        max_size=10,
        open=False,
    )
    await pool.open()
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(
    title="govgis-nov2023-slim-spatial-server",
    description="A REST API for govgis_nov2023-slim-spatial.",
    version=__version__,
    lifespan=lifespan,
)


async def get_connection():
    """FastAPI dependency yielding a pooled async psycopg connection."""
    assert pool is not None, "Connection pool not initialized"  # nosec
    async with pool.connection() as conn:  # connection returned to pool automatically
        yield conn


_embedding_model = HuggingFaceEmbeddings(
    cache_folder=os.environ["EMBEDDING_MODEL_CACHE_FOLDER"],
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def get_embedding_model():
    return _embedding_model


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SemanticSearchRequest,
    conn=Depends(get_connection),
    embedding_model=Depends(get_embedding_model),
) -> SearchResponse:
    query, params = request.build_query(embedding_model)
    print("Executing semantic search query with", len(params), "parameters")  # debug
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
    except Exception as e:  # broad catch to surface error in response model
        print("Search error:", e)
        return SearchResponse(layers=None, error=str(e))
    return SearchResponse(
        layers=[LayerResult.model_validate(dict(r)) for r in rows],
        error=None,
    )
