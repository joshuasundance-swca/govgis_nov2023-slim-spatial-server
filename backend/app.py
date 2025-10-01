import os
from contextlib import asynccontextmanager
import re
import logging
import asyncio
import time

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


def configure_logging() -> None:
    """Configure root logging once using LOG_LEVEL env (default INFO)."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:  # basicConfig is no-op if already configured
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)
    logger.info("Logging configured level=%s", level_name)


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


async def _warm_pool(target: int) -> None:
    """Eagerly create and validate connections up to *target*.

    This issues lightweight 'SELECT 1' on each new connection so that later
    request latency isn't impacted by connection creation or initial DB auth.
    """
    if pool is None:
        return
    created = 0
    while created < target:
        async with pool.connection() as conn:  # creates new one if below max_size
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        created += 1
    logger.info("Pool warm complete: %s connections validated", created)


@asynccontextmanager
async def lifespan(app: FastAPI):  # FastAPI will call this on startup/shutdown
    global pool
    configure_logging()
    await ensure_read_only_user()

    max_size = int(os.getenv("PG_POOL_MAX_SIZE", "10"))
    configured_min = int(os.getenv("PG_POOL_MIN_SIZE", "1"))
    eager = os.getenv("PG_POOL_EAGER", "false").lower() in {"1", "true", "yes", "on"}
    min_size = max_size if eager else min(configured_min, max_size)

    warm_target_env = os.getenv("PG_POOL_WARM_TARGET")
    warm_target = 0
    if warm_target_env and warm_target_env.isdigit():
        warm_target = int(warm_target_env)

    pool_timeout = float(
        os.getenv("PG_POOL_TIMEOUT", "30"),
    )  # seconds to wait for a free conn

    pool = AsyncConnectionPool(
        conninfo=(
            f"host={os.environ['POSTGRES_HOST']} "
            f"user={os.environ['READ_ONLY_POSTGRES_USER']} "
            f"password={os.environ['READ_ONLY_POSTGRES_USER_PASSWORD']} "
            f"dbname={os.environ['POSTGRES_DB']}"
        ),
        min_size=min_size,
        max_size=max_size,
        timeout=pool_timeout,
        open=False,  # we'll open explicitly (avoid double-open)
    )
    await pool.open()
    logger.info(
        "Pool initialized min=%s max=%s eager=%s timeout=%s warm_target=%s",
        min_size,
        max_size,
        eager,
        pool_timeout,
        warm_target,
    )

    # Eager mode warms entire pool; otherwise optional partial warm target.
    if eager:
        try:
            await _warm_pool(max_size)
        except Exception as e:  # pragma: no cover
            logger.warning("Pool warm (eager) failed: %s", e)
        if warm_target:
            logger.info(
                "PG_POOL_WARM_TARGET=%s ignored because PG_POOL_EAGER is true",
                warm_target,
            )
    elif warm_target > 0:
        try:
            await _warm_pool(min(warm_target, max_size))
        except Exception as e:  # pragma: no cover
            logger.warning("Partial pool warm failed: %s", e)

    if os.getenv("EMBEDDING_WARMUP", "true").lower() in {"1", "true", "yes", "on"}:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _embedding_model.embed_query, "warmup")
            logger.info("Embedding model warm-up complete")
        except Exception as e:  # pragma: no cover
            logger.warning("Embedding warm-up skipped/failed: %s", e)

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
    start_time = time.time()
    assert pool is not None, "Connection pool not initialized"  # nosec
    async with pool.connection() as conn:  # connection returned to pool automatically
        elapsed_time = time.time() - start_time
        logger.info("Connection checkout took %.2f seconds", elapsed_time)
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
    start = time.perf_counter()
    query, params = await request.build_query(embedding_model)
    logger.debug("search start params=%d", len(params))
    try:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "search error params=%d duration_ms=%.2f error=%s",
            len(params),
            duration_ms,
            e,
        )
        return SearchResponse(layers=None, error=str(e))
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "search ok rows=%d params=%d duration_ms=%.2f",
        len(rows),
        len(params),
        duration_ms,
    )
    return SearchResponse(
        layers=[LayerResult.model_validate(dict(r)) for r in rows],
        error=None,
    )


@app.get("/health")
async def health():
    """Lightweight health check.

    Returns basic pool sizing info (without forcing any new connections) and static status.
    Safe for readiness / liveness probes and latency benchmarking.
    """
    initialized = pool is not None
    data: dict[str, object] = {
        "status": "ok",
        "version": __version__,
        "pool_initialized": initialized,
    }
    if initialized:
        try:  # Access attributes defensively
            data.update(
                {
                    "pool_min_size": pool.min_size,  # type: ignore[union-attr]
                    "pool_max_size": pool.max_size,  # type: ignore[union-attr]
                },
            )
        except Exception:  # pragma: no cover  # nosec
            pass
    return data
