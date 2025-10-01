import os
from contextlib import asynccontextmanager

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from fastapi import FastAPI, Depends
from langchain_huggingface import HuggingFaceEmbeddings
from models import SemanticSearchRequest, SearchResponse, LayerResult  # type: ignore[attr-defined]

__version__ = "0.0.1"

# Initialize a global async connection pool (opened in FastAPI lifespan)
pool: AsyncConnectionPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # FastAPI will call this on startup/shutdown
    global pool
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
