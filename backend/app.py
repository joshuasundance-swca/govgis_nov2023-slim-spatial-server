import os

import asyncpg
from asyncpg.connection import Connection
from fastapi import FastAPI, Depends
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from models import SemanticSearchRequest, SearchResponse, LayerResult  # type: ignore[attr-defined]

__version__ = "0.0.1"

app = FastAPI(
    title="govgis-nov2023-slim-spatial-server",
    description="A REST API for govgis_nov2023-slim-spatial.",
    version=__version__,
)


async def get_connection() -> Connection:
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        user=os.environ["READ_ONLY_POSTGRES_USER"],
        password=os.environ["READ_ONLY_POSTGRES_USER_PASSWORD"],
        database=os.environ["POSTGRES_DB"],
    )


_embedding_model = HuggingFaceBgeEmbeddings(
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
    conn: Connection = Depends(get_connection),
    embedding_model=Depends(get_embedding_model),
) -> SearchResponse:
    query, params = request.build_query(embedding_model)
    # Debug print (can be switched to logging). Avoid printing sensitive params directly.
    print("Executing semantic search query with", len(params), "parameters")
    try:
        layers = await conn.fetch(query, *params)
    except Exception as e:
        print("Search error:", e)
        return SearchResponse(layers=None, error=str(e))
    return SearchResponse(
        layers=[LayerResult.model_validate(dict(record)) for record in layers],
        error=None,
    )
