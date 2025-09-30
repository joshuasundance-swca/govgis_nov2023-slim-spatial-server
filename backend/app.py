import os

import asyncpg
from asyncpg.connection import Connection
from fastapi import FastAPI, Depends
from langchain.embeddings import HuggingFaceBgeEmbeddings

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


# async def stream_records(cursor: AsyncIOMotorCursor) -> AsyncGenerator[str, None]:
#     yield "["
#     first_item = True
#     async for document in cursor:
#         if not first_item:
#             yield ","
#         yield Layer.model_validate(document).model_dump_json()
#         first_item = False
#     yield "]"


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SemanticSearchRequest,
    conn: Connection = Depends(get_connection),
    embedding_model=Depends(get_embedding_model),
) -> SearchResponse:
    query_str = request.build_query(embedding_model)
    print(query_str)
    try:
        layers = await conn.fetch(query_str)
    except Exception as e:
        print(e)
        return SearchResponse(layers=None, error=str(e))
    return SearchResponse(
        layers=[LayerResult.model_validate(dict(record)) for record in layers],
        error=None,
    )
