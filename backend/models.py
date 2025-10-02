from __future__ import annotations
import time
import os
import re
from typing import Optional, Tuple, List
from functools import lru_cache

from pydantic import BaseModel, field_validator, model_validator, Field
from pyproj import (
    Proj,
    Transformer,
)
from sqlalchemy.sql import quoted_name  # added for safe identifier quoting

from load_data import table_columns
import logging

logger = logging.getLogger(__name__)
MINIMUM_SEARCH_LIMIT = 1
DEFAULT_SEARCH_LIMIT = 5
MAXIMUM_SEARCH_LIMIT = 10

TEXT_FIELDS = table_columns[:-2]

# Embedding cache size (configurable via env)
_EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "256"))


class Point(BaseModel):
    """A point with longitude, latitude, and epsg."""

    longitude: float = Field(description="The longitude of the point.")
    latitude: float = Field(description="The latitude of the point.")
    epsg: int = Field(4326, description="The EPSG code of the point.")

    def reproject(self, dst_epsg: int) -> Point:
        src_proj = Proj(f"EPSG:{self.epsg}")
        dst_proj = Proj(f"EPSG:{dst_epsg}")
        transformer = Transformer.from_proj(src_proj, dst_proj, always_xy=True)
        x, y = transformer.transform(self.longitude, self.latitude)
        return Point(longitude=x, latitude=y, epsg=dst_epsg)


class Point4326(Point):
    """A point with longitude and latitude coordinates in EPSG 4326."""

    @model_validator(mode="after")
    def check_epsg(self) -> Point:
        return self if self.epsg == 4326 else self.reproject(4326)

    @field_validator("longitude", mode="after")
    @classmethod
    def check_longitude_range(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return v

    @field_validator("latitude", mode="after")
    @classmethod
    def check_latitude_range(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("latitude must be between -90 and 90")
        return v


SemanticSearchSQL = """SELECT {output_columns} FROM {table}
{filter}
ORDER BY embeddings <=> '{embeddings}'
LIMIT {limit}
"""


class SemanticSearchRequest(BaseModel):
    """A request for hybrid semantic and spatial search."""

    request_string: str = Field(
        description="A semantic search query.",
    )
    type_filter: Optional[list[str]] = Field(
        None,
        description="Filter by the type of the layer.",
    )
    input_point: Optional[Point] = Field(
        None,
        description="Filter results to those that intersect this point.",
    )
    skip: int = Field(default=0, description="The number of documents to skip", ge=0)
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        description="The maximum number of documents to return",
        ge=MINIMUM_SEARCH_LIMIT,
        le=MAXIMUM_SEARCH_LIMIT,
    )
    timeout: int | None = Field(
        60,
        description="The timeout for the request in seconds",
    )

    @field_validator("input_point")
    @classmethod
    def check_input_point(cls, v: Point) -> Point:
        if not v:
            return v
        return Point4326(**v.model_dump())

    @staticmethod
    def transform_emb(emb: list[float]) -> str:
        return "[" + ",".join(format(x, "g") for x in emb) + "]"

    @staticmethod
    @lru_cache(maxsize=_EMBED_CACHE_SIZE)
    def _cached_embed_sync(request_string: str, model_name: str) -> str:
        """Sync wrapper for caching. Not called directly - used by embed_query."""
        # This is a placeholder that should never be called directly
        # The actual embedding is done in embed_query
        raise NotImplementedError("Use embed_query instead")

    async def embed_query(self, embedding_model) -> list[float]:
        """Generate embeddings for the query with LRU caching.
        
        Returns the raw list[float] for native pgvector adapter support.
        Falls back to string literal if adapter is not available.
        """
        # Create cache key from request string
        # We normalize the string to improve cache hits
        cache_key = self.request_string.strip().lower()
        
        # Try to use a simple dict cache (thread-safe for async single-threaded event loop)
        if not hasattr(self.__class__, '_embed_cache'):
            self.__class__._embed_cache = {}
        
        if cache_key in self.__class__._embed_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Embedding cache hit for query")
            return self.__class__._embed_cache[cache_key]
        
        # Cache miss - generate embedding
        start_time = time.time()
        _emb = await embedding_model.aembed_query(self.request_string)
        elapsed_time = time.time() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Embedding generation took {elapsed_time:.2f} seconds")
        
        # Store in cache (with size limit)
        if len(self.__class__._embed_cache) >= _EMBED_CACHE_SIZE:
            # Simple eviction: remove first item (FIFO-ish)
            self.__class__._embed_cache.pop(next(iter(self.__class__._embed_cache)))
        self.__class__._embed_cache[cache_key] = _emb
        
        return _emb

    async def build_query(self, embedding_model) -> Tuple[str, List]:
        """Build a parameterized SQL query ensuring parameter order matches placeholders.

        Placeholder order in final SQL:
          1..N : optional filter params (type array, lon, lat)
          Next : embedding vector (as list[float] for native adapter, or string for fallback)
          Next : LIMIT
          Next : OFFSET
        """
        try:
            table_name = os.environ["POSTGRES_TABLE"]
        except KeyError as e:
            raise RuntimeError("POSTGRES_TABLE environment variable not set") from e

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
            raise ValueError("Invalid table name (fails identifier whitelist)")

        unsafe_cols = [
            c for c in TEXT_FIELDS if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c)
        ]
        if unsafe_cols:
            raise ValueError(f"Invalid column names detected: {unsafe_cols}")

        quoted_cols = [quoted_name(c, quote=True) for c in TEXT_FIELDS]
        output_columns = ",".join(f'"{c}"' for c in quoted_cols)
        quoted_table = f'"{quoted_name(table_name, quote=True)}"'

        params: List = []
        filter_clauses: List[str] = []

        # Collect filter params first so their placeholders appear first.
        if self.type_filter:
            lowered_types = [t.lower() for t in self.type_filter if t]
            if lowered_types:
                params.append(lowered_types)
                filter_clauses.append('LOWER("type") = ANY(%s)')

        if self.input_point is not None:
            params.append(self.input_point.longitude)
            params.append(self.input_point.latitude)
            filter_clauses.append(
                'ST_Intersects("geom", ST_SetSRID(ST_MakePoint(%s, %s), 4326))',
            )

        where_sql = ""
        if filter_clauses:
            where_sql = "WHERE " + " AND ".join(filter_clauses)

        # Get embedding as list[float] for native pgvector adapter
        emb_vector = await self.embed_query(embedding_model)
        
        # Check if pgvector adapter is available (imported in app.py)
        # If available, pass as list[float]; otherwise convert to string literal
        try:
            from app import PGVECTOR_ADAPTER_AVAILABLE
            if PGVECTOR_ADAPTER_AVAILABLE:
                params.append(emb_vector)  # Native list[float]
            else:
                params.append(self.transform_emb(emb_vector))  # String literal fallback
        except ImportError:
            # Fallback if we can't import from app (e.g., in tests)
            params.append(self.transform_emb(emb_vector))

        # LIMIT and OFFSET last.
        params.append(self.limit)
        params.append(self.skip)

        # Build query without ::vector cast (adapter handles conversion)
        query = (
            f"SELECT {output_columns} FROM {quoted_table} "  # nosec
            f"{where_sql} "
            f'ORDER BY "embeddings" <=> %s '
            f"LIMIT %s OFFSET %s"
        )

        return query, params


class LayerResult(BaseModel):
    id: str = Field(
        description="The unique identifier for the layer.",
    )
    name: str = Field(description="The name of the layer.")
    type: str = Field(description="The type of the layer.")
    description: str = Field(description="A description of the layer.")
    url: str = Field(description="The URL of the layer.")
    metadata_text: str = Field(description="The metadata text of the layer.")


class SearchResponse(BaseModel):
    layers: Optional[list[LayerResult]] = Field(
        description="The layers that match the search request.",
    )
    error: Optional[str] = Field(
        None,
        description="An error message if the search request failed.",
    )
