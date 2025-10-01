from __future__ import annotations

import os
import re
from typing import Optional, Tuple, List

# from pyproj.exceptions import CRSError
# from shapely.errors import GEOSException
# from shapely.geometry import Polygon
from pydantic import BaseModel, field_validator, model_validator, Field
from pyproj import (
    #    CRS,
    Proj,
    Transformer,
)
from sqlalchemy.sql import quoted_name  # added for safe identifier quoting

from load_data import table_columns

# import asyncpg
# from asyncpg.connection import Connection

MINIMUM_SEARCH_LIMIT = 1
DEFAULT_SEARCH_LIMIT = 5
MAXIMUM_SEARCH_LIMIT = 10

TEXT_FIELDS = table_columns[:-2]


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


# NOTE: kept for reference; no longer used directly with formatting + user inputs.
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

    @field_validator("input_point")
    @classmethod
    def check_input_point(cls, v: Point) -> Point:
        if not v:
            return v
        return Point4326(**v.model_dump())

    @staticmethod
    def transform_emb(emb: list[float]) -> str:
        # Produce a pgvector-compatible literal. Values are numeric so safe to join.
        return "[" + ",".join(format(x, "g") for x in emb) + "]"

    def embed_query(self, embedding_model) -> str:
        # Return the vector literal string; parameterized later and cast to vector.
        return self.transform_emb(embedding_model.embed_query(self.request_string))

    # --- Secure query builder ---
    def build_query(self, embedding_model) -> Tuple[str, List]:
        """Build a parameterized SQL query and its parameters.

        Returns
        -------
        (query, params):
            query: A SQL string with positional parameters ($1, $2, ...).
            params: List of parameter values matching placeholders.

        Security considerations:
            * All user-controlled scalar/list values are bound as parameters.
            * Identifiers (table, columns) are validated and safely quoted.
            * Embedding similarity uses a parameterized pgvector literal string.
            * Geometry and type filters fully parameterized.
        """
        # Validate and fetch table name from environment (defense-in-depth)
        try:
            table_name = os.environ["POSTGRES_TABLE"]
        except KeyError as e:
            raise RuntimeError("POSTGRES_TABLE environment variable not set") from e

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
            raise ValueError("Invalid table name (fails identifier whitelist)")

        # Validate output columns (defense-in-depth; table_columns should be trusted but verify)
        unsafe_cols = [c for c in TEXT_FIELDS if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c)]
        if unsafe_cols:
            raise ValueError(f"Invalid column names detected: {unsafe_cols}")

        # Use SQLAlchemy quoted_name to ensure safe quoting of identifiers.
        quoted_cols = [quoted_name(c, quote=True) for c in TEXT_FIELDS]
        output_columns = ",".join(f'"{c}"' for c in quoted_cols)
        quoted_table = f'"{quoted_name(table_name, quote=True)}"'

        params: List = []

        # 1. Embedding vector literal string param ($1)
        emb_literal = self.embed_query(embedding_model)
        params.append(emb_literal)

        filter_clauses = []

        # 2. Type filter (array param ANY($n))
        if self.type_filter:
            lowered_types = [t.lower() for t in self.type_filter if t]
            if lowered_types:
                params.append(lowered_types)
                # Cast param to text[] explicitly for clarity / inference
                filter_clauses.append(f'LOWER("type") = ANY(${len(params)}::text[])')

        # 3. Geometry filter (lon / lat params) (geom column quoted)
        if self.input_point is not None:
            params.append(self.input_point.longitude)
            lon_idx = len(params)
            params.append(self.input_point.latitude)
            lat_idx = len(params)
            filter_clauses.append(
                f'ST_Intersects("geom", ST_SetSRID(ST_MakePoint(${lon_idx}, ${lat_idx}), 4326))'
            )

        where_sql = ""
        if filter_clauses:
            where_sql = "WHERE " + " AND ".join(filter_clauses)

        # 4. LIMIT + 5. OFFSET
        params.append(self.limit)
        limit_idx = len(params)
        params.append(self.skip)
        offset_idx = len(params)

        query = (
            f"SELECT {output_columns} FROM {quoted_table} "
            f"{where_sql} "
            f'ORDER BY "embeddings" <=> $1::vector '
            f"LIMIT ${limit_idx} OFFSET ${offset_idx}"
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
