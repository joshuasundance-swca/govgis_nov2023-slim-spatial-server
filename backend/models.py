from __future__ import annotations

import os
import re
from typing import Optional, Tuple, List

from pydantic import BaseModel, field_validator, model_validator, Field
from pyproj import (
    Proj,
    Transformer,
)
from sqlalchemy.sql import quoted_name  # added for safe identifier quoting

from load_data import table_columns

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
        return "[" + ",".join(format(x, "g") for x in emb) + "]"

    def embed_query(self, embedding_model) -> str:
        return self.transform_emb(embedding_model.embed_query(self.request_string))

    def build_query(self, embedding_model) -> Tuple[str, List]:
        """Build a parameterized SQL query ensuring parameter order matches placeholders.

        Placeholder order in final SQL:
          1..N : optional filter params (type array, lon, lat)
          Next : embedding vector literal for ORDER BY
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

        # Embedding param now appended AFTER filters so its placeholder is next.
        emb_literal = self.embed_query(embedding_model)
        params.append(emb_literal)

        # LIMIT and OFFSET last.
        params.append(self.limit)
        params.append(self.skip)

        query = (
            f"SELECT {output_columns} FROM {quoted_table} "  # nosec
            f"{where_sql} "
            f'ORDER BY "embeddings" <=> %s::vector '
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
