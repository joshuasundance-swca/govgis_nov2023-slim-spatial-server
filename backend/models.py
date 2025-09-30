from __future__ import annotations

import os
from typing import Optional

# from pyproj.exceptions import CRSError
# from shapely.errors import GEOSException
# from shapely.geometry import Polygon
from pydantic import BaseModel, field_validator, model_validator, Field
from pyproj import (
    #    CRS,
    Proj,
    Transformer,
)

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


SemanticSearchSQL = """SELECT {output_columns} FROM {table}
{filter}
ORDER BY embeddings <=> '{embeddings}'
LIMIT {limit}
"""


class SemanticSearchRequest(BaseModel):
    request_string: str = Field(
        description="A string to search for in the database.",
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
        embstr = ",".join(map(str, emb))
        return f"[{embstr}]"

    def embed_query(self, embedding_model) -> str:
        return self.transform_emb(embedding_model.embed_query(self.request_string))

    def build_geometry_filter(self) -> str:
        if self.input_point is None:
            return ""
        pt = f"ST_MakePoint({self.input_point.longitude}, {self.input_point.latitude})"
        srid = f"ST_SetSRID({pt}, 4326)"
        intersects = f"ST_Intersects(geom, {srid})"
        return intersects

    def type_filter_query(self) -> str:
        if self.type_filter is None:
            return ""
        typestr = ",".join([f"'{t.lower()}'" for t in self.type_filter])
        return f"LOWER(type) IN ({typestr})"

    def build_query(self, embedding_model) -> str:
        _type_filter = self.type_filter_query()
        _geometry_filter = self.build_geometry_filter()
        _filter = ""
        if _type_filter:
            _filter = f"WHERE {_type_filter}"
            if _geometry_filter:
                _filter += f" AND {_geometry_filter}"
        elif _geometry_filter:
            _filter = f"WHERE {_geometry_filter}"

        return SemanticSearchSQL.format(
            output_columns=",".join(TEXT_FIELDS),
            filter=_filter,
            table=os.environ["POSTGRES_TABLE"],
            embeddings=self.embed_query(embedding_model),
            limit=self.limit,
        )


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
