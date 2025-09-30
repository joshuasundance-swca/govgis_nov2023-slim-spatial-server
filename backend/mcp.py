from fastmcp import FastMCP
from httpx import AsyncClient
from markdownify import markdownify as md

from models import SemanticSearchRequest, SearchResponse  # type: ignore[attr-defined]

mcp = FastMCP(name="govgis_nov2023")

client = AsyncClient(base_url="http://backend:8080")


def markdownify_all_strings(obj):
    if isinstance(obj, str):
        return md(obj)
    elif isinstance(obj, list):
        return [markdownify_all_strings(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: markdownify_all_strings(value) for key, value in obj.items()}
    else:
        return obj


@mcp.tool
async def gis_layer_search(request: SemanticSearchRequest) -> SearchResponse:
    response = await client.post("/search", json=request.model_dump())
    response.raise_for_status()
    response_dict = response.json()
    response_dict_md = markdownify_all_strings(response_dict)
    return SearchResponse.model_validate(response_dict_md)
