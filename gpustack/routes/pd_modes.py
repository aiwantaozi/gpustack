import math
from typing import List, Optional

from fastapi import APIRouter

from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.pd_modes import PDMode
from gpustack.server.pd_mode_catalog import get_pd_mode, get_pd_modes
from gpustack.server.deps import ListParamsDep

router = APIRouter()


@router.get("", response_model=PaginatedList[PDMode])
async def list_pd_modes(
    params: ListParamsDep,
    search: Optional[str] = None,
):
    """The catalog behind the deployment form's single PD-mode dropdown.
    Read-only and bundled with the release, like the cache-provider
    catalog."""
    modes: List[PDMode] = get_pd_modes()
    if search:
        search = search.strip().lower()
        modes = [
            mode
            for mode in modes
            if search in mode.name.lower()
            or (mode.display_name and search in mode.display_name.lower())
        ]

    count = len(modes)

    if params.page < 1 or params.perPage < 1:
        # Return all items.
        pagination = Pagination(
            page=1,
            perPage=count,
            total=count,
            totalPage=1,
        )
        return PaginatedList[PDMode](items=modes, pagination=pagination)

    # Paginate results.
    total_page = math.ceil(count / params.perPage)

    start_index = (params.page - 1) * params.perPage
    end_index = start_index + params.perPage

    paginated_items = modes[start_index:end_index]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[PDMode](items=paginated_items, pagination=pagination)


@router.get("/{name}", response_model=PDMode)
async def get_pd_mode_by_name(name: str):
    mode = get_pd_mode(name)
    if mode is None:
        raise NotFoundException(message=f"PD mode '{name}' not found")
    return mode
