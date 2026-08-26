"""PD-mode catalog route checks.

The handlers are driven directly, like the other read-only catalog route
tests: the endpoint is the deployment form's single dropdown, so what
matters is that it serves every declared mode with its connection state
intact.
"""

import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.routes.pd_modes import get_pd_mode_by_name, list_pd_modes
from gpustack.schemas.common import ListParams
from gpustack.schemas.models import PDModeEnum
from gpustack.server.pd_mode_catalog import load_pd_modes


def params(page=1, perPage=100):
    return ListParams(page=page, perPage=perPage, watch=False, sort_by=None)


@pytest.mark.asyncio
async def test_list_returns_the_whole_catalog():
    result = await list_pd_modes(params=params(), search=None)
    assert [item.name for item in result.items] == [
        mode.name for mode in load_pd_modes()
    ]
    assert {item.name for item in result.items} == {mode.value for mode in PDModeEnum}
    assert result.pagination.total == len(result.items)


@pytest.mark.asyncio
async def test_list_serves_the_connection_state_the_ui_never_asks_for():
    result = await list_pd_modes(params=params(), search=None)
    modes = {item.name: item for item in result.items}

    nixl = modes[PDModeEnum.VLLM_NIXL.value]
    assert nixl.display_name == "vLLM + NIXL"
    assert nixl.backends == ["vLLM"]
    assert nixl.role("prefill").connector["kv_connector"] == "NixlConnector"
    assert nixl.router.capabilities.metrics is True
    # The window is resolved onto the mode, not left as a reference the
    # client would have to dereference.
    assert nixl.kv_lease.param == "kv_lease_duration"
    assert nixl.kv_lease.engine_default == 30

    ascend = modes[PDModeEnum.VLLM_ASCEND_MOONCAKE.value]
    assert ascend.runtime == "ascend"
    # True since the Ascend recipe swapped its example-proxy router for
    # vllm-router, verified on 910B2 through a real prefill-decode pair.
    assert ascend.router.capabilities.metrics is True
    assert ascend.kv_lease.engine_default == 480


@pytest.mark.asyncio
async def test_list_search_and_pagination():
    matched = await list_pd_modes(params=params(), search="  SGLang ")
    assert {item.name for item in matched.items} == {
        PDModeEnum.SGLANG_MOONCAKE.value,
        PDModeEnum.SGLANG_NIXL.value,
    }

    first_page = await list_pd_modes(params=params(page=1, perPage=2), search=None)
    assert len(first_page.items) == 2
    assert first_page.pagination.total == len(load_pd_modes())
    assert first_page.pagination.totalPage == 3

    unpaged = await list_pd_modes(params=params(page=0, perPage=0), search=None)
    assert len(unpaged.items) == len(load_pd_modes())
    assert unpaged.pagination.totalPage == 1


@pytest.mark.asyncio
async def test_get_by_name():
    mode = await get_pd_mode_by_name(name="VLLM-NIXL")
    assert mode.name == PDModeEnum.VLLM_NIXL.value
    with pytest.raises(NotFoundException):
        await get_pd_mode_by_name(name="ascend-mooncake")


def test_route_is_registered_under_the_versioned_prefix():
    """The dropdown is fetched by every user who can deploy a model, so the
    route sits on the same read-only catalog surface as cache providers."""
    from gpustack.routes import routes

    paths = {
        route.path
        for route in routes.api_router.routes
        if getattr(route, "path", "").startswith("/v2/pd-modes")
    }
    assert paths == {"/v2/pd-modes", "/v2/pd-modes/{name}"}
