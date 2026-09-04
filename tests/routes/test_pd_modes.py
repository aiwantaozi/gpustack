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
    assert ascend.gpu_filters.vendor == ["ascend"]
    # The direction the old `runtime` field could not express: a recipe that
    # only fits NVIDIA. `runtime: None` meant both "unconstrained" and "not
    # declared", so these three were offered on every accelerator.
    assert nixl.gpu_filters.vendor == ["nvidia"]
    # `custom` injects nothing and must stay selectable everywhere.
    custom = modes[PDModeEnum.CUSTOM.value]
    assert custom.gpu_filters is None or not custom.gpu_filters.vendor
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
    assert paths == {
        "/v2/pd-modes",
        "/v2/pd-modes/resolve",
        "/v2/pd-modes/{name}",
    }


def test_resolve_is_matched_before_the_name_parameter():
    """FastAPI matches routes in declaration order, so `/{name}` would swallow
    `/resolve` and turn the derived answer into a 404 for a mode named
    "resolve"."""
    from gpustack.routes import routes

    ordered = [
        route.path
        for route in routes.api_router.routes
        if getattr(route, "path", "").startswith("/v2/pd-modes")
    ]
    assert ordered.index("/v2/pd-modes/resolve") < ordered.index("/v2/pd-modes/{name}")


def test_the_classified_router_parts_survive_the_endpoint():
    """The form renders the editor from these, so a field lost in the response
    is a field the user cannot see — and `connection_args` doubly so, because
    the read-only half of the editor is what tells them which flags are not
    theirs."""
    from gpustack.schemas.pd_modes import PDMode, PDRouterProtocolEnum
    from gpustack.server.pd_mode_catalog import get_pd_modes

    for mode in get_pd_modes():
        router = mode.router
        if router is None or router.protocol == PDRouterProtocolEnum.USER_PROVIDED:
            continue
        served = PDMode.model_validate(mode.model_dump()).router
        assert served.entrypoint == router.entrypoint, mode.name
        assert served.connection_args == router.connection_args, mode.name
        assert [a.flag for a in served.tunable_args] == [
            a.flag for a in router.tunable_args
        ], mode.name
        # Options and bounds drive the select and the number field.
        assert [a.options for a in served.tunable_args] == [
            a.options for a in router.tunable_args
        ], mode.name
