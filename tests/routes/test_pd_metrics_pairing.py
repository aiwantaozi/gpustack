"""The one figure on the PD panel that is placement, not measurement.

`pairing_locality` answers "can a request's KV stay off the network", and it
is knowable the moment the members are bound -- no traffic, no Prometheus. So
it has to survive the case where everything else on the response is missing,
which is exactly when a reader most needs it: a group whose monitoring is
unreachable can still be told that none of its pairs are local, and on a link
without RDMA that is the difference between PD helping and PD being strictly
worse than not disaggregating.

It also has to be the SAME number as the `pairing_remote` degradation on the
model row. Two implementations of one quantity drift, and a marker that
disagrees with the figure beside it is worse than either being absent -- hence
one server-side function, called from both.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.routes import models as route
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelInstanceStateEnum,
    PDModeEnum,
    RoleSpec,
)
from gpustack.server.pd_metrics import PDMetricsPublic


def _model() -> Model:
    return Model(
        name="llama-pd",
        cluster_id=1,
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )


def _instance(role, worker_id):
    return SimpleNamespace(
        role=role,
        worker_id=worker_id,
        state=ModelInstanceStateEnum.RUNNING,
    )


async def _call(instances, collected: PDMetricsPublic) -> PDMetricsPublic:
    model = _model()
    with (
        patch.object(route, "_get_model", AsyncMock(return_value=model)),
        patch.object(route, "get_pd_mode", lambda _name: None),
        patch.object(route, "collect_pd_metrics", AsyncMock(return_value=collected)),
        patch.object(
            route.ModelInstance, "all_by_field", AsyncMock(return_value=instances)
        ),
    ):
        return await route.get_model_pd_metrics(session=None, ctx=None, id=1)


@pytest.mark.asyncio
async def test_locality_is_filled_from_where_the_members_landed():
    """2P2D, one pair per host: the router's two independent picks meet half
    the time. Not a fault -- the arithmetic of `m == x`."""
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 2),
        _instance("decode", 1),
        _instance("decode", 2),
    ]
    result = await _call(instances, PDMetricsPublic(available=True))
    assert result.pairing_locality == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_packing_the_same_group_tighter_reports_more():
    """Same replica counts, same roles -- only the host count changed.

    This is the whole reason the figure exists beside the deploy form's
    estimate: the form knows the replica count and cannot know this.
    """
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 1),
        _instance("decode", 1),
        _instance("decode", 1),
    ]
    result = await _call(instances, PDMetricsPublic(available=True))
    assert result.pairing_locality == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_it_survives_an_unavailable_measurement():
    """🔴 The case it exists for.

    Prometheus unreachable, so every measured field is empty -- and the one
    field that needs no measurement must still be there. Reporting nothing
    here would lose a correctness signal along with the graphs.
    """
    instances = [_instance("prefill", 1), _instance("decode", 2)]
    result = await _call(
        instances,
        PDMetricsPublic(available=False, reason="prometheus unreachable"),
    )
    assert result.available is False
    assert result.pairing_locality == 0.0


@pytest.mark.asyncio
async def test_zero_is_reported_as_zero_and_not_as_absent():
    """Prefill on one host, decode on another: every transfer crosses the
    network. The loudest reading this figure has, so it must not be dropped
    for looking falsy."""
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 1),
        _instance("decode", 2),
        _instance("decode", 2),
    ]
    result = await _call(instances, PDMetricsPublic(available=True))
    assert result.pairing_locality == 0.0
    assert result.pairing_locality is not None


@pytest.mark.asyncio
async def test_silence_rather_than_zero_before_both_roles_are_running():
    """A group with no decode yet has no pairing to report. Zero there would
    read as the alarm above rather than as "not placed yet"."""
    result = await _call([_instance("prefill", 1)], PDMetricsPublic(available=True))
    assert result.pairing_locality is None
