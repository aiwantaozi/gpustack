"""Port assignment on the worker: named connector bands, fencing, refill.

The whole point of these is that a port handed out twice does not fail
cleanly — the second engine crash-loops on bind and the instance sits in
`starting` forever — so every path that can leak a port back into the free
pool is pinned here.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpustack.schemas.models import (
    BackendEnum,
    DisaggregationSpec,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    PDModeEnum,
    PortBand,
)
from gpustack.schemas.pd_modes import (
    PDInjectTargetEnum,
    PDMode,
    PDModeRole,
    PDPortSpec,
)
from gpustack.utils import network
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model, new_model_instance

PORT_RANGE = "40000-40063"


def _manager(port_range: str = PORT_RANGE, worker_id: int = 1) -> ServeManager:
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    cfg = SimpleNamespace(
        log_dir="/tmp",
        service_port_range=port_range,
        system_default_container_registry=None,
    )
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager


def _pd_model(mode: PDModeEnum = PDModeEnum.VLLM_NIXL):
    return new_model(
        1,
        "pd-model",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        disaggregation=DisaggregationSpec(mode=mode),
    )


def _instance(instance_id: int = 1, role: str = "prefill"):
    mi = new_model_instance(instance_id, f"pd-{instance_id}", 1, worker_id=1)
    mi.worker_ip = "127.0.0.1"
    mi.role = role
    return mi


@pytest.fixture(autouse=True)
def all_ports_free(monkeypatch):
    """The allocator's scan, not the host's port table, is what's under test."""
    monkeypatch.setattr(network, "is_port_available", lambda port, host=None: True)


def _wide_band_mode(count: int = 4) -> PDMode:
    """A mode whose prefill role wants a band wider than one port."""
    return PDMode(
        name="test-wide-band",
        roles={
            "prefill": PDModeRole(
                ports=[
                    PDPortSpec(
                        name="kv_port",
                        count=count,
                        inject_to=PDInjectTargetEnum.ARGS,
                    )
                ],
                connector={"kv_port": "{{ports.kv_port}}"},
            )
        },
    )


def test_named_band_lands_in_both_indexes():
    """`named_ports` is the new index; `mi.ports` is the one the runtime turns
    into hostPorts, and it is the only reason a same-host collision shows up
    as a scheduling event instead of a crash loop."""
    manager = _manager()
    mi = _instance(role="prefill")

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    band = mi.named_ports["kv_side_channel"]
    assert band.count == 1
    assert mi.ports[0] == mi.port
    assert band.base in mi.ports
    assert band.base != mi.port


def test_router_role_reads_the_router_declaration():
    """A router's bands sit on `mode.router`, not in `mode.roles`."""
    manager = _manager()
    mi = _instance(role="router")

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert set(mi.named_ports) == {"prometheus"}
    assert mi.named_ports["prometheus"].base in mi.ports


def test_whole_band_is_fenced_not_just_its_base():
    """Every port a connector derives from the base is bound just as surely
    as the base, so the fence has to cover the run."""
    manager = _manager()
    mi = _instance(1)

    with patch(
        "gpustack.worker.serve_manager.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    band = mi.named_ports["kv_port"]
    expected = list(range(band.base, band.base + 4))
    assert band.count == 4
    assert [p for p in mi.ports if p != mi.port] == expected
    assert set(expected) <= manager._assigned_ports[mi.id]


def test_a_second_instance_gets_a_disjoint_band():
    manager = _manager()
    first, second = _instance(1), _instance(2)

    with patch(
        "gpustack.worker.serve_manager.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(first, _pd_model(), BackendEnum.VLLM)
        manager._assign_ports(second, _pd_model(), BackendEnum.VLLM)

    assert not set(first.ports) & set(second.ports)


def test_named_bands_go_before_the_connecting_port():
    """The distributed backends read the connecting port as `ports[-1]`
    (VLLM_DP_MASTER_PORT / VLLM_PORT), so "append at the tail" must not
    quietly redefine which port that is."""
    manager = _manager()
    mi = _instance()
    mi.distributed_servers = DistributedServers(
        subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=2)]
    )

    with patch(
        "gpustack.worker.serve_manager.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    band = mi.named_ports["kv_port"]
    named = set(range(band.base, band.base + band.count))
    assert mi.ports[-1] not in named
    assert mi.ports[0] == mi.port
    # And the band is still there, just not last.
    assert named <= set(mi.ports)


def test_a_non_pd_instance_takes_the_old_path_unchanged():
    manager = _manager()
    mi = new_model_instance(1, "plain", 1, worker_id=1)
    mi.worker_ip = "127.0.0.1"
    model = new_model(1, "plain", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")

    manager._assign_ports(mi, model, BackendEnum.VLLM)

    assert mi.ports == [mi.port]
    assert not mi.named_ports


def test_a_role_the_mode_does_not_declare_allocates_nothing():
    manager = _manager()
    mi = _instance(role="decode")

    with patch(
        "gpustack.worker.serve_manager.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert mi.ports == [mi.port]
    assert not mi.named_ports


def test_early_return_refills_the_registry_with_the_whole_band():
    """A restart reuses the persisted ports and used to return without
    re-registering them, so this process saw the entire band as free and
    handed it to the next instance."""
    manager = _manager()
    mi = _instance()
    mi.port = 40000
    mi.ports = [40000, 40010, 40011, 40012, 40013]
    mi.named_ports = {"kv_port": PortBand(base=40010, count=4)}

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert manager._assigned_ports[mi.id] == {40000, 40010, 40011, 40012, 40013}
    # Nothing was reallocated.
    assert mi.port == 40000


def test_early_return_expands_a_band_absent_from_mi_ports():
    """Bands persisted before the append-to-`mi.ports` rule existed are still
    fenced: `count` is what defines the run, not the list."""
    manager = _manager()
    mi = _instance()
    mi.port = 40000
    mi.ports = [40000]
    mi.named_ports = {"kv_port": PortBand(base=40020, count=3)}

    manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert manager._assigned_ports[mi.id] == {40000, 40020, 40021, 40022}


def test_refilled_ports_are_honoured_by_the_next_allocation():
    manager = _manager()
    restarted = _instance(1)
    restarted.port = 40000
    restarted.ports = [40000]
    restarted.named_ports = {"kv_port": PortBand(base=40001, count=4)}
    manager._assign_ports(restarted, _pd_model(), BackendEnum.VLLM)

    fresh = _instance(2)
    with patch(
        "gpustack.worker.serve_manager.get_pd_mode", return_value=_wide_band_mode(4)
    ):
        manager._assign_ports(fresh, _pd_model(), BackendEnum.VLLM)

    assert not set(fresh.ports) & {40000, 40001, 40002, 40003, 40004}


def test_templated_count_is_narrowed_to_one_and_says_so(caplog):
    """Phase one has no resolver for `{{tensor_parallel_size}}`. Reserving one
    port where the connector will bind eight is exactly the collision this
    mechanism exists to prevent, so it is never silent."""
    manager = _manager()
    mi = _instance(role="prefill")

    with caplog.at_level(logging.WARNING, logger="gpustack.worker.serve_manager"):
        manager._assign_ports(
            mi, _pd_model(PDModeEnum.VLLM_ASCEND_MOONCAKE), BackendEnum.VLLM
        )

    assert mi.named_ports["kv_port"].count == 1
    assert "templated count" in caplog.text
    assert "tensor_parallel_size" in caplog.text


def test_unknown_mode_warns_and_assigns_no_bands(caplog):
    manager = _manager()
    mi = _instance()

    with caplog.at_level(logging.WARNING, logger="gpustack.worker.serve_manager"):
        with patch("gpustack.worker.serve_manager.get_pd_mode", return_value=None):
            manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    assert not mi.named_ports
    assert "not in the catalog" in caplog.text


def test_exhaustion_names_the_role_and_the_band():
    """The allocator knows the arithmetic but not who was asking, and "which
    role of which deployment" is the first thing an operator needs."""
    manager = _manager(port_range="40000-40000")
    mi = _instance(role="prefill")

    with pytest.raises(network.PortRangeExhaustedError) as excinfo:
        with patch(
            "gpustack.worker.serve_manager.get_pd_mode", return_value=_wide_band_mode(4)
        ):
            manager._assign_ports(mi, _pd_model(), BackendEnum.VLLM)

    message = str(excinfo.value)
    assert "role 'prefill'" in message
    assert "'kv_port'" in message
    assert "Widen the port range" in message


def test_start_model_instance_persists_named_ports(tmp_path):
    """The bands are allocated on the worker and read on the server (the
    router's peer config), so they persist the way `port`/`ports` do."""
    manager = _manager()
    manager._serve_log_dir = str(tmp_path)
    mi = _instance()
    model = _pd_model()

    with (
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_start_container_log_persistence"),
        patch.object(manager, "_update_model_instance") as update,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
    ):
        process.return_value.pid = 4242
        manager._start_model_instance(mi)

    patch_dict = update.call_args.kwargs
    assert patch_dict["named_ports"] == mi.named_ports
    assert patch_dict["named_ports"]["kv_side_channel"].base in patch_dict["ports"]
