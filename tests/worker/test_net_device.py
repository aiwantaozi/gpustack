import pytest

from gpustack.config.config import Config
from gpustack.schemas.workers import (
    GPUDeviceStatus,
    GPUNetworkInfo,
    Worker,
    WorkerStatus,
)
from gpustack.worker.net_device import derive_net_device


@pytest.fixture
def config(tmp_path):
    def _config(**kwargs) -> Config:
        return Config(data_dir=str(tmp_path / "data"), **kwargs)

    return _config


def _worker(ifname: str = "eth0", gpu_devices=None) -> Worker:
    status = WorkerStatus.get_default_status()
    if gpu_devices is not None:
        status.gpu_devices = gpu_devices
    return Worker(
        id=1,
        name="test-worker",
        hostname="test-host",
        ip="192.168.1.100",
        ifname=ifname,
        port=10150,
        worker_uuid="test-uuid",
        cluster_id=1,
        status=status,
    )


def test_kv_ifname_overrides_worker_ifname(config):
    """The escape hatch exists precisely because the auto-detected NIC is the
    management one; if it did not win, setting it would be a no-op."""
    assert derive_net_device(_worker(ifname="eth0"), config(kv_ifname="ib0")) == "ib0"


def test_falls_back_to_worker_ifname(config):
    assert derive_net_device(_worker(ifname="bond0"), config()) == "bond0"


def test_returns_none_when_no_source_has_a_value(config):
    """No source means no answer. Anything else -- notably ``all`` -- turns a
    clean failure into an unroutable address inside the NIXL metadata."""
    assert derive_net_device(_worker(ifname=""), config()) is None


def test_placeholder_worker_row_is_not_a_value(config):
    """Pool-provisioned workers are stored with ifname="" long before they
    report in, so emptiness has to read as unknown, not as an interface name."""
    assert derive_net_device(_worker(ifname="   "), config()) is None


def test_blank_kv_ifname_defers_instead_of_blanking_the_result(config):
    assert derive_net_device(_worker(ifname="eth1"), config(kv_ifname="  ")) == "eth1"


def test_ascend_per_card_iface_is_never_used(config):
    """hccn_tool reports eth0-eth7 for the card-internal ports. Those devices do
    not exist in the host netns, so UCX cannot bind to them -- the host NIC is
    still the only usable answer."""
    ascend_cards = [
        GPUDeviceStatus(
            vendor="Huawei",
            type="cann",
            index=i,
            name="Ascend 910B2",
            network=GPUNetworkInfo(
                status="up",
                inet=f"10.10.0.{i + 1}",
                iface=f"eth{i}",
                mtu=8192,
            ),
        )
        for i in range(8)
    ]
    worker = _worker(ifname="enp1s0f0", gpu_devices=ascend_cards)

    assert derive_net_device(worker, config()) == "enp1s0f0"


def test_ascend_per_card_iface_does_not_rescue_a_missing_worker_ifname(config):
    """The per-card data being present must not make the None case disappear --
    that would smuggle a card identifier in as a UCX device name."""
    worker = _worker(
        ifname="",
        gpu_devices=[
            GPUDeviceStatus(
                vendor="Huawei",
                type="cann",
                index=0,
                name="Ascend 910B2",
                network=GPUNetworkInfo(status="up", inet="10.10.0.1", iface="eth0"),
            )
        ],
    )

    assert derive_net_device(worker, config()) is None


def test_kv_ifname_defaults_to_none(config):
    assert config().kv_ifname is None
