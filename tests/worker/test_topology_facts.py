"""From per-device hints to one host's position."""

from types import SimpleNamespace

from gpustack.detectors.runtime.runtime import _topology_hints
from gpustack.worker.topology_facts import (
    ACCELERATOR_DOMAIN_KEY,
    NVIDIA_CLIQUE_KEY,
    SWITCH_KEY,
    SWITCH_NAME_KEY,
    HostSwitchProbe,
    facts_from_devices,
    merge_facts,
)


def dev(**hints):
    return SimpleNamespace(topology_hints=hints or None)


# --- runtime appendix -> device hints --------------------------------------- #


def test_a_multi_node_fabric_becomes_a_clique():
    hints = _topology_hints({"fabric_cluster_uuid": "aaaa-bbbb", "fabric_clique_id": 7})
    assert hints == {NVIDIA_CLIQUE_KEY: "aaaa-bbbb.7"}


def test_a_single_host_hgx_reports_no_domain():
    """🔴 H100/H200 with NVSwitch report the fabric as COMPLETED with an all-zero
    UUID and clique 0; passing it through would put every HGX in the fleet into
    one shared domain."""
    hints = _topology_hints(
        {
            "fabric_cluster_uuid": "00000000-0000-0000-0000-000000000000",
            "fabric_clique_id": 0,
        }
    )
    assert NVIDIA_CLIQUE_KEY not in hints


def test_a_super_pod_and_a_switch_become_facts():
    hints = _topology_hints(
        {
            "super_pod_id": 3,
            "roce_lldp_chassis_id": "c0:f9:b0:c7:13:71",
            "roce_lldp_system_name": "CE8875-50",
        }
    )
    assert hints == {
        ACCELERATOR_DOMAIN_KEY: "spod-3",
        SWITCH_KEY: "c0:f9:b0:c7:13:71",
        SWITCH_NAME_KEY: "CE8875-50",
    }


# --- device hints -> host facts --------------------------------------------- #


def test_eight_cards_agreeing_is_one_domain():
    facts = facts_from_devices([dev(**{ACCELERATOR_DOMAIN_KEY: "spod-3"})] * 8)
    assert facts == {ACCELERATOR_DOMAIN_KEY: "spod-3"}


def test_cards_disagreeing_claim_no_domain():
    """The model has one domain per host; a host straddling two is not
    represented, and claiming either would be wrong for half its cards."""
    facts = facts_from_devices(
        [dev(**{NVIDIA_CLIQUE_KEY: "u.1"}), dev(**{NVIDIA_CLIQUE_KEY: "u.2"})]
    )
    assert NVIDIA_CLIQUE_KEY not in facts


def test_switches_collect_into_a_sorted_set():
    """Eight ports on one switch is one value; eight ports on eight switches
    (a rail-optimised fabric) is the set, and two hosts with the same set are
    in the same rail group."""
    facts = facts_from_devices(
        [
            dev(**{SWITCH_KEY: "bb", SWITCH_NAME_KEY: "leaf-2"}),
            dev(**{SWITCH_KEY: "aa", SWITCH_NAME_KEY: "leaf-1"}),
            dev(**{SWITCH_KEY: "bb", SWITCH_NAME_KEY: "leaf-2"}),
        ]
    )
    assert facts == {SWITCH_KEY: "aa+bb", SWITCH_NAME_KEY: "leaf-1+leaf-2"}


def test_devices_without_hints_yield_nothing():
    assert facts_from_devices([dev(), SimpleNamespace()]) == {}
    assert facts_from_devices(None) == {}


# --- host ports ------------------------------------------------------------- #


def test_device_facts_win_over_host_facts():
    """KV transfer runs over the card's own port where it has one."""
    merged = merge_facts({SWITCH_KEY: "card-side"}, {SWITCH_KEY: "host-side", "x": "y"})
    assert merged == {SWITCH_KEY: "card-side", "x": "y"}


def test_the_probe_folds_neighbors_like_devices(monkeypatch):
    neighbors = [
        SimpleNamespace(chassis_id="bb", system_name="leaf-2"),
        SimpleNamespace(chassis_id="aa", system_name="leaf-1"),
    ]
    import gpustack_runtime.detector as detector

    monkeypatch.setattr(
        detector,
        "detect_lldp_neighbors",
        lambda interfaces=None: neighbors,
        raising=False,
    )

    assert HostSwitchProbe.probe_once(interfaces=["eth0"]) == {
        SWITCH_KEY: "aa+bb",
        SWITCH_NAME_KEY: "leaf-1+leaf-2",
    }


def test_the_probe_reports_nothing_when_nothing_was_heard(monkeypatch):
    import gpustack_runtime.detector as detector

    monkeypatch.setattr(
        detector, "detect_lldp_neighbors", lambda interfaces=None: [], raising=False
    )
    assert HostSwitchProbe.probe_once(interfaces=["eth0"]) == {}
