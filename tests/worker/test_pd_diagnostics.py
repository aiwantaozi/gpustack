"""Turning a disaggregated member's failure into something actionable.

Two measured problems shape this, and both tests below are written against
them rather than against the code.

`_handle_failed_transfer` raises `IndexError: list index out of range` while
the real reason was logged earlier, so reporting the last exception reports the
symptom of the symptom. It was first assumed this only happened on a
tensor-parallelism mismatch; it happens on any handshake failure. Hence
earliest-match-wins, which is the single property most of these tests exist to
pin.

And every port-level failure produces the same shape — bind, fail, exit,
restart, `starting` forever — so a member restarting repeatedly without ever
serving has failed, whatever its state column says.
"""

from datetime import datetime, timedelta, timezone

from gpustack.schemas.models import PortBand
from gpustack.worker.pd_diagnostics import RestartTracker, diagnose

T0 = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


# --- earliest match wins --------------------------------------------------- #


def test_the_root_cause_beats_the_exception_that_escaped():
    """The measured case. `IndexError` is what propagates out of vLLM, and it
    says nothing a user can act on."""
    log = "\n".join(
        [
            "INFO loading model",
            "ERROR NIXL_ERR_BACKEND: failed to load remote metadata",
            "ERROR Traceback (most recent call last):",
            "ERROR   File nixl_connector.py, in _handle_failed_transfer",
            "ERROR IndexError: list index out of range",
        ]
    )
    found = diagnose(log)

    assert found.signature == "NIXL_ERR_BACKEND"
    assert "kv_ifname" in found.summary


def test_the_earliest_signature_wins_over_a_later_one():
    """Not a priority order — the first line in a cascade is the one that
    caused the rest."""
    log = "\n".join(
        [
            "ERROR Address already in use: 0.0.0.0:40031",
            "ERROR NIXL_ERR_BACKEND: handshake failed",
        ]
    )
    assert diagnose(log).signature == "address already in use"

    reordered = "\n".join(
        [
            "ERROR NIXL_ERR_BACKEND: handshake failed",
            "ERROR Address already in use: 0.0.0.0:40031",
        ]
    )
    assert diagnose(reordered).signature == "NIXL_ERR_BACKEND"


def test_a_clean_log_diagnoses_nothing():
    assert diagnose("INFO started\nINFO listening on 40027") is None
    assert diagnose("") is None
    assert diagnose(None) is None


# --- the signatures -------------------------------------------------------- #


def test_a_hash_mismatch_points_at_the_factors_that_are_hashed():
    found = diagnose("ERROR kv compatibility hash mismatch with remote engine")

    assert found.signature == "compatibility hash mismatch"
    assert "block size" in found.summary


def test_rdma_unavailability_points_at_the_container_capability():
    found = diagnose("ERROR rdma_create_event_channel failed: No such device")

    assert found.signature == "rdma unavailable"
    assert "IPC_LOCK" in found.summary


def test_the_measured_zmq_error_is_recognised():
    """The M0 failure verbatim: a template placeholder that reached the engine
    unrendered, reported as a device error."""
    found = diagnose("ZMQError: No such device (addr='tcp://{{worker_ip}}:5600')")

    assert found is not None
    assert "kv_ifname" in found.summary or "placeholder" in found.summary


def test_an_unrendered_placeholder_is_named():
    found = diagnose("INFO UCX_NET_DEVICES={{net_device}}")

    assert found.signature == "unresolved placeholder"
    assert "{{net_device}}" in found.summary


def test_two_kv_transfer_configs_are_recognised():
    """The engine accepts exactly one, and the pre-checks refuse the
    combination — this catches the case that got past them."""
    found = diagnose("ERROR --kv-transfer-config already specified")

    assert found.signature == "kv connector conflict"
    assert "extended KV cache" in found.summary


# --- attributing a port to its band ---------------------------------------- #


def test_a_taken_port_is_attributed_to_its_band():
    """ "Port 40031 is taken" is a number; "the kv_side_channel band is taken"
    is something to fix."""
    found = diagnose(
        "ERROR Address already in use: 192.168.50.15:40031",
        named_ports={"kv_side_channel": PortBand(base=40031, count=1)},
    )

    assert "kv_side_channel" in found.summary


def test_a_derived_port_is_attributed_to_the_band_it_falls_inside():
    """The number in the log is usually base+n, not the base: a connector
    derives several ports from one. Matching only the base would attribute
    nothing in exactly the cases where two members collided."""
    found = diagnose(
        "ERROR Address already in use: 41107",
        named_ports={"kv_port": PortBand(base=41100, count=8)},
    )

    assert "kv_port" in found.summary


def test_a_port_outside_every_band_is_not_attributed():
    """An engine-chosen port GPUStack cannot reserve. Naming a band it does not
    belong to would send the reader to the wrong setting."""
    found = diagnose(
        "ERROR Address already in use: 15051",
        named_ports={"kv_port": PortBand(base=41100, count=8)},
    )

    assert found.signature == "address already in use"
    assert "kv_port" not in found.summary


# --- crash-loop detection -------------------------------------------------- #


def test_repeated_restarts_without_serving_are_a_failure():
    tracker = RestartTracker(threshold=3, window=timedelta(minutes=5))

    assert tracker.observe_restart_count(1, 0, T0) is False
    assert tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=10)) is False
    assert tracker.observe_restart_count(1, 2, T0 + timedelta(seconds=20)) is False
    assert tracker.observe_restart_count(1, 3, T0 + timedelta(seconds=30)) is True


def test_a_member_that_served_is_never_reported_as_never_started():
    """A member that ran and then began crash-looping is a different failure:
    it ran, so its configuration is not the problem."""
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)
    tracker.observe_running(1)

    assert tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5)) is False
    assert tracker.observe_restart_count(1, 9, T0 + timedelta(seconds=10)) is False


def test_restarts_spread_out_are_not_a_loop():
    """Restarts alone are normal — a member may legitimately be replaced. The
    signal is a rate, which is why a cumulative count on the row cannot carry
    it."""
    tracker = RestartTracker(threshold=3, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)

    assert tracker.observe_restart_count(1, 1, T0 + timedelta(minutes=10)) is False
    assert tracker.observe_restart_count(1, 2, T0 + timedelta(minutes=20)) is False
    assert tracker.observe_restart_count(1, 3, T0 + timedelta(minutes=30)) is False


def test_the_first_observation_is_not_counted_as_a_restart():
    """A worker restarting mid-loop inherits a non-zero count it did not
    witness, and counting it would attribute the previous history to now."""
    tracker = RestartTracker(threshold=1, window=timedelta(minutes=5))

    assert tracker.observe_restart_count(1, 7, T0) is False


def test_a_recreated_workload_resets_rather_than_underflowing():
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 5, T0)
    tracker.observe_restart_count(1, 6, T0 + timedelta(seconds=5))

    # Workload recreated: the count starts over.
    assert tracker.observe_restart_count(1, 0, T0 + timedelta(seconds=10)) is False
    assert tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=15)) is False


def test_an_unchanged_count_does_not_add_a_restart():
    """A polling loop observes the same count many times between restarts."""
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)
    tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5))
    for i in range(20):
        looping = tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=10 + i))
    assert looping is False


def test_forgetting_a_member_clears_its_verdict():
    """A recreated instance reusing the id must not inherit one."""
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)
    tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5))
    assert tracker.observe_restart_count(1, 2, T0 + timedelta(seconds=10)) is True

    tracker.forget(1)
    assert tracker.observe_restart_count(1, 0, T0 + timedelta(seconds=15)) is False


def test_members_are_tracked_independently():
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    for instance_id in (1, 2):
        tracker.observe_restart_count(instance_id, 0, T0)
    tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5))

    assert tracker.observe_restart_count(1, 2, T0 + timedelta(seconds=10)) is True
    assert tracker.observe_restart_count(2, 1, T0 + timedelta(seconds=10)) is False
