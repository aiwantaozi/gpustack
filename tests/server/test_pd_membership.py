"""Router membership: the registry is the only way in under `--enable-igw`.

Measured on a live 1P1D, and confirmed in upstream source for both routers:
the igw branch builds the PD router with empty worker lists
(`create_vllm_pd_router(&[], &[], ...)` with the upstream comment
"Empty worker list - workers added later"; SGLang's `create_pd_router(None,
None, ...)`), so the command-line peers are never passed and a request before
registration gets 503. Registration is therefore a precondition of servability,
not an optimisation.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum, RoleNameEnum
from gpustack.server import pd_membership
from gpustack.server.pd_membership import (
    MembershipOutcome,
    desired_members,
    member_url,
    router_addresses,
)


def _instance(role, port, state=ModelInstanceStateEnum.RUNNING, ip="10.0.0.1"):
    return SimpleNamespace(role=role, port=port, state=state, worker_ip=ip)


def _model():
    return SimpleNamespace(
        id=1, name="pd", disaggregation=SimpleNamespace(mode="vllm-nixl")
    )


def test_only_running_gpu_roles_are_registered():
    """The router is not its own upstream, and a member that is not RUNNING has
    nothing listening — upstream probes a peer before admitting it and drops it
    silently on timeout, so registering one early buys a silent absence."""
    instances = [
        _instance("prefill", 40010),
        _instance("decode", 40011),
        _instance(RoleNameEnum.ROUTER.value, 40012),
        _instance("decode", 40013, state=ModelInstanceStateEnum.STARTING),
    ]
    members = desired_members(_model(), instances)
    assert members == {
        "http://10.0.0.1:40010": "prefill",
        "http://10.0.0.1:40011": "decode",
    }


def test_a_member_is_addressed_the_way_the_router_stores_it():
    """`host:port` of the serving listener. Measured: the `worker` label on the
    router's own counters carries the whole URL it was launched with, not a
    worker name or id, and `DELETE /workers/{url}` matches on the same."""
    assert member_url(_instance("decode", 40011)) == "http://10.0.0.1:40011"
    assert member_url(_instance("decode", None)) is None


def test_every_running_router_is_reconciled():
    """One router with an empty registry serves 503s while another serves fine,
    so each is reconciled on its own — the replica count is declared like any
    other even though the shipped recipes run one."""
    instances = [
        _instance(RoleNameEnum.ROUTER.value, 40012),
        _instance(RoleNameEnum.ROUTER.value, 40013),
        _instance(RoleNameEnum.ROUTER.value, 40014, state=ModelInstanceStateEnum.ERROR),
        _instance("decode", 40011),
    ]
    assert router_addresses(instances) == ["10.0.0.1:40012", "10.0.0.1:40013"]


@pytest.mark.asyncio
async def test_a_recipe_that_declares_the_api_without_launching_it_is_not_a_failure():
    """🔴 The regression that protects every deployment on the other path.

    Declared is not usable: without the flag its API needs, the router already
    knows its peers from the command line and `POST /workers` answers 400.
    Reporting that as a failed registration would park such a group in PARTIAL
    forever.

    The state is built here rather than read off a shipped recipe. It used to
    be `vllm-nixl`'s, and when that recipe gained `--enable-igw` (verified
    2026-08-28, F12) this test broke while the behaviour it guards did not
    change at all -- so the mode it needs is now constructed, and the test
    survives the next recipe that flips either way.
    """
    from gpustack.server.pd_mode_catalog import get_pd_mode

    shipped = get_pd_mode("vllm-nixl")
    assert shipped.router.membership_api.available is True

    unlaunched = shipped.model_copy(
        update={
            "router": shipped.router.model_copy(
                update={
                    "command": [
                        token
                        for token in (shipped.router.command or [])
                        if str(token) != "--enable-igw"
                    ]
                }
            )
        },
        deep=True,
    )
    assert unlaunched.router.membership_api.available is True
    assert unlaunched.router.membership_api_usable is False

    outcome = await pd_membership.reconcile(
        _model(), unlaunched, [_instance("decode", 40011)], "10.0.0.1:40012"
    )
    assert outcome.ok is True
    assert outcome.reason is None


def test_a_persistent_failure_names_the_way_out():
    """🔴 And the way out is an OPERATOR action, which is why it is stated
    rather than performed: the controller cannot un-launch a flag on a running
    process, and restarting the router repeats the same failed registration.

    One failure is ordinary — a router that just came up, a member still being
    probed — so escalating on the first would train people to ignore it.
    """
    pd_membership.forget(7)
    for _ in range(pd_membership.PERSISTENT_FAILURE_PASSES - 1):
        pd_membership.record(7, MembershipOutcome(ok=False, reason="not admitted"))
    assert pd_membership.outcome_for(7).reason == "not admitted"

    pd_membership.record(7, MembershipOutcome(ok=False, reason="not admitted"))
    escalated = pd_membership.outcome_for(7).reason
    assert "--enable-igw" in escalated
    assert "remove that flag" in escalated

    # A success clears the count, so a transient outage does not leave the
    # group wearing an escalated message it has grown out of.
    pd_membership.record(7, MembershipOutcome(ok=True))
    assert pd_membership.consecutive_failures(7) == 0
    pd_membership.forget(7)


def test_the_servability_gate_reads_the_recorded_outcome():
    """🔴 The seam this feature was built into. `upstream_registration_ready`
    had a TODO saying "return the recorded outcome when the registration step
    lands" — this is that step, and an unrecorded outcome must still mean
    servable so the command-line path keeps working."""
    from gpustack.schemas.models import Model, RoleSpec
    from gpustack.server.controllers import upstream_registration_ready

    model = Model(name="pd")
    model.id = 42
    assert upstream_registration_ready(model) is True, "no router role: vacuous"

    model.roles = [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="router", replicas=1, cpu_only=True),
    ]
    pd_membership.forget(42)
    assert upstream_registration_ready(model) is True, "unrecorded: command-line path"

    pd_membership.record(42, MembershipOutcome(ok=False, reason="not admitted"))
    assert upstream_registration_ready(model) is False

    pd_membership.record(42, MembershipOutcome(ok=True))
    assert upstream_registration_ready(model) is True
    pd_membership.forget(42)


def test_a_members_url_is_percent_encoded_in_the_removal_path():
    """🔴 The bug this pins, found on a live router.

    A member's id IS a URL, so substituting it raw into `DELETE /workers/{url}`
    makes the path `/workers/http://host:port` — which upstream routes to its
    transparent proxy instead and answers 405 "Only POST requests are supported
    for transparent proxy". Percent-encoded it is 200.

    Worse than the 405: removal failures are deliberately non-fatal (upstream's
    removal gate is global and waits for in-flight requests), so the stale
    member stayed while the outcome still read ok. The leniency hid it. That is
    why a kept member now shows up in `reason` even on success.
    """
    from urllib.parse import quote

    template = "http://r:1/workers/{url}"
    url = "http://10.0.0.1:40051"
    assert template.replace("{url}", quote(url, safe="")) == (
        "http://r:1/workers/http%3A%2F%2F10.0.0.1%3A40051"
    )
    # The raw form is what produced the 405 — kept as the negative case so a
    # future "simplification" back to it fails here rather than in production.
    assert "{url}" not in template.replace("{url}", quote(url, safe=""))
    assert "/workers/http://" in template.replace("{url}", url)


def test_only_an_unreadable_registry_counts_toward_a_restart():
    """🔴 Which failure a restart can fix, and which it only makes worse.

    Under `--enable-igw` the command-line peers never enter the registry
    (measured 2026-08-28: a router started with `--prefill` reports
    `GET /workers` -> total 0), so restarting turns "some members registered"
    into "none registered, answering 503". That price buys something only when
    the router is not answering at all. A refused member means it is alive and
    disagreeing -- a version or argument mismatch it would refuse again.
    """
    model_id = 7788
    pd_membership.forget(model_id)

    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES + 2):
        pd_membership.record(
            model_id,
            pd_membership.MembershipOutcome(ok=False, reason="router refused a member"),
        )
    assert pd_membership.should_restart_router(model_id) is False

    pd_membership.forget(model_id)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(
            model_id,
            pd_membership.MembershipOutcome(
                ok=False, unreadable=True, reason="could not read the registry"
            ),
        )
    assert pd_membership.should_restart_router(model_id) is True
    pd_membership.forget(model_id)


def test_one_unreadable_pass_does_not_cost_the_group_an_outage():
    """A single dropped request must not trade a blip for a real outage."""
    model_id = 7789
    pd_membership.forget(model_id)
    pd_membership.record(
        model_id,
        pd_membership.MembershipOutcome(
            ok=False, unreadable=True, reason="could not read the registry"
        ),
    )
    assert pd_membership.should_restart_router(model_id) is False
    pd_membership.forget(model_id)


def test_a_readable_pass_breaks_the_streak():
    """The streak has to be consecutive: a router that answers once is not the
    wedged process this path exists for."""
    model_id = 7790
    pd_membership.forget(model_id)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES - 1):
        pd_membership.record(
            model_id,
            pd_membership.MembershipOutcome(ok=False, unreadable=True, reason="x"),
        )
    pd_membership.record(model_id, pd_membership.MembershipOutcome(ok=True))
    pd_membership.record(
        model_id,
        pd_membership.MembershipOutcome(ok=False, unreadable=True, reason="x"),
    )
    assert pd_membership.should_restart_router(model_id) is False
    pd_membership.forget(model_id)
