"""Whether a group's KV transfers can stay off the network.

The gap this closes was found by reading the manual-selection path: a user
picks host A's cards for prefill and host B's for decode, both roles fit, the
group starts, every request works — and every single KV transfer crosses the
network. Nothing said a word. On a link without RDMA that arrangement makes PD
strictly worse than not disaggregating at all (§0.2 T4), and the only feedback
today is a TTFT regression nobody attributes to placement.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.controllers import _pairing_remote, pairing_locality


def _model(roles=("prefill", "decode")):
    return SimpleNamespace(
        roles=[SimpleNamespace(name=name, replicas=1) for name in roles]
    )


def _instance(role, worker_id, state=ModelInstanceStateEnum.RUNNING):
    return SimpleNamespace(role=role, worker_id=worker_id, state=state)


def test_one_host_makes_every_transfer_local():
    instances = [_instance("prefill", 1), _instance("decode", 1)]
    assert pairing_locality(_model(), instances) == 1.0
    assert _pairing_remote(_model(), instances) is False


def test_roles_split_across_hosts_can_never_be_local():
    """The arrangement manual selection produces by accident."""
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 1),
        _instance("decode", 2),
        _instance("decode", 2),
    ]
    assert pairing_locality(_model(), instances) == 0.0
    assert _pairing_remote(_model(), instances) is True


def test_an_even_spread_reproduces_the_one_over_x_ceiling():
    """🔴 The reason the threshold is zero rather than a fraction.

    A 2P2D placed one pair per host — the best placement there is — still only
    keeps half its transfers local, because the router picks a prefill and a
    decode independently. A threshold expressed as a fraction would fire here,
    on the arrangement it is supposed to reward.
    """
    for x in (2, 3, 4):
        instances = [_instance("prefill", w) for w in range(1, x + 1)]
        instances += [_instance("decode", w) for w in range(1, x + 1)]
        assert pairing_locality(_model(), instances) == pytest.approx(1 / x)
        assert _pairing_remote(_model(), instances) is False


def test_a_partial_overlap_is_between_the_two():
    # prefill on 1,2 · decode both on 1 -> half the prefill picks are local.
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 2),
        _instance("decode", 1),
    ]
    assert pairing_locality(_model(), instances) == pytest.approx(0.5)
    assert _pairing_remote(_model(), instances) is False


def test_only_running_members_count():
    """A member that is not up occupies no host yet, and counting it would let
    a starting group look local before anything is placed."""
    instances = [
        _instance("prefill", 1),
        _instance("decode", 2),
        _instance("decode", 1, state=ModelInstanceStateEnum.INITIALIZING),
    ]
    assert pairing_locality(_model(), instances) == 0.0


def test_the_router_is_not_part_of_the_pairing():
    """It holds no KV, so where it sits cannot make a transfer local."""
    instances = [
        _instance("prefill", 1),
        _instance("decode", 1),
        _instance("router", 2),
    ]
    assert pairing_locality(_model(), instances) == 1.0


@pytest.mark.parametrize(
    "instances",
    [
        [],
        [_instance("prefill", 1)],
        [_instance("decode", 1)],
        [_instance("prefill", None), _instance("decode", 1)],
    ],
)
def test_silence_rather_than_zero_when_the_question_does_not_apply(instances):
    """🔴 `None` and `0.0` are different answers.

    A role with no running member has no placement to judge, and reporting 0
    there would mark every group PAIRING_REMOTE for the whole window between
    the first member starting and the last."""
    assert pairing_locality(_model(), instances) is None
    assert _pairing_remote(_model(), instances) is False


def test_a_model_without_roles_is_not_a_group():
    model = SimpleNamespace(roles=None)
    assert pairing_locality(model, [_instance("prefill", 1)]) is None
    assert _pairing_remote(model, [_instance("prefill", 1)]) is False
