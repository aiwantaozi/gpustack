from types import SimpleNamespace

import pytest

from gpustack.policies.scorers.group_locality_scorer import GroupLocalityScorer


def _instance(group_id, worker_id, role="prefill"):
    return SimpleNamespace(group_id=group_id, worker_id=worker_id, role=role)


def _candidate(worker_id):
    return SimpleNamespace(worker=SimpleNamespace(id=worker_id), score=None)


@pytest.mark.asyncio
async def test_a_worker_holding_a_sibling_wins():
    """The whole point: the router forwards every token to these members, so
    sitting on one of their hosts removes a network hop per request."""
    scorer = GroupLocalityScorer(
        "g1", [_instance("g1", 7), _instance("g1", 7, "decode")]
    )
    a, b = _candidate(7), _candidate(9)

    await scorer.score([a, b])

    assert a.score == 100.0
    assert b.score == 0.0


@pytest.mark.asyncio
async def test_two_siblings_on_one_host_are_worth_no_more_than_one():
    """Counted per worker, not per member. The router talks to prefill and
    decode over the same link, so the first sibling buys the hop and the
    second buys nothing."""
    one = GroupLocalityScorer("g1", [_instance("g1", 7)])
    two = GroupLocalityScorer("g1", [_instance("g1", 7), _instance("g1", 7, "decode")])
    a, b = _candidate(7), _candidate(7)

    await one.score([a])
    await two.score([b])

    assert a.score == b.score


@pytest.mark.asyncio
async def test_another_groups_members_do_not_pull():
    """Two disaggregated deployments on one cluster must not attract each
    other's routers."""
    scorer = GroupLocalityScorer("g1", [_instance("g2", 7)])
    candidate = _candidate(7)

    await scorer.score([candidate])

    assert candidate.score is None or candidate.score == 0


@pytest.mark.asyncio
async def test_nothing_placed_yet_leaves_every_candidate_alone():
    """A router whose siblings have no worker yet cannot be near them. The
    other scorers decide, which is the behaviour before this scorer existed —
    and the ordering that creates the router last normally prevents it."""
    scorer = GroupLocalityScorer("g1", [_instance("g1", None)])
    a, b = _candidate(7), _candidate(9)

    await scorer.score([a, b])

    assert a.score is None and b.score is None


@pytest.mark.asyncio
async def test_it_never_removes_a_candidate():
    """🔴 A scorer, not a filter, and this is the safety property. If no
    sibling's worker can take the router it still places, one hop further
    away: the cost is latency, never schedulability. A filter here would let a
    full worker hold the whole group hostage."""
    scorer = GroupLocalityScorer("g1", [_instance("g1", 7)])
    candidates = [_candidate(1), _candidate(2), _candidate(3)]

    result = await scorer.score(candidates)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_a_role_less_deployment_is_untouched():
    """No group id, no opinion — the scorer is constructed only for a member
    of a group, but it must be inert rather than wrong if that ever changes."""
    scorer = GroupLocalityScorer(None, [_instance("g1", 7)])
    candidate = _candidate(7)

    await scorer.score([candidate])

    assert candidate.score is None


@pytest.mark.asyncio
async def test_zero_max_score_turns_it_off():
    """The env knob is an off switch, so a cluster that wants its router
    placed by resource fit alone has one."""
    scorer = GroupLocalityScorer("g1", [_instance("g1", 7)], max_score=0)
    candidate = _candidate(7)

    await scorer.score([candidate])

    assert candidate.score is None
