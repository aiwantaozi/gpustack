"""«As close as possible», meaning more than «the same host».

🔴 The two scorers that pull a later member toward its group — pairing affinity
for a scaled-out prefill or decode, group locality for the router — both
compare `worker_id` and nothing else. Between a worker in the members' own rack
and one three racks away they were indifferent, so the deploy form's top option
stopped at the host, and every tier below it was a target to be reported
against rather than one to aim at.
"""

from types import SimpleNamespace

import pytest

from gpustack.policies.scorers.topology_proximity_scorer import (
    TopologyProximityScorer,
)
from gpustack.scheduler.topology_view import build_view
from tests.utils.topology_layers import layer_obj

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.gpustack.ai/zone"


def _worker(id_: int, rack=None, zone=None):
    labels = {}
    if rack:
        labels[RACK] = rack
    if zone:
        labels[ZONE] = zone
    return SimpleNamespace(id=id_, name=f"w{id_}", labels=labels, cluster_id=1)


def _instance(id_, role, worker_id=None, group_id="g1"):
    return SimpleNamespace(id=id_, role=role, worker_id=worker_id, group_id=group_id)


def _candidate(worker):
    return SimpleNamespace(worker=worker, score=None)


def _view(workers, layers=None):
    return build_view(
        SimpleNamespace(
            layers=layers or [layer_obj("zone", [ZONE]), layer_obj("rack", [RACK])]
        ),
        workers,
    )


async def _score(workers, placed, candidates=None, anchors=("prefill", "decode")):
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers), anchors, max_score=150.0
    )
    scored = await scorer.score([_candidate(w) for w in (candidates or workers)])
    return {c.worker.id: (c.score or 0) for c in scored}


@pytest.mark.asyncio
async def test_a_worker_in_the_members_rack_beats_one_a_zone_away():
    """The case the old scorers could not see: neither candidate is the host a
    member sits on, so `worker_id` comparison calls them equal."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-2"),
    ]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed, candidates=workers[1:])

    assert scores[2] > scores[3]


@pytest.mark.asyncio
async def test_the_same_zone_still_beats_another_zone():
    """Every declared rung counts, not just the tightest one — a group split
    across two racks of one zone is closer than one split across two zones."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-b", "zone-1"),
        _worker(3, "rack-c", "zone-2"),
    ]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed, candidates=workers[1:])

    assert scores[2] > scores[3]


@pytest.mark.asyncio
async def test_the_members_own_host_scores_highest():
    workers = [_worker(1, "rack-a", "zone-1"), _worker(2, "rack-a", "zone-1")]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed)

    assert scores[1] > scores[2]


@pytest.mark.asyncio
async def test_unclassified_workers_are_not_treated_as_close():
    """🔴 The bucket means "we do not know where these are". Reading that as
    "these are together" turns a missing label into a confident wrong answer,
    and `common_layer` refuses to do it — this must not undo that refusal."""
    workers = [_worker(1), _worker(2)]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed)

    assert scores[2] == 0


@pytest.mark.asyncio
async def test_nothing_placed_yet_scores_nothing():
    """A group whose first member is being placed has nothing to be near, and
    the resource scorers decide alone — exactly as before this existed."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]

    scores = await _score(workers, [])

    assert set(scores.values()) == {0}


@pytest.mark.asyncio
async def test_the_router_does_not_anchor_the_group():
    """It holds no weights, so where it sits is not where the group is. A
    stray router would otherwise pull every later member after it."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(9, "router", worker_id=2)]

    scores = await _score(workers, placed)

    assert set(scores.values()) == {0}


@pytest.mark.asyncio
async def test_the_router_is_still_pulled_toward_the_members():
    """Not an anchor, but scored against them like anyone: it forwards every
    token the group serves."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed, anchors=("prefill", "decode"))

    assert scores[1] > scores[2]


@pytest.mark.asyncio
async def test_another_groups_members_are_not_anchors():
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(1, "prefill", worker_id=2, group_id="other")]

    scores = await _score(workers, placed)

    assert set(scores.values()) == {0}


@pytest.mark.asyncio
async def test_a_cluster_with_no_declared_topology_still_prefers_the_host():
    """The built-in host rung exists whatever the operator declared, so «as
    close as possible» keeps the one meaning it always had."""
    workers = [_worker(1), _worker(2)]
    placed = [_instance(1, "prefill", worker_id=1)]
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers, layers=[]), ("prefill",), max_score=150.0
    )

    scored = await scorer.score([_candidate(w) for w in workers])
    scores = {c.worker.id: (c.score or 0) for c in scored}

    assert scores[1] > scores[2]


@pytest.mark.asyncio
async def test_a_zero_weight_turns_it_off():
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(1, "prefill", worker_id=1)]
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers), ("prefill",), max_score=0
    )

    scored = await scorer.score([_candidate(w) for w in workers])

    assert all(c.score is None for c in scored)
