from types import SimpleNamespace

import pytest

from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.scorers.pairing_affinity_scorer import PairingAffinityScorer
from gpustack.schemas.models import ComputedResourceClaim, RoleNameEnum

GROUP = "g-1"


def _member(worker_id, role, group_id=GROUP):
    return SimpleNamespace(worker_id=worker_id, role=role, group_id=group_id)


def _candidate(worker_id):
    return ModelInstanceScheduleCandidate(
        worker=SimpleNamespace(id=worker_id),
        gpu_indexes=[],
        computed_resource_claim=ComputedResourceClaim(ram=0, vram={}),
        score=None,
    )


async def _scores(role, members, worker_ids, **kwargs):
    candidates = [_candidate(worker_id) for worker_id in worker_ids]
    scorer = PairingAffinityScorer(GROUP, role, members, **kwargs)
    scored = await scorer.score(candidates)
    return {c.worker.id: c.score for c in scored}


@pytest.mark.asyncio
async def test_prefill_prefers_the_worker_with_most_decodes():
    """The rule itself: scaling prefill out follows decode, not prefill."""
    members = [
        _member(1, RoleNameEnum.DECODE.value),
        _member(1, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.DECODE.value),
        _member(3, RoleNameEnum.PREFILL.value),
    ]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2, 3])

    assert scores[1] > scores[2] > scores[3]
    assert scores[3] == 0


@pytest.mark.asyncio
async def test_decode_prefers_the_worker_with_most_prefills():
    """Symmetric, and not the same worker as the test above picks."""
    members = [
        _member(1, RoleNameEnum.DECODE.value),
        _member(1, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.PREFILL.value),
    ]

    scores = await _scores(RoleNameEnum.DECODE.value, members, [1, 2])

    assert scores[2] > scores[1]


@pytest.mark.asyncio
async def test_group_member_count_is_not_the_rule():
    """🔴 The failure this scorer exists to prevent.

    Worker 1 holds three prefills and no decode; worker 2 holds one decode.
    "Most members of this group" would pick worker 1 and move
    `sum p_j * d_j` by exactly nothing. 3P1D is the shape that separates the
    two rules — under a balanced ratio they agree, which is why the wrong one
    survives review.
    """
    members = [
        _member(1, RoleNameEnum.PREFILL.value),
        _member(1, RoleNameEnum.PREFILL.value),
        _member(1, RoleNameEnum.PREFILL.value),
        _member(2, RoleNameEnum.DECODE.value),
    ]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2])

    assert scores[2] > scores[1]
    assert scores[1] == 0


@pytest.mark.asyncio
async def test_one_more_sibling_outweighs_every_other_scale_up_scorer():
    """Affinity is the primary order; capacity may only break ties.

    The chain sums the scorers, so this holds only while a single step of
    affinity is worth more than the whole range the resource scorers can
    move a candidate. Normalising the score into a fixed band would break it
    silently, and only for large groups.
    """
    import gpustack.envs as envs

    members = [
        _member(1, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.DECODE.value),
    ]

    scores = await _scores(
        RoleNameEnum.PREFILL.value,
        members,
        [1, 2],
        max_score=envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE,
    )

    other_scorers_ceiling = (
        envs.SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE
        + envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE
    )
    assert scores[2] - scores[1] > other_scorers_ceiling


@pytest.mark.asyncio
async def test_placed_but_not_yet_running_siblings_count():
    """Counted off `worker_id`, so a burst of scale-outs does not stack.

    A sibling that is scheduled but still starting already holds that worker's
    cards and will pair from there.
    """
    members = [_member(7, RoleNameEnum.DECODE.value)]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [7, 8])

    assert scores[7] > scores[8]


@pytest.mark.asyncio
async def test_other_groups_and_unplaced_members_are_ignored():
    members = [
        _member(1, RoleNameEnum.DECODE.value, group_id="another-group"),
        _member(2, RoleNameEnum.DECODE.value),
        _member(None, RoleNameEnum.DECODE.value),
    ]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2])

    assert scores[1] == 0
    assert scores[2] > 0


@pytest.mark.asyncio
async def test_no_opposite_member_placed_leaves_every_score_untouched():
    """Not zero — untouched, so the chain adds nothing and the resource
    scorers decide alone."""
    members = [_member(1, RoleNameEnum.PREFILL.value)]

    candidates = [_candidate(1), _candidate(2)]
    scorer = PairingAffinityScorer(GROUP, RoleNameEnum.PREFILL.value, members)
    scored = await scorer.score(candidates)

    assert all(c.score is None for c in scored)


@pytest.mark.asyncio
async def test_router_and_role_less_members_are_not_scored():
    """The two ways a member reaches this scorer with nothing to pair to."""
    members = [_member(1, RoleNameEnum.DECODE.value)]
    candidates = [_candidate(1), _candidate(2)]

    router = PairingAffinityScorer(GROUP, RoleNameEnum.ROUTER.value, members)
    assert all(c.score is None for c in await router.score(candidates))

    role_less = PairingAffinityScorer(None, None, members)
    assert all(c.score is None for c in await role_less.score(candidates))
