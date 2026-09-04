"""The interface between the solver's commits and the allocation accounting.

🔴 This file exists because of a defect a live run found and 14 unit tests did
not: those tests stubbed the capacity function entirely, so the boundary
between `group_solver` and `group_capacity` was never exercised. On a real
server it surfaced as

    Stopped counting capacity on worker gw1: '_Committed' object has no
    attribute 'gpu_type'

`_Committed` carries only `worker_id` and `role` — deliberately, its docstring
says "the capacity function re-derives the real resource claim itself, and
inventing one here would be a second, quieter accounting of the same
placement". `compute_worker_allocated` meanwhile reads four fields off every
entry it is handed. The re-derivation is `GroupCapacity._translate`, and its
absence made every domain look unmeasurable.

What is pinned here is the *contract*, not the implementation: whatever
`_Committed` grows or loses, an entry reaching the accounting must carry a
claim.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.schemas.models import ComputedResourceClaim, Model, RoleSpec
from gpustack.scheduler.group_capacity import GroupCapacity, _RoleProjection
from gpustack.scheduler.group_solver import _Committed

CLAIM = ComputedResourceClaim(vram={0: 36 * 1024**3}, ram=0)


def _worker(id: int):
    return SimpleNamespace(id=id, name=f"w{id}", ip=f"10.0.0.{id}", labels={})


def _model():
    model = Model(name="pd", source="huggingface", huggingface_repo_id="x/y")
    model.id = 1
    model.roles = [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
    ]
    return model


class _FakeSelector:
    """Answers "one more fits, on card 0" and records what it was asked."""

    seen: list = []

    def __init__(self, instances):
        _FakeSelector.seen.append(list(instances))

    async def select_candidates(self, workers):
        return [
            SimpleNamespace(
                worker=workers[0],
                gpu_indexes=[0],
                gpu_type="cuda",
                gpu_addresses=None,
                computed_resource_claim=CLAIM,
                subordinate_workers=None,
                overcommit=False,
            )
        ]


def _capacity(workers):
    cap = GroupCapacity(SimpleNamespace(), _model(), workers, [])
    # Both stubbed: the real ones need model metadata off the network, and what
    # is under test is the translation between them.
    cap._selector = lambda model, instances, cpu_only, ram_claim=None: _FakeSelector(
        instances
    )
    return cap


@pytest.mark.asyncio
async def test_a_solver_commit_reaches_the_accounting_with_a_claim():
    """🔴 The regression. `_Committed` has no claim of its own, and handing it
    to the accounting raised `AttributeError` from deep inside
    `compute_worker_allocated` — reported as "capacity could not be measured",
    which reads as a broken cluster rather than a broken translation."""
    workers = [_worker(1)]
    cap = _capacity(workers)
    _FakeSelector.seen = []

    with patch.object(
        GroupCapacity, "_eligible_for", return_value={1: workers[0]}
    ) as eligible:
        eligible.return_value = {1: workers[0]}
        cap._eligible["prefill"] = {1: workers[0]}
        cap._eligible["decode"] = {1: workers[0]}
        cap._projected["prefill"] = _RoleProjection(_model(), False)
        cap._projected["decode"] = _RoleProjection(_model(), False)

        # First role: this is where the claim is learned.
        first = await cap(role="prefill", worker_ids=[1], already_placed=[])
        assert first == {1: 1}

        # Second role, carrying the solver's own commit object.
        _FakeSelector.seen = []
        second = await cap(
            role="decode",
            worker_ids=[1],
            already_placed=[_Committed(worker_id=1, role="prefill")],
        )

    # Did not raise, and the commit arrived as something with a claim on it.
    assert second == {1: 1}
    handed_over = _FakeSelector.seen[0]
    translated = [e for e in handed_over if getattr(e, "worker_id", None) == 1]
    assert translated, "the commit was dropped instead of translated"
    entry = translated[0]
    # The four fields `compute_worker_allocated` reads. Asserted by name
    # because that is the contract that broke.
    assert entry.computed_resource_claim is CLAIM
    assert entry.gpu_type == "cuda"
    assert hasattr(entry, "gpu_indexes")
    assert hasattr(entry, "distributed_servers")


@pytest.mark.asyncio
async def test_an_untranslatable_commit_is_dropped_not_invented():
    """A claim guessed here would be the "second, quieter accounting" the
    solver's docstring warns about — it would make the domain look either
    roomier or tighter than the placement it is meant to reflect. Unreachable
    in practice (the solver only commits where a count already ran), so
    dropping with a warning is the right shape for a guard."""
    cap = _capacity([_worker(1)])
    out = cap._translate([_Committed(worker_id=99, role="never-counted")])
    assert out == []


@pytest.mark.asyncio
async def test_entries_that_already_carry_a_claim_pass_through():
    """The commit pass builds real stand-ins itself; translating them again
    would drop the concrete card assignment it just computed."""
    cap = _capacity([_worker(1)])
    ready = SimpleNamespace(
        worker_id=1,
        gpu_indexes=[3],
        gpu_type="cuda",
        computed_resource_claim=CLAIM,
        distributed_servers=None,
    )
    assert cap._translate([ready]) == [ready]
