"""The topology preview: what a declaration does to a real fleet.

The endpoint is a thin shell over `scheduler.topology`, so what is tested here
is the part the scheduler does not have an opinion about — the counts, the
unclassified bucket's payload, and the fact that the *request body* wins over
the saved declaration. That last one is the reason the endpoint is a POST: the
question an operator asks is "what if this layer matched a different key",
asked before saving.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import cluster_topology as route
from gpustack.schemas.clusters import ClusterTopology
from gpustack.scheduler.topology import NODE_LAYER, UNCLASSIFIED

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.kubernetes.io/zone"


def _worker(id: int, name: str, labels=None, gpus=2):
    return SimpleNamespace(
        id=id,
        name=name,
        labels=labels or {},
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(index=i) for i in range(gpus)]
        ),
    )


def _topology(layers):
    return ClusterTopology.model_validate({"layers": layers})


async def _preview(workers, saved=None, body=None, allocated=None):
    """Call the handler with the ORM and the allocation cache stubbed.

    `allocated` maps worker id -> {gpu_index: vram}; a worker absent from it
    has nothing allocated. A value of the string "raise" makes the allocation
    read fail, which is a case of its own.
    """
    allocated = allocated or {}

    async def fake_allocated(worker_id):
        entry = allocated.get(worker_id, {})
        if entry == "raise":
            raise RuntimeError("no global config")
        return SimpleNamespace(ram=0, vram=entry)

    cluster = SimpleNamespace(id=1, topology=saved)
    with (
        patch.object(route.Cluster, "one_by_id", AsyncMock(return_value=cluster)),
        patch.object(route.Worker, "all_by_field", AsyncMock(return_value=workers)),
        patch.object(route, "assert_cluster_visible", lambda *a, **k: None),
        patch(
            "gpustack.server.worker_allocated_cache.get_worker_allocated",
            new=AsyncMock(side_effect=fake_allocated),
        ),
    ):
        return await route.preview_cluster_topology(
            session=None,
            ctx=None,
            id=1,
            body=route.TopologyPreviewRequest(topology=body) if body else None,
        )


def _find(node, name):
    if node.name == name:
        return node
    for child in node.children:
        found = _find(child, name)
        if found:
            return found
    return None


# --- a cluster that declared nothing ---------------------------------------- #


@pytest.mark.asyncio
async def test_no_declaration_still_produces_a_usable_tree():
    """The leaf is built in and takes the worker's name, so a cluster with no
    layers is not a degraded state — it simply cannot tell two workers apart
    above the host."""
    result = await _preview([_worker(1, "w1"), _worker(2, "w2")])

    assert result.layers == [NODE_LAYER]
    assert result.label_keys == {}
    assert result.total_workers == 2
    assert result.unclassified_workers == 0
    assert [c.name for c in result.root.children] == ["w1", "w2"]


@pytest.mark.asyncio
async def test_the_tightest_choice_is_offered_even_with_no_layers():
    """`layers` is what the deployment form's choices are built from, and the
    "at least on the same host" option must not depend on configuration."""
    result = await _preview([_worker(1, "w1")])
    assert NODE_LAYER in result.layers


# --- counts ----------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_counts_roll_up_through_the_tree():
    workers = [
        _worker(1, "w1", {RACK: "rack-a"}, gpus=4),
        _worker(2, "w2", {RACK: "rack-a"}, gpus=4),
        _worker(3, "w3", {RACK: "rack-b"}, gpus=2),
    ]
    result = await _preview(
        workers, saved=_topology([{"name": "Rack", "labelKeys": [RACK]}])
    )

    assert result.root.workers == 3
    assert result.root.gpus == 10
    rack_a = _find(result.root, "rack-a")
    assert (rack_a.workers, rack_a.gpus) == (2, 8)
    rack_b = _find(result.root, "rack-b")
    assert (rack_b.workers, rack_b.gpus) == (1, 2)


@pytest.mark.asyncio
async def test_free_gpus_excludes_what_is_already_allocated():
    """ "Can my 2P2D fit in that rack" is the question, so an allocated card is
    not free. Allocation comes from the model-instance bindings, the same
    source the scheduler reads."""
    workers = [_worker(1, "w1", {RACK: "rack-a"}, gpus=4)]
    result = await _preview(
        workers,
        saved=_topology([{"name": "Rack", "labelKeys": [RACK]}]),
        allocated={1: {0: 1024, 1: 2048}},
    )
    rack = _find(result.root, "rack-a")
    assert rack.gpus == 4
    assert rack.free_gpus == 2


@pytest.mark.asyncio
async def test_a_zero_allocation_still_counts_as_free():
    """Present-and-zero is measured and empty; only a non-zero claim occupies
    a card."""
    result = await _preview(
        [_worker(1, "w1", gpus=2)],
        allocated={1: {0: 0, 1: 0}},
    )
    assert result.root.free_gpus == 2


@pytest.mark.asyncio
async def test_an_unreadable_allocation_counts_as_used_not_free():
    """🔴 The pessimistic direction is the only safe one. The preview answers
    "can my group fit here", and an optimistic guess is the one answer that
    sends an operator to a rack that cannot take the group. Measured upstream:
    one missing global config made every worker report zero."""
    result = await _preview([_worker(1, "w1", gpus=4)], allocated={1: "raise"})
    assert result.root.gpus == 4
    assert result.root.free_gpus == 0


# --- the unclassified bucket ------------------------------------------------ #


@pytest.mark.asyncio
async def test_an_unlabelled_worker_lands_in_a_bucket_that_names_itself():
    workers = [
        _worker(1, "w1", {RACK: "rack-a"}),
        _worker(2, "w2"),
        _worker(3, "w3"),
    ]
    result = await _preview(
        workers, saved=_topology([{"name": "Rack", "labelKeys": [RACK]}])
    )

    bucket = _find(result.root, UNCLASSIFIED)
    assert bucket.unclassified is True
    assert bucket.workers == 2
    # The payload that turns "2 workers are missing this label" into one bulk
    # action instead of a hunt.
    assert sorted(bucket.worker_ids) == [2, 3]
    # And which key they are missing.
    assert result.label_keys["Rack"] == [RACK]


@pytest.mark.asyncio
async def test_unclassified_workers_are_deduplicated_across_layers():
    """One worker missing two labels is one worker to go and label."""
    workers = [_worker(1, "w1"), _worker(2, "w2", {ZONE: "z1"})]
    result = await _preview(
        workers,
        saved=_topology(
            [
                {"name": "Zone", "labelKeys": [ZONE]},
                {"name": "Rack", "labelKeys": [RACK], "parentLayer": "Zone"},
            ]
        ),
    )
    # w1 is unclassified at Zone *and* at Rack; w2 only at Rack.
    assert result.unclassified_workers == 2


@pytest.mark.asyncio
async def test_a_blank_label_value_is_absent_not_a_domain():
    """An empty label is a labelling accident; treating it as a domain name
    would gather every half-labelled worker into one bogus domain."""
    result = await _preview(
        [_worker(1, "w1", {RACK: ""})],
        saved=_topology([{"name": "Rack", "labelKeys": [RACK]}]),
    )
    assert _find(result.root, UNCLASSIFIED) is not None


# --- any-of ----------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_matching_key_is_reported_so_a_mixed_fleet_is_readable():
    """any-of is the feature that lets a mixed fleet work without relabelling;
    without reporting the winner the operator cannot tell which spelling
    applied."""
    workers = [
        _worker(1, "w1", {RACK: "rack-a"}),
        _worker(2, "w2", {ZONE: "rack-b"}),
    ]
    result = await _preview(
        workers, saved=_topology([{"name": "Rack", "labelKeys": [RACK, ZONE]}])
    )
    assert _find(result.root, "rack-a").matched_label_key == RACK
    assert _find(result.root, "rack-b").matched_label_key == ZONE


@pytest.mark.asyncio
async def test_the_first_declared_key_wins_when_both_are_present():
    result = await _preview(
        [_worker(1, "w1", {RACK: "from-rack", ZONE: "from-zone"})],
        saved=_topology([{"name": "Rack", "labelKeys": [RACK, ZONE]}]),
    )
    assert _find(result.root, "from-rack") is not None
    assert _find(result.root, "from-zone") is None


# --- the body wins, which is why this is a POST ----------------------------- #


@pytest.mark.asyncio
async def test_an_unsaved_declaration_in_the_body_overrides_the_saved_one():
    """🔑 The reason the endpoint exists. Editing a labelKey has to redraw the
    tree *before* saving; making the operator commit to find out turns a
    keystroke into a two-step wizard against a live cluster."""
    workers = [_worker(1, "w1", {RACK: "rack-a", ZONE: "zone-1"})]
    saved = _topology([{"name": "Rack", "labelKeys": [RACK]}])
    candidate = _topology([{"name": "Zone", "labelKeys": [ZONE]}])

    result = await _preview(workers, saved=saved, body=candidate)

    assert result.layers == ["Zone", NODE_LAYER]
    assert _find(result.root, "zone-1") is not None
    assert _find(result.root, "rack-a") is None


@pytest.mark.asyncio
async def test_no_body_falls_back_to_the_saved_declaration():
    """So the page's first render needs no special case."""
    result = await _preview(
        [_worker(1, "w1", {RACK: "rack-a"})],
        saved=_topology([{"name": "Rack", "labelKeys": [RACK]}]),
    )
    assert result.layers == ["Rack", NODE_LAYER]


# --- refusals: the declaration only, never the data ------------------------- #


@pytest.mark.asyncio
async def test_a_declaration_that_cannot_become_a_tree_is_refused():
    """A fork has no single answer to "how many layers up", so the intent is
    unknowable."""
    with pytest.raises(BadRequestException):
        await _preview(
            [_worker(1, "w1")],
            body=_topology(
                [
                    {"name": "Rack", "labelKeys": [RACK]},
                    {"name": "Zone", "labelKeys": [ZONE]},
                ]
            ),
        )


@pytest.mark.asyncio
async def test_missing_labels_are_never_a_refusal():
    """The tree is *how* you find out who still needs labelling, so requiring
    labels to look at it would be backwards."""
    result = await _preview(
        [_worker(1, "w1"), _worker(2, "w2")],
        saved=_topology([{"name": "Rack", "labelKeys": [RACK]}]),
    )
    assert result.unclassified_workers == 2


# --- payload discipline ----------------------------------------------------- #


@pytest.mark.asyncio
async def test_worker_ids_are_carried_only_where_they_are_acted_on():
    """Every other domain's membership is implied by its children; shipping it
    everywhere would be a large list nobody reads."""
    workers = [_worker(1, "w1", {RACK: "rack-a"}), _worker(2, "w2")]
    result = await _preview(
        workers, saved=_topology([{"name": "Rack", "labelKeys": [RACK]}])
    )

    assert result.root.worker_ids == []
    assert _find(result.root, "rack-a").worker_ids == []
    assert _find(result.root, "w1").worker_ids == [1]
    assert _find(result.root, UNCLASSIFIED).worker_ids == [2]
