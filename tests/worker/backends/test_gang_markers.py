"""Kueue pod-group markers on a disaggregated group's members.

Without them each member queues on its own, and a group can sit half
admitted: two of four members holding cards while the other two wait for cards
the first two are occupying. That is not a slow start — it is a deadlock that
resolves only when someone deletes the deployment.

The pair is what matters. A group name without a total count leaves Kueue
waiting for a membership it can never confirm, so half the pair is worse than
neither, and the code refuses to emit it.
"""

from types import SimpleNamespace

from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.worker.backends.base import InferenceServer

GROUP = "7-abc123"


def _model(prefill=2, decode=2):
    return Model(
        id=7,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        roles=[
            RoleSpec(name="prefill", replicas=prefill),
            RoleSpec(name="decode", replicas=decode),
            RoleSpec(name="router", replicas=1),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )


class _Plan:
    """A workload plan that can carry annotations."""

    def __init__(self):
        self.labels = {}
        self.annotations = {}


class _OldPlan:
    """A plan from a runtime that predates annotation support."""

    def __init__(self):
        self.labels = {}


def _apply(plan, role, model=None, group_id=GROUP):
    model = model or _model()
    fake = SimpleNamespace(
        _model=model,
        _model_spec=model,
        _model_instance=SimpleNamespace(role=role, group_id=group_id, name="m-x"),
    )
    InferenceServer._apply_gang_markers(fake, plan)
    return plan


def test_a_gpu_member_carries_both_marks():
    plan = _apply(_Plan(), "prefill")

    assert plan.labels["kueue.x-k8s.io/pod-group-name"] == GROUP
    assert plan.annotations["kueue.x-k8s.io/pod-group-total-count"] == "4"


def test_the_count_covers_only_the_gpu_roles():
    """D11: the router takes no accelerator and is created after its peers are
    already running, so counting it declares a membership that cannot be
    reached."""
    plan = _apply(_Plan(), "decode", model=_model(prefill=3, decode=1))

    assert plan.annotations["kueue.x-k8s.io/pod-group-total-count"] == "4"


def test_the_router_is_not_in_the_gang():
    plan = _apply(_Plan(), "router")

    assert plan.labels == {}
    assert plan.annotations == {}


def test_every_member_of_one_group_agrees_on_the_name_and_count():
    """Kueue joins them on exactly these two values; a disagreement is a group
    that never completes."""
    prefill = _apply(_Plan(), "prefill")
    decode = _apply(_Plan(), "decode")

    assert prefill.labels == decode.labels
    assert prefill.annotations == decode.annotations


def test_a_role_less_deployment_is_untouched():
    plain = Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
    )
    plan = _apply(_Plan(), None, model=plain, group_id=None)

    assert plan.labels == {}
    assert plan.annotations == {}


def test_an_old_runtime_gets_neither_mark_rather_than_one():
    """Half the pair is the one outcome worse than none: Kueue would hold the
    group forever waiting to learn how big it is."""
    plan = _apply(_OldPlan(), "prefill")

    assert plan.labels == {}
    assert not hasattr(plan, "annotations")


def test_existing_labels_and_annotations_survive():
    plan = _Plan()
    plan.labels = {"mine": "1"}
    plan.annotations = {"mine": "2"}
    _apply(plan, "prefill")

    assert plan.labels["mine"] == "1"
    assert plan.annotations["mine"] == "2"
    assert "kueue.x-k8s.io/pod-group-name" in plan.labels
