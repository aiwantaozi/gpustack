"""Telling Kueue that a group is finished, rather than merely short a member.

Kueue holds an admitted pod-group's quota while it waits for a replacement,
which is exactly what a group wants when one member crashes and exactly wrong
when the group is being retired: the quota is never released, the Pod sits in
Terminating behind Kueue's own finalizer, and the cards stay claimed by a
deployment that no longer exists.

The two are told apart by asking whether any sibling of the generation
survives, which needs no new field and leaves no way for the server and the
worker to disagree about what is being torn down.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpustack.worker import serve_manager as sm

MARKER = "kueue.x-k8s.io/retriable-in-group"


def _instance(id, group_id="g1", model_id=7):
    return SimpleNamespace(id=id, name=f"m-{id}", group_id=group_id, model_id=model_id)


def _manager(siblings):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=siblings)
    manager = SimpleNamespace(_clientset=clientset)
    return manager


def _call(mi, siblings, marker=MARKER):
    with patch.object(sm, "ANNOTATION_KUEUE_RETRIABLE_IN_GROUP", marker):
        return sm.ServeManager._gang_delete_annotations(_manager(siblings), mi)


def test_the_last_member_of_a_group_releases_the_quota():
    mi = _instance(1)
    assert _call(mi, [mi]) == {"annotations": {MARKER: "false"}}


def test_a_surviving_sibling_means_a_replacement():
    """One member out of three going away is a scale-down. Marking the group
    finished there would release a quota the remaining members are using."""
    mi = _instance(1)
    assert _call(mi, [mi, _instance(2), _instance(3)]) == {}


def test_a_member_of_another_generation_does_not_count():
    """Its group is a different group; it cannot keep this one alive."""
    mi = _instance(1, group_id="g1")
    assert _call(mi, [mi, _instance(9, group_id="g2")]) == {
        "annotations": {MARKER: "false"}
    }


def test_a_role_less_instance_is_never_marked():
    assert _call(_instance(1, group_id=None), []) == {}


def test_a_runtime_without_the_capability_takes_the_old_path():
    """Deleting still works; the quota release waits for the pin bump."""
    mi = _instance(1)
    assert _call(mi, [mi], marker=None) == {}


def test_a_failed_lookup_does_not_stop_the_delete():
    """Failing to look is not failing to delete. Leaving the annotation off
    risks a leaked quota; refusing to delete guarantees one."""
    clientset = MagicMock()
    clientset.model_instances.list.side_effect = RuntimeError("api down")
    manager = SimpleNamespace(_clientset=clientset)

    with patch.object(sm, "ANNOTATION_KUEUE_RETRIABLE_IN_GROUP", MARKER):
        assert sm.ServeManager._gang_delete_annotations(manager, _instance(1)) == {}


@pytest.mark.parametrize("surviving,expected", [(0, True), (1, False)])
def test_the_marker_reaches_delete_workload(surviving, expected):
    """The kwargs shape matters: a runtime without the parameter must never be
    handed it."""
    mi = _instance(1)
    siblings = [mi] + [_instance(2 + i) for i in range(surviving)]
    result = _call(mi, siblings)
    assert bool(result) is expected
