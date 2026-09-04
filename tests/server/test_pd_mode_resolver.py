from gpustack.schemas.models import BackendEnum, PDModeEnum
from gpustack.server.pd_mode_resolver import resolve_pd_mode

VLLM = BackendEnum.VLLM.value
SGLANG = BackendEnum.SGLANG.value


def verdict(resolution, name):
    return next(option for option in resolution.options if option.name == name)


# ---- the three cells phase 1 ships ---------------------------------------


def test_vllm_on_nvidia_has_one_candidate_and_needs_no_question():
    resolution = resolve_pd_mode(VLLM, {"nvidia"})
    assert resolution.mode == PDModeEnum.VLLM_NIXL.value
    assert resolution.vendor == "nvidia"
    assert resolution.unresolved_reason is None
    assert verdict(resolution, PDModeEnum.VLLM_NIXL.value).recommended is True


def test_vllm_on_ascend_resolves_to_the_ascend_recipe():
    resolution = resolve_pd_mode(VLLM, {"ascend"})
    assert resolution.mode == PDModeEnum.VLLM_ASCEND_MOONCAKE.value
    assert resolution.vendor == "ascend"


def test_sglang_on_nvidia_is_the_only_ambiguous_cell_and_preferred_settles_it():
    """Two recipes fit, so this is the one cell where a tie-break is needed.
    Keeping it in the catalog (`preferred`) rather than here means neither the
    API nor the UI holds a "which one is better" rule."""
    resolution = resolve_pd_mode(SGLANG, {"nvidia"})
    assert resolution.mode == PDModeEnum.SGLANG_MOONCAKE.value
    assert verdict(resolution, PDModeEnum.SGLANG_NIXL.value).eligible is True
    assert verdict(resolution, PDModeEnum.SGLANG_NIXL.value).recommended is False


# ---- the cells it does not ship ------------------------------------------


def test_sglang_on_ascend_has_no_recipe_and_points_at_custom():
    resolution = resolve_pd_mode(SGLANG, {"ascend"})
    assert resolution.mode is None
    assert "custom" in resolution.unresolved_reason
    assert verdict(resolution, PDModeEnum.CUSTOM.value).eligible is True


def test_amd_has_no_recipe_and_points_at_custom():
    """Phase 1 ships no AMD recipe. Before `gpu_filters` the three NVIDIA
    recipes were unconstrained, so they were offered here."""
    resolution = resolve_pd_mode(VLLM, {"amd"})
    assert resolution.mode is None
    assert verdict(resolution, PDModeEnum.VLLM_NIXL.value).eligible is False
    assert "nvidia" in verdict(resolution, PDModeEnum.VLLM_NIXL.value).ineligible_reason


def test_custom_is_eligible_on_every_accelerator():
    """🔴 The load-bearing invariant: "no built-in recipe" must never become
    "no PD"."""
    for vendors in ({"nvidia"}, {"ascend"}, {"amd"}, {"metax"}, set()):
        for backend in (VLLM, SGLANG, "SomeBYOEngine", None):
            resolution = resolve_pd_mode(backend, vendors)
            assert verdict(resolution, PDModeEnum.CUSTOM.value).eligible is True


def test_custom_is_never_the_derived_answer():
    """It injects nothing, so choosing it is a decision about who supplies the
    connection parameters -- not something the platform can make for you."""
    for vendors in ({"nvidia"}, {"ascend"}, {"amd"}, set()):
        resolution = resolve_pd_mode(VLLM, vendors)
        assert resolution.mode != PDModeEnum.CUSTOM.value


# ---- mixed-vendor clusters ------------------------------------------------


def test_a_mixed_cluster_returns_the_candidates_and_no_answer():
    """A PD group cannot span vendors, so this is a placement question. The
    platform deliberately does not pick the larger partition: the user may
    want the idle one."""
    resolution = resolve_pd_mode(VLLM, {"nvidia", "ascend"})
    assert resolution.mode is None
    assert sorted(resolution.candidate_vendors) == ["ascend", "nvidia"]
    assert "cannot span vendors" in resolution.unresolved_reason


def test_choosing_a_vendor_settles_a_mixed_cluster():
    resolution = resolve_pd_mode(VLLM, {"nvidia", "ascend"}, vendor="ascend")
    assert resolution.mode == PDModeEnum.VLLM_ASCEND_MOONCAKE.value
    assert resolution.vendor == "ascend"
    # The NVIDIA recipe is out relative to the *chosen partition*, not the
    # whole cluster -- that is the difference from the unresolved case above.
    assert verdict(resolution, PDModeEnum.VLLM_NIXL.value).eligible is False


def test_a_mixed_cluster_with_only_one_usable_partition_still_resolves():
    """SGLang has no Ascend recipe, so a mixed cluster is unambiguous for it:
    only the NVIDIA partition can host the group."""
    resolution = resolve_pd_mode(SGLANG, {"nvidia", "ascend"})
    assert resolution.mode == PDModeEnum.SGLANG_MOONCAKE.value
    assert resolution.candidate_vendors == ["nvidia"]


def test_choosing_a_vendor_the_cluster_lacks_is_refused():
    resolution = resolve_pd_mode(VLLM, {"nvidia"}, vendor="ascend")
    assert resolution.mode is None
    assert "no ascend accelerator" in resolution.unresolved_reason


# ---- unknown accelerators -------------------------------------------------


def test_unknown_accelerators_decide_nothing_and_disable_nothing():
    """A cluster whose workers have not reported devices yet reads the same as
    one with none. Neither may be judged unable to run anything -- absence of
    evidence is not a mismatch, so scheduling gets the call."""
    resolution = resolve_pd_mode(VLLM, set())
    assert resolution.mode is None
    assert resolution.cluster_vendors == []
    for option in resolution.options:
        if VLLM in (get_backends(option.name) or [VLLM]):
            assert option.eligible is True


def get_backends(name):
    from gpustack.server.pd_mode_catalog import get_pd_mode

    mode = get_pd_mode(name)
    return mode.backends if mode else None


# ---- engine mismatch is reported, not hidden ------------------------------


def test_engine_mismatch_carries_its_reason():
    resolution = resolve_pd_mode(VLLM, {"nvidia"})
    sglang = verdict(resolution, PDModeEnum.SGLANG_MOONCAKE.value)
    assert sglang.eligible is False
    assert "SGLang" in sglang.ineligible_reason
    assert "custom" in sglang.ineligible_reason
