"""Derive which PD recipe a deployment gets, instead of asking the user.

Phase 1 ships recipes for three engine × accelerator cells (vLLM on NVIDIA,
vLLM on Ascend, SGLang on NVIDIA). Two of them have a single candidate and the
third has a declared `preferred`, so **no cell needs a question**: the answer
follows from the engine the user already picked and the accelerators the
cluster already has.

Three rules the shape of this module exists to enforce:

1. **Judgement is server-side.** The facts it needs -- which accelerators the
   cluster's ready workers report -- are not in the deploy form. A client that
   tried would be guessing from `provider` (Docker / Kubernetes), which is the
   infrastructure provider, not the vendor.
2. **A cell with no recipe is not a cell without PD.** `custom` injects
   nothing and stays eligible everywhere, so "unsupported" degrades to
   "write the connection parameters yourself", never to "PD unavailable".
3. **Never guess across vendors.** A PD group cannot straddle two vendors --
   the KV path differs (HCCL/MemFabric vs UCX/RDMA verbs) -- so a mixed
   cluster is a *placement* question, not a preference. When more than one
   vendor partition could host the group this returns the candidates and no
   answer, rather than picking the biggest one: the user may well want the
   idle Ascend partition even though NVIDIA has more cards.
"""

import logging
from typing import Dict, List, Optional, Set

from gpustack.schemas.pd_modes import PDMode
from gpustack.server.pd_mode_catalog import get_pd_modes
from gpustack.schemas.pd_mode_resolution import (
    PDModeEligibility,
    PDModeResolution,
)

logger = logging.getLogger(__name__)


def _mode_vendors(mode: PDMode) -> Set[str]:
    if mode.gpu_filters is None or not mode.gpu_filters.vendor:
        return set()
    return {vendor.lower() for vendor in mode.gpu_filters.vendor}


def _fits_backend(mode: PDMode, backend: Optional[str]) -> bool:
    if not mode.backends or not backend:
        return True
    return backend in mode.backends


def _built_in_candidates(
    modes: List[PDMode], backend: Optional[str], vendor: str
) -> List[PDMode]:
    """Recipes that fit this engine on this one accelerator vendor.

    Excludes the unconstrained recipe (`custom`): it fits everything, so
    counting it would make every cell look supported and no cell look
    ambiguous.
    """
    return [
        mode
        for mode in modes
        if _mode_vendors(mode)
        and vendor in _mode_vendors(mode)
        and _fits_backend(mode, backend)
    ]


def _pick(candidates: List[PDMode]) -> Optional[PDMode]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    preferred = [mode for mode in candidates if mode.preferred]
    if len(preferred) == 1:
        return preferred[0]
    # The catalog loader asserts one `preferred` per cell, so this is a
    # can't-happen. Refusing beats picking by list order, which would make the
    # derived answer depend on how the YAML happens to be sorted.
    logger.warning(
        "pd mode cell has %d candidates and %d preferred: %s",
        len(candidates),
        len(preferred),
        [mode.name for mode in candidates],
    )
    return None


def resolve_pd_mode(
    backend: Optional[str],
    cluster_vendors: Set[str],
    *,
    vendor: Optional[str] = None,
) -> PDModeResolution:
    """Resolve the recipe, and describe every option either way.

    `cluster_vendors` empty means unknown -- no worker has reported devices
    yet, or no cluster was picked. Nothing is narrowed on that: absence of
    evidence is not a mismatch, and scheduling will judge later.

    `vendor` is the caller's explicit accelerator choice, which is both the
    tie-break for a mixed cluster and a placement constraint on the group.
    """
    modes = get_pd_modes()
    cluster_vendors = {v.lower() for v in cluster_vendors if v}
    chosen = vendor.lower() if vendor else None

    if chosen and cluster_vendors and chosen not in cluster_vendors:
        return _resolution(
            modes,
            backend,
            effective_vendors=set(),
            cluster_vendors=cluster_vendors,
            mode=None,
            resolved_vendor=None,
            candidate_vendors=[],
            unresolved_reason=(
                f"This cluster reports no {chosen} accelerator "
                f"(it has: {', '.join(sorted(cluster_vendors))})."
            ),
        )

    search_vendors = {chosen} if chosen else cluster_vendors

    # Unknown accelerators: describe the catalog, decide nothing.
    if not search_vendors:
        return _resolution(
            modes,
            backend,
            effective_vendors=set(),
            cluster_vendors=cluster_vendors,
            mode=None,
            resolved_vendor=None,
            candidate_vendors=[],
            unresolved_reason=(
                "The cluster's accelerators are not known yet, so the recipe "
                "cannot be derived."
            ),
        )

    per_vendor: Dict[str, List[PDMode]] = {
        v: _built_in_candidates(modes, backend, v) for v in sorted(search_vendors)
    }
    candidate_vendors = [v for v, found in per_vendor.items() if found]

    if not candidate_vendors:
        return _resolution(
            modes,
            backend,
            effective_vendors=search_vendors,
            cluster_vendors=cluster_vendors,
            mode=None,
            resolved_vendor=chosen,
            candidate_vendors=[],
            unresolved_reason=(
                f"No built-in recipe covers {backend or 'this engine'} on "
                f"{', '.join(sorted(search_vendors))}. Use pd mode 'custom' "
                f"to supply the connection parameters yourself."
            ),
        )

    if len(candidate_vendors) > 1:
        # 🔴 Deliberately no tie-break. See rule 3 in the module docstring.
        return _resolution(
            modes,
            backend,
            effective_vendors=search_vendors,
            cluster_vendors=cluster_vendors,
            mode=None,
            resolved_vendor=None,
            candidate_vendors=candidate_vendors,
            unresolved_reason=(
                "This cluster has more than one accelerator vendor that could "
                f"host the group ({', '.join(candidate_vendors)}), and a PD "
                "group cannot span vendors. Pick one."
            ),
        )

    resolved_vendor = candidate_vendors[0]
    picked = _pick(per_vendor[resolved_vendor])
    return _resolution(
        modes,
        backend,
        effective_vendors={resolved_vendor},
        cluster_vendors=cluster_vendors,
        mode=picked.name if picked else None,
        resolved_vendor=resolved_vendor,
        candidate_vendors=candidate_vendors,
        unresolved_reason=(
            None if picked else "Several recipes fit and none is marked preferred."
        ),
    )


def _resolution(
    modes: List[PDMode],
    backend: Optional[str],
    *,
    effective_vendors: Set[str],
    cluster_vendors: Set[str],
    mode: Optional[str],
    resolved_vendor: Optional[str],
    candidate_vendors: List[str],
    unresolved_reason: Optional[str],
) -> PDModeResolution:
    """Attach the per-option verdicts to a decision already made.

    Ineligible options are described, not dropped: an option the user cannot
    pick still says the capability exists and what it would take to reach it.
    """
    options = []
    for entry in modes:
        wanted = _mode_vendors(entry)
        backend_ok = _fits_backend(entry, backend)
        vendor_ok = (
            not wanted or not effective_vendors or bool(wanted & effective_vendors)
        )

        reason = None
        if not backend_ok:
            reason = (
                f"Requires {' / '.join(entry.backends)}; the selected engine "
                f"is {backend}. Mixing engines across roles needs pd mode "
                f"'custom'."
            )
        elif not vendor_ok:
            reason = (
                f"Requires {' / '.join(sorted(wanted))} accelerators; "
                f"{'the chosen partition has' if resolved_vendor else 'this cluster reports'} "
                f"{', '.join(sorted(effective_vendors))}."
            )

        options.append(
            PDModeEligibility(
                name=entry.name,
                eligible=backend_ok and vendor_ok,
                recommended=entry.name == mode,
                ineligible_reason=reason,
            )
        )

    return PDModeResolution(
        mode=mode,
        vendor=resolved_vendor,
        unresolved_reason=unresolved_reason,
        candidate_vendors=candidate_vendors,
        cluster_vendors=sorted(cluster_vendors),
        options=options,
    )
