from typing import List, Optional

from pydantic import BaseModel


class PDModeEligibility(BaseModel):
    """One catalog entry's verdict for one engine × accelerator combination.

    Ineligible entries are returned with a reason rather than omitted: an
    option the user cannot pick still tells them the capability exists and
    what it would take to reach it. Dropping it reads as "this product has no
    Ascend support" instead of "this cluster has no Ascend card".
    """

    name: str
    eligible: bool
    recommended: bool = False
    """The derived answer. At most one entry carries it."""

    ineligible_reason: Optional[str] = None
    """Why this entry cannot be picked here. None when eligible."""


class PDModeResolution(BaseModel):
    """The derived recipe for one deployment, plus every option either way."""

    mode: Optional[str] = None
    """The recipe to use. None when the answer is a question rather than a
    value -- see `unresolved_reason`. Never `custom`: that one is an explicit
    user choice, not something the platform derives."""

    vendor: Optional[str] = None
    """The accelerator partition the group resolved onto. Doubles as a
    placement constraint: a PD group cannot span vendors."""

    unresolved_reason: Optional[str] = None
    """Why `mode` is None. Three shapes, and they need different handling by
    the caller: the cluster's accelerators are not known yet (wait), no
    built-in recipe covers this pair (offer `custom`), or several vendor
    partitions could host the group (ask which)."""

    candidate_vendors: List[str] = []
    """Vendor partitions that could host the group. More than one means the
    caller must choose -- the platform deliberately does not pick the largest,
    because the user may want the idle partition rather than the big one."""

    cluster_vendors: List[str] = []
    """Everything the cluster's workers report. Empty means unknown."""

    options: List[PDModeEligibility] = []
