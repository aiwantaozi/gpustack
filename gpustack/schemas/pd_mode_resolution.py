from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class PDModeUnresolvedCode(str, Enum):
    """Why `mode` came back None, as something a UI can translate.

    🔑 The prose in `unresolved_reason` is English and assembled here, so a
    client that renders it verbatim puts an English sentence inside an
    otherwise localized form. The code plus `unresolved_params` is the same
    statement in a shape the client can look up in its own message catalog;
    the prose stays as the fallback for an older client and as the readable
    form in logs.
    """

    VENDOR_NOT_IN_CLUSTER = "vendor_not_in_cluster"
    """The caller asked for an accelerator this cluster does not report."""

    VENDORS_UNKNOWN = "vendors_unknown"
    """No worker has reported devices yet. Wait, do not offer `custom`."""

    NO_BUILT_IN_RECIPE = "no_built_in_recipe"
    """Nothing in the catalog covers this engine on these accelerators."""

    MULTIPLE_VENDORS = "multiple_vendors"
    """More than one vendor partition could host the group. Ask which."""

    NO_PREFERRED_RECIPE = "no_preferred_recipe"
    """Several recipes fit and the catalog marks none of them preferred."""


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
    """Why `mode` is None, in English prose. Three shapes, and they need
    different handling by the caller: the cluster's accelerators are not known
    yet (wait), no built-in recipe covers this pair (offer `custom`), or
    several vendor partitions could host the group (ask which).

    ⚠️ Not for a localized UI to render verbatim -- see `unresolved_code`.
    Kept as the fallback for a client that predates the code, and as the
    readable form in logs."""

    unresolved_code: Optional[PDModeUnresolvedCode] = None
    """The same reason, as a stable key the client translates itself."""

    unresolved_params: Optional[Dict[str, str]] = None
    """Interpolation values for the code's message. Pre-joined into display
    strings rather than sent as lists: the separator (`, ` between vendor
    names, ` / ` between engine names) is part of the sentence, and a client
    that had to reassemble it would be re-deriving a decision made here.

    The values themselves are identifiers -- vendor slugs, engine names --
    so they are the same in every language."""

    candidate_vendors: List[str] = []
    """Vendor partitions that could host the group. More than one means the
    caller must choose -- the platform deliberately does not pick the largest,
    because the user may want the idle partition rather than the big one."""

    cluster_vendors: List[str] = []
    """Everything the cluster's workers report. Empty means unknown."""

    options: List[PDModeEligibility] = []
