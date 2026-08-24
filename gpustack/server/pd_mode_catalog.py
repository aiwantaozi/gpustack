import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional

import yaml

from gpustack.schemas.models import PD_MODE_BACKENDS, PDModeEnum
from gpustack.schemas.pd_modes import PDKVLease, PDMode, PDModeCatalog

logger = logging.getLogger(__name__)

_ASSET_NAME = "pd-modes.yaml"

_catalog: Optional[PDModeCatalog] = None


class PDModeCatalogError(Exception):
    """A malformed or out-of-sync PD-mode catalog.

    Unlike the cache-provider catalog, a broken declaration here is not
    survivable by degrading to an empty catalog: a mode the user can select
    but the catalog cannot answer for is a silent table miss at deploy time
    — nothing injected, and a deployment that comes up looking healthy while
    serving aggregated. So this raises, and the start-up caller lets it fail
    the process.
    """


def load_pd_mode_catalog(reload: bool = False) -> PDModeCatalog:
    """
    Load and validate the declarative PD-mode catalog from the bundled
    asset. The catalog is read-only and cached for the process lifetime, so
    calling this on a request path costs a dict lookup after the first time.
    """
    global _catalog
    if _catalog is not None and not reload:
        return _catalog

    yaml_file = files("gpustack.assets").joinpath(_ASSET_NAME)
    if not yaml_file.is_file():
        raise PDModeCatalogError(f"{_ASSET_NAME} is missing from the installation")
    try:
        raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PDModeCatalogError(f"{_ASSET_NAME} is not valid YAML: {e}") from e

    _catalog = parse_pd_mode_catalog(raw)
    logger.debug(f"Loaded {len(_catalog.modes)} PD modes from {_ASSET_NAME}")
    return _catalog


def parse_pd_mode_catalog(raw: Any) -> PDModeCatalog:
    """
    Parse a catalog document into typed models, resolving each mode's
    kv_lease reference into the connector's declaration, then run the
    cross-source assertions. Split out from the asset read so the contract
    tests can drive it with a document.
    """
    if not isinstance(raw, dict):
        raise PDModeCatalogError(
            f"{_ASSET_NAME} must be a mapping with 'kv_leases' and 'modes' keys"
        )

    leases = _parse_kv_leases(raw.get("kv_leases"))
    modes: List[PDMode] = []
    for entry in raw.get("modes") or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(f"mode entries must be mappings, got {entry!r}")
        entry = dict(entry)
        name = entry.get("name")
        reference = entry.get("kv_lease")
        if isinstance(reference, str):
            lease = leases.get(reference)
            if lease is None:
                raise PDModeCatalogError(
                    f"mode '{name}' references kv_lease '{reference}', which is "
                    f"not declared. Declared: {sorted(leases)}"
                )
            entry["kv_lease"] = lease
        try:
            modes.append(PDMode(**entry))
        except Exception as e:
            raise PDModeCatalogError(f"mode '{name}' is invalid: {e}") from e

    _assert_names_match_enum(modes)
    _assert_backends_match_table(modes)
    _assert_expired_metric_agrees(modes)
    return PDModeCatalog(kv_leases=leases, modes=modes)


def _parse_kv_leases(raw: Any) -> Dict[str, PDKVLease]:
    leases: Dict[str, PDKVLease] = {}
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise PDModeCatalogError(
                f"kv_lease entries must be mappings, got {entry!r}"
            )
        try:
            lease = PDKVLease(**entry)
        except Exception as e:
            raise PDModeCatalogError(f"invalid kv_lease entry {entry!r}: {e}") from e
        if lease.connector in leases:
            raise PDModeCatalogError(f"duplicate kv_lease for '{lease.connector}'")
        leases[lease.connector] = lease
    return leases


def _assert_names_match_enum(modes: List[PDMode]) -> None:
    """The catalog's entry names must be exactly PDModeEnum's values.

    A name present on only one side does not raise anywhere near where it
    was written: the request validates against the enum, the injector then
    misses the table and injects nothing, and the deployment serves
    aggregated with no error anywhere. This has happened once already
    (`ascend-mooncake` vs `vllm-ascend-mooncake`), so it fails start-up
    instead.
    """
    declared = {mode.name for mode in modes}
    expected = {mode.value for mode in PDModeEnum}
    if declared == expected:
        return
    missing = sorted(expected - declared)
    unknown = sorted(declared - expected)
    raise PDModeCatalogError(
        f"{_ASSET_NAME} and PDModeEnum disagree — the catalog is looked up by "
        "mode name, so a mismatch is a silent table miss, not an error: "
        f"in the enum but missing from the catalog: {missing}; "
        f"in the catalog but not in the enum: {unknown}"
    )


def _assert_backends_match_table(modes: List[PDMode]) -> None:
    """Each entry's `backends` must agree with PD_MODE_BACKENDS.

    That table exists only so request validation does not have to read the
    catalog. This catalog is the authoritative declaration of which engines
    a recipe may be injected into, and this assertion is what keeps the copy
    honest — otherwise a mode added here could pass a validation the catalog
    itself would refuse.
    """
    disagreements = []
    for mode in modes:
        declared = sorted(mode.backends)
        expected = sorted(PD_MODE_BACKENDS.get(mode.name, []))
        if declared != expected:
            disagreements.append(
                f"'{mode.name}': catalog {declared} != table {expected}"
            )
    if disagreements:
        raise PDModeCatalogError(
            "PD_MODE_BACKENDS is a copy of this catalog's `backends` kept for "
            "request validation, and the two disagree: " + "; ".join(disagreements)
        )


def _assert_expired_metric_agrees(modes: List[PDMode]) -> None:
    """`router.capabilities.kv_expired_metric` is the boolean the metrics
    collector reads; the lease registry carries the metric's name. Two
    spellings of one fact, so they have to agree — Mooncake exports no
    counter, and a mode claiming otherwise would poll for a metric that
    never appears."""
    for mode in modes:
        if mode.router is None:
            continue
        claimed = mode.router.capabilities.kv_expired_metric
        available = mode.kv_lease is not None and bool(mode.kv_lease.expired_metric)
        if claimed != available:
            declared = mode.kv_lease.expired_metric if mode.kv_lease else None
            raise PDModeCatalogError(
                f"mode '{mode.name}' claims kv_expired_metric={claimed} but its "
                f"kv_lease declares expired_metric={declared}"
            )


def load_pd_modes(reload: bool = False) -> List[PDMode]:
    return load_pd_mode_catalog(reload=reload).modes


def get_pd_modes() -> List[PDMode]:
    return load_pd_mode_catalog().modes


def get_pd_mode(name: str) -> Optional[PDMode]:
    return load_pd_mode_catalog().mode(name)


def get_kv_leases() -> Dict[str, PDKVLease]:
    """Every connector's lease window, including connectors no shipped mode
    uses (MoRIIO). The diagnostics side reads this; a mode's own window is
    already resolved onto the mode."""
    return load_pd_mode_catalog().kv_leases


def get_kv_lease(connector: str) -> Optional[PDKVLease]:
    return load_pd_mode_catalog().kv_leases.get(connector)
