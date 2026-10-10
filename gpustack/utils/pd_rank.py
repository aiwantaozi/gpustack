"""Rank identities for PD routers with one generation endpoint per instance."""

import re
from typing import Optional, Tuple

from gpustack.schemas.models import Model, find_role, role_effective_model
from gpustack.utils.command import find_bool_parameter, find_last_int_parameter


SGLANG_DP_ROUTER_VERSION = '0.3.2+gpustack.dp1'
_RANK_URL = re.compile(r'^(https?://[^/@]+)@([0-9]+)$')


def split_rank_url(url: str) -> Tuple[str, Optional[int]]:
    """Separate a logical worker rank from its physical HTTP endpoint."""
    match = _RANK_URL.fullmatch(url)
    return (match[1], int(match[2])) if match else (url, None)


def sglang_dp_aware(model: Model) -> bool:
    """Whether the managed SGLang router explicitly enables rank routing."""
    mode = getattr(getattr(model, 'disaggregation', None), 'mode', None)
    if str(mode) not in ('sglang-mooncake', 'sglang-nixl'):
        return False
    if not getattr(model, 'roles', None):
        return False
    router = find_role(model, 'router')
    return bool(
        router
        and not router.run_command
        and find_bool_parameter(router.backend_parameters or [], ['dp-aware'])
    )


def sglang_request_rank_count(model: Model, role: str) -> int:
    """Read the request-rank count for the supported SGLang 0.5.18 layout.

    Ordinary DP and attention DP both use dp_size in this engine version.
    TP, PP and the connector's port count do not change this namespace.
    """
    effective = role_effective_model(model, role)
    parameters = effective.backend_parameters or []
    attn_dp = find_last_int_parameter(parameters, ['attn-dp-size'])
    if attn_dp is not None:
        raise ValueError(
            'SGLang PD rank routing currently supports the dp-size layout; '
            'attn-dp-size requires engine and scheduler integration'
        )
    size = find_last_int_parameter(parameters, ['dp-size', 'data-parallel-size', 'dp'])
    if size is not None and size < 1:
        raise ValueError('SGLang PD dp-size must be positive')
    return size if size is not None else 1
