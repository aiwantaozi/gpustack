"""Whether this model's KV will fit through the network in the time available.

The judgement PD disaggregation most needs before deployment is not "does the
link work" but "is it fast enough to be worth it", and those have opposite
failure modes: a link that is merely slow still passes every connectivity check
and then makes TTFT worse than not disaggregating at all. Measured on a 2.5GbE
pair: 0.293 GB/s, which moves a 70B-class 4K prompt's KV in 4.6 seconds — an
order of magnitude slower than computing it from scratch.

**The whole point of this module is that it needs no measurement.** The usual
form of the question ("I have X GB/s, how long will the transfer take?") cannot
be asked until a cluster exists and a probe has run. Turned around it needs
nothing but the model's own config:

    required bandwidth = KV bytes / (TTFT budget - prefill time)

So a threshold can be shown the moment a model is selected, before any
deployment, and the eventual measured number is one more line against it rather
than a different feature.

Everything here is pure arithmetic over `ModelParameters`. Nothing reads a
worker, a network or a running instance.
"""

import enum
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Decimal, not binary. Every published link rate is decimal (a "25GbE" NIC
# carries 25 * 10^9 bits, not 2^34), and the whole output of this module is
# compared against link rates. Mixing the two here would put a silent 7% error
# straight into the comparison that decides "sufficient" or not.
GB = 1_000_000_000
GBPS_IN_BYTES = GB


class KVCacheDType(str, enum.Enum):
    """How many bytes one KV element occupies on the wire.

    Named separately from the model's compute dtype because they are set
    separately: `--kv-cache-dtype fp8` halves the transfer without changing
    weights, and it is the cheapest lever a user has for making PD viable at
    all. A module that read only `torch_dtype` would tell someone who had
    already pulled that lever that their link is twice as short as it is.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"

    @property
    def bytes_per_element(self) -> int:
        return {"fp32": 4, "fp16": 2, "fp8": 1}[self.value]


_DTYPE_ALIASES = {
    "float32": KVCacheDType.FP32,
    "fp32": KVCacheDType.FP32,
    "float16": KVCacheDType.FP16,
    "fp16": KVCacheDType.FP16,
    "half": KVCacheDType.FP16,
    "bfloat16": KVCacheDType.FP16,
    "bf16": KVCacheDType.FP16,
    "float8": KVCacheDType.FP8,
    "fp8": KVCacheDType.FP8,
    "fp8_e4m3": KVCacheDType.FP8,
    "fp8_e5m2": KVCacheDType.FP8,
}

# `auto` deliberately absent: it means "follow the model dtype", so mapping it
# to a size here would override the very value it defers to.
_KV_CACHE_DTYPE_FLAG = re.compile(
    r"--kv[-_]cache[-_]dtype[=\s]+([A-Za-z0-9_]+)", re.IGNORECASE
)


def parse_kv_cache_dtype(
    backend_parameters: Optional[Sequence[str]],
) -> Optional[KVCacheDType]:
    """The KV dtype a deployment has already chosen, if any.

    Read from the engine arguments rather than asked for again: the user who
    set `--kv-cache-dtype fp8` has answered this question, and asking twice
    invites the two answers to disagree.
    """
    if not backend_parameters:
        return None
    # Joined before matching, because the flag and its value are as likely to
    # be two list entries (`["--kv-cache-dtype", "fp8"]`) as one
    # (`["--kv-cache-dtype=fp8"]`) -- `backend_parameters` is a concatenated
    # argv, not a normalised map. Matching per entry silently misses the split
    # form, which is the one the UI produces.
    joined = " ".join(str(token) for token in backend_parameters)
    match = _KV_CACHE_DTYPE_FLAG.search(joined)
    if not match:
        return None
    return _DTYPE_ALIASES.get(match.group(1).lower())


def resolve_kv_cache_dtype(
    torch_dtype: Optional[str],
    backend_parameters: Optional[Sequence[str]] = None,
) -> Optional[KVCacheDType]:
    explicit = parse_kv_cache_dtype(backend_parameters)
    if explicit:
        return explicit
    if not torch_dtype:
        return None
    return _DTYPE_ALIASES.get(str(torch_dtype).lower())


@dataclass(frozen=True)
class KVFootprint:
    """How many bytes one request's KV occupies, and how that was derived.

    The derivation travels with the number because the number alone is not
    actionable: 1.34 GB tells a user nothing about what to change, while "GQA,
    8 kv heads, fp16" tells them that fp8 halves it and that an MLA model would
    cut it by another order of magnitude.
    """

    bytes_per_token: int
    seq_len: int
    kv_cache_dtype: KVCacheDType

    layers: int
    kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    """MLA only: the width of the compressed latent that is stored instead of
    per-head K and V."""

    @property
    def bytes_per_request(self) -> int:
        return self.bytes_per_token * self.seq_len


def kv_footprint(
    *,
    layers: int,
    seq_len: int,
    kv_cache_dtype: KVCacheDType,
    kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
) -> Optional[KVFootprint]:
    """One request's KV bytes, or None when the config does not say.

    Two shapes, and conflating them is the mistake this function exists to
    avoid:

      per-head (MHA/GQA/MQA)  2 * kv_heads * head_dim  -- K and V, separately
      latent (MLA)            kv_lora_rank + qk_rope_head_dim  -- ONE tensor

    The `2 *` does not belong to MLA. Applying it there doubles the number that
    is the entire reason DeepSeek-class models can be disaggregated over
    ordinary links, and doubling it is exactly the error that would turn a
    green verdict red.

    None rather than a guess: a wrong footprint produces a confident bandwidth
    threshold, and a threshold is the one output here that a user will act on.
    """
    if layers <= 0 or seq_len <= 0:
        return None

    element = kv_cache_dtype.bytes_per_element

    if kv_lora_rank:
        # MLA. `qk_rope_head_dim` rides alongside the latent and is part of
        # what gets stored, so it is added rather than ignored; models that do
        # not report it just contribute the latent.
        latent_dim = kv_lora_rank + (qk_rope_head_dim or 0)
        return KVFootprint(
            bytes_per_token=latent_dim * element * layers,
            seq_len=seq_len,
            kv_cache_dtype=kv_cache_dtype,
            layers=layers,
            latent_dim=latent_dim,
        )

    if not kv_heads or not head_dim:
        return None

    return KVFootprint(
        bytes_per_token=2 * kv_heads * head_dim * element * layers,
        seq_len=seq_len,
        kv_cache_dtype=kv_cache_dtype,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )


def from_model_parameters(
    params,
    *,
    seq_len: int,
    backend_parameters: Optional[Sequence[str]] = None,
) -> Optional[KVFootprint]:
    """`kv_footprint` over the `ModelParameters` the scheduler already parses.

    Kept as an adapter rather than the primary entry point so the arithmetic
    stays testable without a pretrained config, which is the part of this that
    needs a network or a downloaded model.
    """
    dtype = resolve_kv_cache_dtype(
        getattr(params, "torch_dtype", None), backend_parameters
    )
    if dtype is None:
        return None
    return kv_footprint(
        layers=getattr(params, "num_hidden_layers", 0) or 0,
        seq_len=seq_len,
        kv_cache_dtype=dtype,
        kv_heads=getattr(params, "num_key_value_heads", None),
        head_dim=getattr(params, "head_dim", None),
        kv_lora_rank=getattr(params, "kv_lora_rank", None),
        qk_rope_head_dim=getattr(params, "qk_rope_head_dim", None),
    )


# The share of the TTFT budget left for the KV transfer when the caller does
# not say how long prefill takes. One third, because that is what the product
# copy promises ("KV transfer within a third of the TTFT budget") and because
# prefill time is the one term here that genuinely cannot be derived from the
# model config alone -- it depends on the accelerator.
DEFAULT_TRANSFER_SHARE = 1.0 / 3.0


def transfer_budget_seconds(
    ttft_budget_seconds: float,
    prefill_seconds: Optional[float] = None,
    transfer_share: float = DEFAULT_TRANSFER_SHARE,
) -> Optional[float]:
    """How much of the TTFT budget the transfer may consume.

    A measured or estimated `prefill_seconds` wins over the share: it is the
    real quantity, and the share is only standing in for it. Returns None when
    prefill already exceeds the budget -- there is no transfer time to
    allocate, and dividing by a negative would report a *negative* required
    bandwidth, which reads as "no requirement".
    """
    if ttft_budget_seconds <= 0:
        return None
    if prefill_seconds is None:
        if not 0 < transfer_share <= 1:
            return None
        return ttft_budget_seconds * transfer_share
    remaining = ttft_budget_seconds - prefill_seconds
    return remaining if remaining > 0 else None


def required_bandwidth(
    kv_bytes: int, transfer_seconds_available: float
) -> Optional[float]:
    """The reversed formula: bytes per second the link must sustain."""
    if kv_bytes <= 0 or transfer_seconds_available <= 0:
        return None
    return kv_bytes / transfer_seconds_available


def transfer_seconds(
    kv_bytes: int, bandwidth_bytes_per_second: float
) -> Optional[float]:
    if bandwidth_bytes_per_second <= 0:
        return None
    return kv_bytes / bandwidth_bytes_per_second


@dataclass(frozen=True)
class ReferenceLink:
    name: str
    bandwidth_bytes_per_second: float


# Effective throughput, not line rate. The 2.5GbE figure is measured (~94% of
# line rate on that pair); the rest are the conventional ~85-90% of line rate
# that TCP or RDMA actually delivers. Quoting line rate here would make links
# look sufficient at exactly the margin where they are not.
REFERENCE_LINKS: List[ReferenceLink] = [
    ReferenceLink("1GbE", 0.12 * GB),
    ReferenceLink("2.5GbE", 0.29 * GB),
    ReferenceLink("10GbE", 1.1 * GB),
    ReferenceLink("25GbE", 3.0 * GB),
    ReferenceLink("100GbE", 12.0 * GB),
    ReferenceLink("400G RDMA", 50.0 * GB),
]


class BandwidthVerdict(str, enum.Enum):
    SUFFICIENT = "sufficient"
    TIGHT = "tight"
    INSUFFICIENT = "insufficient"


# Expressed as a fraction of the requirement rather than absolute bandwidths so
# the same thresholds apply to every model: an MLA model and a 70B GQA model
# differ by 10x in what they need, and a fixed GB/s cutoff would be wrong for
# one of them whichever value it took.
TIGHT_RATIO = 0.3

REMEDIES = (
    "Switch the KV cache to fp8 (`--kv-cache-dtype fp8`), which halves the transfer.",
    "Choose an MLA-architecture model (DeepSeek-V3 class), whose KV is roughly "
    "an order of magnitude smaller than a GQA model of the same size.",
    "Put prefill and decode behind RDMA, or co-locate the group on one host.",
)


def verdict(ratio: float) -> BandwidthVerdict:
    """Where a measured-over-required ratio falls.

    Below 0.3 the transfer is at or beyond the same order as computing the
    prefill again, which is the point at which disaggregating costs more than
    it saves -- not a slower version of working.
    """
    if ratio >= 1.0:
        return BandwidthVerdict.SUFFICIENT
    if ratio >= TIGHT_RATIO:
        return BandwidthVerdict.TIGHT
    return BandwidthVerdict.INSUFFICIENT
