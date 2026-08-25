"""One `--kv-transfer-config`, however many things want to write it.

Three sources want that flag — the PD mode's connector, the extended KV cache
(local LMCache or a cache service's injected snapshot), and whatever the user
typed in `backend_parameters` — and vLLM reads it once. Until this module the
answer was to forbid the overlap, which made a disaggregated deployment and a
shared KV cache mutually exclusive.

That was never an engine limit. vLLM ships a `MultiConnector` that composes
child connectors under one flag, listing them in
`kv_connector_extra_config.connectors`, and its documented semantics are
*load from the first connector advertising the tokens, save to all*. So the
two are not in conflict; they are complementary, and the composition is
GPUStack's to assemble.

**Order is priority, and it differs per role** — which is the whole reason
this is not a set union:

- **prefill** gets `[cache, pd]`. Ask the shared cache first: a prefix it
  already holds is prefill work that does not have to happen at all, and what
  is left is the part disaggregation exists to optimise. "Save to all" then
  writes the computed KV back to the cache *and* hands it to decode.
- **decode** gets `[pd, cache]`. The request already carries
  `do_remote_prefill`, so its KV is known to be waiting on the prefill side;
  the cache is the fallback, not the first question.

Getting that backwards is not a performance detail. A decode that asks a
cache first can be served a stale or partial entry ahead of the KV its own
prefill just computed for it.

Deliberately a post-pass over the assembled argv rather than a rewrite of
either producer. The PD injector renders its descriptor from the mode
catalog, and a cache provider renders its own arguments through a contract
this module has no business reaching into; both keep emitting exactly what
they emit today, and a deployment with only one of them comes out of here
byte-identical.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

KV_TRANSFER_CONFIG_FLAG = "--kv-transfer-config"

MULTI_CONNECTOR = "MultiConnector"

# The composite both produces and consumes: it carries a producer (the PD
# connector on a prefill) and a store that is read and written (the cache).
# The child connectors keep their own roles, which is what the engine acts on.
_COMPOSITE_ROLE = "kv_both"

# Which side of the pair asks the cache first. Anything else — a router, a
# role-less deployment that somehow reached here — keeps the order the
# arguments arrived in, because inventing a priority for a role whose
# semantics are unknown is how a wrong answer gets served confidently.
_CACHE_FIRST_ROLES = frozenset({"prefill"})


def _parse(value: str, *, where: str) -> Optional[Dict[str, Any]]:
    """A descriptor, or None if it is not one this module can compose.

    Unparseable is not an error here. The value may be a user's own, in a
    shape GPUStack never generated, and refusing to launch over it would be
    worse than leaving it exactly as it was — which is what returning None
    does.
    """
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        logger.debug("Not composing %s: %s is not JSON", where, KV_TRANSFER_CONFIG_FLAG)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _split_flags(arguments: List[str]) -> Tuple[List[int], List[str]]:
    """Positions of the flag and the raw value that follows each.

    A flag at the very end has no value and is left alone rather than being
    treated as an empty descriptor: it is malformed input, and the engine's
    own parser reports that far better than this module could.
    """
    positions: List[int] = []
    values: List[str] = []
    index = 0
    while index < len(arguments) - 1:
        if arguments[index] == KV_TRANSFER_CONFIG_FLAG:
            positions.append(index)
            values.append(arguments[index + 1])
            index += 2
            continue
        index += 1
    return positions, values


def compose_descriptors(
    descriptors: List[Dict[str, Any]],
    role: Optional[str],
) -> Dict[str, Any]:
    """Fold several connector descriptors into one `MultiConnector`.

    `descriptors` arrives cache-first; a role that wants the opposite gets it
    reversed here, so the caller never has to know the rule.
    """
    ordered = list(descriptors)
    if role not in _CACHE_FIRST_ROLES:
        ordered.reverse()
    return {
        "kv_connector": MULTI_CONNECTOR,
        "kv_role": _COMPOSITE_ROLE,
        "kv_connector_extra_config": {"connectors": ordered},
    }


def compose_kv_transfer_config(
    arguments: List[str],
    role: Optional[str],
    *,
    cache_first: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Collapse every `--kv-transfer-config` in `arguments` into one.

    Args:
        arguments: The assembled argv. Returned unchanged when it carries
            fewer than two of the flag, which is every deployment that uses
            disaggregation or a KV cache but not both.
        role: The PD role this member serves, which decides the order.
        cache_first: The descriptor that must lead for a cache-first role,
            when the caller knows which one came from the cache. Without it
            the arguments' own order is taken as cache-first, which is the
            order they are assembled in.

    Returns:
        A new argv with one flag, or the original list object when there was
        nothing to do — callers rely on the untouched case being untouched.
    """
    positions, values = _split_flags(arguments)
    if len(positions) < 2:
        return arguments

    descriptors: List[Dict[str, Any]] = []
    for value in values:
        parsed = _parse(value, where=f"role '{role}'")
        if parsed is None:
            # One value nobody can read makes the whole composition a guess.
            # Leaving every flag in place hands the engine a duplicate it will
            # reject or resolve by its own rule — visible either way, which is
            # better than a silently dropped connector.
            logger.warning(
                "Leaving %d %s arguments unmerged for role '%s': one of them "
                "is not a descriptor this can compose.",
                len(positions),
                KV_TRANSFER_CONFIG_FLAG,
                role,
            )
            return arguments
        descriptors.append(parsed)

    if cache_first is not None:
        # Order by origin rather than by position: the two producers append at
        # different points in the build, and reading the order off the argv
        # would make the composition depend on where each happened to land.
        rest = [d for d in descriptors if d != cache_first]
        descriptors = [cache_first] + rest

    composed = compose_descriptors(descriptors, role)
    logger.info(
        "Composed %d KV connectors into a %s for role '%s': %s",
        len(descriptors),
        MULTI_CONNECTOR,
        role,
        " then ".join(
            str(d.get("kv_connector", "?"))
            for d in composed["kv_connector_extra_config"]["connectors"]
        ),
    )

    encoded = json.dumps(composed, separators=(",", ":"))
    result: List[str] = []
    drop = set()
    for position in positions:
        drop.add(position)
        drop.add(position + 1)
    first = positions[0]
    for index, token in enumerate(arguments):
        if index == first:
            result.append(KV_TRANSFER_CONFIG_FLAG)
            result.append(encoded)
            continue
        if index in drop:
            continue
        result.append(token)
    return result


def descriptor_in(arguments: List[str]) -> Optional[Dict[str, Any]]:
    """The single descriptor these arguments carry, if exactly one do.

    Used to tell the cache's contribution apart from the PD injector's: the
    caller renders the cache branch on its own and hands the result here.
    """
    positions, values = _split_flags(arguments)
    if len(positions) != 1:
        return None
    return _parse(values[0], where="cache arguments")
