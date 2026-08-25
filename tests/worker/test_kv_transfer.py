"""Composing several KV connectors into the one flag the engine reads.

The property that matters most here is the *order*, because getting it wrong
does not fail — it serves. `MultiConnector` loads from the first connector
that advertises the tokens and saves to all, so a decode that asks a shared
cache before its own prefill can be handed a stale or partial entry ahead of
the KV that prefill just computed for it, and answer 200 with plausible text.
"""

import json

import pytest

from gpustack.worker.kv_transfer import (
    KV_TRANSFER_CONFIG_FLAG,
    MULTI_CONNECTOR,
    compose_kv_transfer_config,
    descriptor_in,
)

CACHE = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
PD = {"kv_connector": "NixlConnector", "kv_role": "kv_producer"}


def _flag(descriptor):
    return [KV_TRANSFER_CONFIG_FLAG, json.dumps(descriptor, separators=(",", ":"))]


def _connectors(arguments):
    index = arguments.index(KV_TRANSFER_CONFIG_FLAG)
    composed = json.loads(arguments[index + 1])
    assert composed["kv_connector"] == MULTI_CONNECTOR
    return [
        c["kv_connector"] for c in composed["kv_connector_extra_config"]["connectors"]
    ]


# --- the order is the point ------------------------------------------------ #


def test_prefill_asks_the_cache_first():
    """A prefix the cache already holds is prefill work that does not have to
    happen at all; what is left is what disaggregation is optimising."""
    arguments = ["serve", *_flag(PD), "--x", *_flag(CACHE)]

    result = compose_kv_transfer_config(arguments, "prefill", cache_first=CACHE)

    assert _connectors(result) == ["LMCacheConnectorV1", "NixlConnector"]


def test_decode_asks_its_own_prefill_first():
    """The request already carries do_remote_prefill, so its KV is known to be
    waiting on the prefill side. The cache is the fallback, not the first
    question — reversed, a decode can be served a stale entry ahead of the KV
    computed for it and still answer 200."""
    arguments = ["serve", *_flag(PD), *_flag(CACHE)]

    result = compose_kv_transfer_config(arguments, "decode", cache_first=CACHE)

    assert _connectors(result) == ["NixlConnector", "LMCacheConnectorV1"]


def test_the_order_follows_origin_not_position():
    """The two producers append at different points in the build, so reading
    the order off the argv would make the composition depend on where each
    happened to land."""
    cache_first_in_argv = ["serve", *_flag(CACHE), *_flag(PD)]
    pd_first_in_argv = ["serve", *_flag(PD), *_flag(CACHE)]

    assert _connectors(
        compose_kv_transfer_config(cache_first_in_argv, "decode", cache_first=CACHE)
    ) == _connectors(
        compose_kv_transfer_config(pd_first_in_argv, "decode", cache_first=CACHE)
    )


def test_an_unknown_role_keeps_the_order_it_was_given():
    """Inventing a priority for a role whose semantics are unknown is how a
    wrong answer gets served confidently."""
    arguments = ["serve", *_flag(CACHE), *_flag(PD)]

    result = compose_kv_transfer_config(arguments, None, cache_first=CACHE)

    assert _connectors(result) == ["NixlConnector", "LMCacheConnectorV1"]


# --- the untouched cases have to stay untouched ---------------------------- #


@pytest.mark.parametrize("role", ["prefill", "decode", None])
def test_a_single_connector_is_returned_verbatim(role):
    """Every deployment that uses disaggregation OR a cache but not both must
    come out of here byte-identical — this runs on every vLLM launch."""
    arguments = ["serve", "--model", "m", *_flag(PD)]

    result = compose_kv_transfer_config(arguments, role)

    assert result is arguments


def test_no_connector_at_all_is_returned_verbatim():
    arguments = ["serve", "--model", "m"]

    assert compose_kv_transfer_config(arguments, "prefill") is arguments


def test_everything_around_the_flags_is_preserved_in_order():
    arguments = ["serve", "--a", "1", *_flag(PD), "--b", "2", *_flag(CACHE), "--c"]

    result = compose_kv_transfer_config(arguments, "prefill", cache_first=CACHE)

    assert [t for t in result if t in ("serve", "--a", "1", "--b", "2", "--c")] == [
        "serve",
        "--a",
        "1",
        "--b",
        "2",
        "--c",
    ]
    assert result.count(KV_TRANSFER_CONFIG_FLAG) == 1


# --- refusing to guess ----------------------------------------------------- #


def test_a_value_that_cannot_be_read_leaves_every_flag_alone():
    """One unreadable value makes the whole composition a guess. Leaving the
    duplicates hands the engine something it will reject or resolve by its own
    rule — visible either way, which beats a silently dropped connector."""
    arguments = ["serve", *_flag(PD), KV_TRANSFER_CONFIG_FLAG, "not json"]

    result = compose_kv_transfer_config(arguments, "prefill")

    assert result is arguments


def test_a_trailing_flag_with_no_value_is_left_alone():
    """Malformed input the engine's own parser reports far better."""
    arguments = ["serve", *_flag(PD), KV_TRANSFER_CONFIG_FLAG]

    assert compose_kv_transfer_config(arguments, "prefill") is arguments


# --- telling the cache's contribution apart -------------------------------- #


def test_the_cache_branch_yields_its_descriptor():
    assert descriptor_in(_flag(CACHE)) == CACHE


def test_a_cache_branch_that_contributed_nothing_yields_nothing():
    """A degraded shared cache injects no arguments, and then there is nothing
    to compose — the PD connector must be left exactly as it was."""
    assert descriptor_in([]) is None
