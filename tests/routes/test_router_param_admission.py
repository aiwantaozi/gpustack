"""A router parameter that would collide with an injected one.

The router's tunable flags are meant to be overridden — appending them is
last-wins, verified against both shipped wheels. The connection flags are not,
and refusing them is not tidiness:

- ``--prefill`` / ``--decode`` are ``action="append"`` in both routers, so a
  second one does not replace the injected peer. It adds one the router then
  forwards to and cannot reach, and the only symptom is a member that quietly
  never gets traffic.
- ``--host`` / ``--port`` / ``--prometheus-*`` are last-wins, which is worse in
  a different way: the router comes up bound somewhere the gateway and the
  metrics scraper are not looking.
"""

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import _reject_router_params_the_platform_owns
from gpustack.schemas.models import DisaggregationSpec, PDModeEnum, RoleSpec


def _roles(router_params):
    return [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1, backend_parameters=router_params),
    ]


def _spec(mode=PDModeEnum.VLLM_NIXL):
    return DisaggregationSpec(mode=mode)


def test_a_router_with_no_parameters_of_its_own_is_untouched():
    _reject_router_params_the_platform_owns(_roles(None), _spec())
    _reject_router_params_the_platform_owns(_roles([]), _spec())


@pytest.mark.parametrize(
    "param",
    [
        # Appends rather than replaces: a phantom peer the router forwards to.
        "--prefill",
        "--decode",
        # Last-wins: the router binds where nothing is looking for it.
        "--host",
        "--port",
        "--prometheus-port",
        # The transport handshake — the recipe's whole subject.
        "--kv-connector",
        "--vllm-pd-disaggregation",
    ],
)
def test_a_flag_the_platform_renders_is_refused(param):
    with pytest.raises(BadRequestException) as excinfo:
        _reject_router_params_the_platform_owns(_roles([param, "x"]), _spec())
    # The message has to name the flag and offer the alternative, or the user
    # is left guessing which of their parameters was the problem.
    assert param in excinfo.value.message
    assert "--prefill-policy" in excinfo.value.message


def test_both_spellings_of_a_parameter_are_caught():
    """`--flag=value` is as valid on a command line as `--flag value`, and a
    check that only splits on whitespace lets the first one through."""
    for spelling in ("--host=1.2.3.4", "--port=9999"):
        with pytest.raises(BadRequestException):
            _reject_router_params_the_platform_owns(_roles([spelling]), _spec())


@pytest.mark.parametrize(
    "param",
    ["--prefill-policy", "--decode-policy", "--cb-failure-threshold"],
)
def test_a_tunable_flag_is_accepted(param):
    """The point of the split. `--prefill-policy` shares a prefix with the
    refused `--prefill`, so a check written as a prefix match would forbid
    exactly the flag the feature exists to allow."""
    _reject_router_params_the_platform_owns(_roles([param, "cache_aware"]), _spec())


def test_a_flag_the_platform_does_not_own_is_accepted():
    """Anything the recipe never mentions is the user's business — the refusal
    list is what the catalog declares, not an allowlist of known flags."""
    _reject_router_params_the_platform_owns(
        _roles(["--shutdown-grace-period-secs", "30"]), _spec()
    )


def test_the_refused_set_follows_the_mode():
    """Read off the chosen recipe rather than a list in Python, so the two
    SGLang modes refuse their own router's flags and not vLLM's."""
    # `--kv-connector` is a vLLM-router flag; the SGLang recipes never pass it,
    # so there is nothing of ours for a user value to collide with.
    _reject_router_params_the_platform_owns(
        _roles(["--kv-connector", "nixl"]), _spec(PDModeEnum.SGLANG_MOONCAKE)
    )
    # `--host` is injected by every recipe.
    with pytest.raises(BadRequestException):
        _reject_router_params_the_platform_owns(
            _roles(["--host", "1.2.3.4"]), _spec(PDModeEnum.SGLANG_MOONCAKE)
        )


def test_a_hand_written_mode_refuses_nothing():
    """`custom` injects no connection state at all, so every flag is the
    user's — refusing one would be refusing a value nothing else supplies."""
    _reject_router_params_the_platform_owns(
        _roles(["--host", "1.2.3.4"]), _spec(PDModeEnum.CUSTOM)
    )
