"""Prefill/decode pairing pre-checks.

The division of labour with the engine (X1 3.1) is what these tests encode.
Most handshake factors are hashed by the connector and rejected on contact, so
checking them here buys attribution rather than safety. Two are different, and
they are why this validation exists at all:

* `max_model_len` is checked by nothing. Measured at prefill 8192 / decode
  4096: the handshake passes, KV transfers, short prompts answer normally, and
  only a prompt above decode's window fails — at decode, after prefill has
  already computed it. The user believes the deployment serves 8192.
* a decode narrower than its prefill is asserted at run time, but surfaces as
  an `IndexError` inside decode rather than as a configuration error.
"""

from contextlib import contextmanager

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_role_pairing
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelCreate,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)


@contextmanager
def rejects(fragment):
    """The API's HTTPException carries its text on ``.message``, not on
    ``str()``, so ``pytest.raises(match=...)`` would match the empty string."""
    with pytest.raises(BadRequestException) as excinfo:
        yield
    assert fragment in excinfo.value.message, excinfo.value.message


def _model_in(roles=None, disaggregation=True, backend_parameters=None):
    return ModelCreate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend="vLLM",
        backend_parameters=backend_parameters,
        roles=roles,
        disaggregation=(
            DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL) if disaggregation else None
        ),
    )


def _roles(prefill_params=None, decode_params=None):
    return [
        RoleSpec(name="prefill", replicas=1, backend_parameters=prefill_params),
        RoleSpec(name="decode", replicas=1, backend_parameters=decode_params),
        RoleSpec(name="router", replicas=1, cpu_only=True),
    ]


# --- the one nothing else checks ------------------------------------------- #


def test_mismatched_context_length_is_refused():
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--max-model-len=8192"],
                    decode_params=["--max-model-len=4096"],
                )
            )
        )


def test_the_refusal_names_the_window_that_would_break():
    """The message has to carry the number, because the symptom the user would
    otherwise see is a 400 on a long prompt with nothing pointing here."""
    with rejects("4096"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--max-model-len", "8192"],
                    decode_params=["--max-model-len", "4096"],
                )
            )
        )


def test_sglangs_spelling_of_the_same_thing_is_caught():
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--context-length=8192"],
                    decode_params=["--context-length=4096"],
                )
            )
        )


def test_matching_context_lengths_pass():
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["--max-model-len=8192"],
                decode_params=["--max-model-len=8192"],
            )
        )
    )


# --- tensor parallelism ---------------------------------------------------- #


def test_a_decode_narrower_than_its_prefill_is_refused():
    with rejects("tensor parallelism"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--tensor-parallel-size=8"],
                    decode_params=["--tensor-parallel-size=4"],
                )
            )
        )


def test_a_wider_decode_is_allowed():
    """The constraint is one-directional: decode must be at least prefill."""
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["-tp", "4"],
                decode_params=["-tp", "8"],
            )
        )
    )


# --- factors the engine also checks, caught here for attribution ----------- #


@pytest.mark.parametrize(
    "flag,label",
    [
        ("--dtype", "dtype"),
        ("--kv-cache-dtype", "KV cache dtype"),
        ("--block-size", "block size"),
        ("--kv-cache-layout", "KV cache layout"),
    ],
)
def test_disagreeing_handshake_factors_are_refused(flag, label):
    with rejects(label):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=[f"{flag}=a"],
                    decode_params=[f"{flag}=b"],
                )
            )
        )


# --- inheritance ----------------------------------------------------------- #


def test_roles_inheriting_the_model_parameters_agree_by_construction():
    """`None` inherits, so two roles that override nothing cannot disagree."""
    validate_role_pairing(
        _model_in(_roles(), backend_parameters=["--max-model-len=8192"])
    )


def test_one_role_overriding_is_compared_against_the_inherited_value():
    """The dangerous shape: the user edits only decode and never sees the
    model-level value they are now contradicting."""
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(decode_params=["--max-model-len=4096"]),
                backend_parameters=["--max-model-len=8192"],
            )
        )


def test_an_empty_override_does_not_inherit():
    """A role that deliberately clears the model's parameters must not get
    them back — so nothing is compared, rather than the model's value being
    compared against itself."""
    validate_role_pairing(
        _model_in(
            _roles(prefill_params=[]),
            backend_parameters=["--max-model-len=8192"],
        )
    )


# --- scope ----------------------------------------------------------------- #


def test_a_role_less_model_is_untouched():
    validate_role_pairing(_model_in(roles=None, disaggregation=False))


def test_plain_multi_role_without_disaggregation_is_untouched():
    """`roles` without `disaggregation` is not PD, so there is no pairing to
    check — nothing transfers KV between them."""
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["--max-model-len=8192"],
                decode_params=["--max-model-len=4096"],
            ),
            disaggregation=False,
        )
    )


def test_a_sparse_update_is_judged_against_the_stored_roles():
    """A PUT that changes only the model-level parameters would otherwise be
    read as a role-less model, and the contradiction it creates with the
    stored decode override would be accepted by not mentioning it."""
    from gpustack.schemas.models import ModelUpdate

    stored = Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        roles=_roles(decode_params=["--max-model-len=4096"]),
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )
    update = ModelUpdate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend_parameters=["--max-model-len=8192"],
    )

    with rejects("context lengths"):
        validate_role_pairing(update, stored=stored)
