import pytest

from tests.fixtures.pd_rank import model_with_ranks
from gpustack.schemas.models import role_effective_model
from gpustack.utils.pd_rank import sglang_request_rank_count, split_rank_url


@pytest.mark.parametrize(
    'url,expected',
    [
        ('http://host:80@2', ('http://host:80', 2)),
        ('http://[::1]:80@0', ('http://[::1]:80', 0)),
        ('http://user@host:80', ('http://user@host:80', None)),
        ('http://host:80/path@2', ('http://host:80/path@2', None)),
    ],
)
def test_logical_rank_url(url, expected):
    assert split_rank_url(url) == expected


@pytest.mark.parametrize('dpa', [False, True])
def test_ordinary_dp_and_attention_dp_have_one_rank_namespace(dpa):
    model = model_with_ranks(dpa=dpa)
    parameters = role_effective_model(model, 'prefill').backend_parameters
    assert ('--enable-dp-attention' in parameters) is dpa
    assert sglang_request_rank_count(model, 'prefill') == 2
    assert sglang_request_rank_count(model, 'decode') == 4


def test_new_attention_layout_requires_scheduler_support():
    model = model_with_ranks()
    model.roles[0].backend_parameters = ['--attn-dp-size=2']
    with pytest.raises(ValueError, match='scheduler integration'):
        sglang_request_rank_count(model, 'prefill')


@pytest.mark.parametrize('size', [0, -1])
def test_nonpositive_rank_count_is_rejected(size):
    model = model_with_ranks(p=size)
    with pytest.raises(ValueError, match='must be positive'):
        sglang_request_rank_count(model, 'prefill')


def test_role_parameters_replace_inherited_parallelism_and_last_alias_wins():
    model = model_with_ranks()
    model.backend_parameters = ['--dp=8', '--tp=4', '--pp=2']
    model.roles[0].backend_parameters = ['--dp=2', '--data-parallel-size', '3']
    assert sglang_request_rank_count(model, 'prefill') == 3
    model.roles[0].backend_parameters = ['--tp=4', '--pp=2']
    assert sglang_request_rank_count(model, 'prefill') == 1
    model.roles[0].backend_parameters = None
    assert sglang_request_rank_count(model, 'prefill') == 8
