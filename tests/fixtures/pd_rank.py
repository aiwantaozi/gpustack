from gpustack.schemas.models import DisaggregationSpec, Model, RoleSpec, SourceEnum


def model_with_ranks(p=2, d=4, dpa=False):
    parameters = ['--tp=4', '--pp=1']
    if dpa:
        parameters.append('--enable-dp-attention')
    return Model(
        name='pd',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/model',
        backend='SGLang',
        backend_parameters=parameters,
        disaggregation=DisaggregationSpec(mode='sglang-mooncake'),
        roles=[
            RoleSpec(
                name='prefill',
                replicas=1,
                backend_parameters=[*parameters, f'--dp={p}'],
            ),
            RoleSpec(
                name='decode', replicas=1, backend_parameters=[*parameters, f'--dp={d}']
            ),
            RoleSpec(name='router', replicas=1, backend_parameters=['--dp-aware']),
        ],
    )
