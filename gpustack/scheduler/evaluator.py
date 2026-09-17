import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

from gpustack_runtime.detector import ManufacturerEnum
from sqlmodel.ext.asyncio.session import AsyncSession
from cachetools import TTLCache
from aiolimiter import AsyncLimiter

from gpustack.api.exceptions import HTTPException
from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.config.config import Config
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack import envs
from gpustack.routes.models import validate_model_in
from gpustack.scheduler import scheduler
from gpustack.server.catalog import get_catalog_spec_by_source_key
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_evaluations import (
    ModelEvaluationResult,
    ModelSpec,
    ResourceClaim,
    RoleResourceClaim,
)
from gpustack.schemas.models import (
    ModelInstance,
    BackendEnum,
    SourceEnum,
    get_backend,
    is_gguf_model,
    is_audio_model,
    role_container_resources,
    role_takes_no_accelerator,
)
from gpustack.schemas.principals import _platform_principal_id
from gpustack.scheduler.group_capacity import (
    GroupCapacity,
    attendant_demands,
    role_demands,
)
from gpustack.scheduler.group_schedule import (
    cache_instances_in,
    gather_request,
    stand_in,
)
from gpustack.scheduler.group_solver import (
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.scheduler.topology import TopologyError
from gpustack.scheduler.topology_view import build_view
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.worker_selector import WorkerSelector

from gpustack.utils.gpu import (
    all_gpu_match,
    any_gpu_match,
    find_one_gpu,
    compare_compute_capability,
)
from gpustack.utils.hub import (
    auth_check,
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    is_repo_cached,
)
from gpustack.utils.task import run_in_thread
from gpustack.utils.profiling import time_decorator

logger = logging.getLogger(__name__)

evaluate_cache = TTLCache(
    maxsize=envs.MODEL_EVALUATION_CACHE_MAX_SIZE, ttl=envs.MODEL_EVALUATION_CACHE_TTL
)

# To reduce the likelihood of hitting the Hugging Face API rate limit (600 RPM)
# Limit the number of concurrent evaluations to 50 per 10 seconds
evaluate_model_limiter = AsyncLimiter(50, 10)


@time_decorator
async def evaluate_models(
    config: Config,
    session: AsyncSession,
    model_specs: List[ModelSpec],
    cluster_id: Optional[int] = None,
) -> List[ModelEvaluationResult]:
    """
    Evaluate the compatibility of a list of model specs with the available workers.
    """
    fields = {
        "deleted_at": None,
    }
    if cluster_id is not None:
        fields["cluster_id"] = cluster_id
    extra_conditions = [
        ~(
            Worker.state.in_(
                [
                    WorkerStateEnum.PROVISIONING,
                    WorkerStateEnum.DELETING,
                    WorkerStateEnum.ERROR,
                ]
            )
        )
    ]
    workers = await Worker.all_by_fields(
        session, fields=fields, extra_conditions=extra_conditions
    )

    model_instances = await ModelInstance.all_by_fields(session, fields=fields)

    if len(model_specs) == 1:
        # Sort worker for single-model evaluation only. No need for batch evaluation.
        workers = await scheduler.prioritize_workers_with_model_files(
            session, model_specs[0], workers
        )

    async def evaluate(model: ModelSpec):
        return await evaluate_model_with_cache(
            config,
            session,
            model,
            workers,
            model_instances,
            cluster_id=cluster_id,
        )

    tasks = [evaluate(model) for model in model_specs]
    results = await asyncio.gather(*tasks)
    return results


def make_hashable_key(model: ModelSpec, workers: List[Worker]) -> str:
    key_data = json.dumps(
        {
            "model": model.model_dump(mode="json"),
            # Excluded from model_dump (response-hidden field), but it
            # changes which Org-scoped backend versions the evaluation
            # sees — without it cached results would leak across Orgs.
            "owner_principal_id": getattr(model, "owner_principal_id", None),
            "workers": [
                w.model_dump(
                    mode="json",
                    exclude={
                        "status": {
                            "cpu": True,
                            "swap": True,
                            "filesystem": True,
                            "os": True,
                            "kernel": True,
                            "uptime": True,
                            "memory": {"utilization_rate", "used"},
                            "gpu_devices": {
                                "__all__": {
                                    "temperature": True,
                                    "core": {"utilization_rate"},
                                    "memory": {"utilization_rate", "used"},
                                },
                            },
                        },
                        "heartbeat_time": True,
                        "created_at": True,
                        "updated_at": True,
                    },
                )
                for w in workers
            ],
        },
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode()).hexdigest()


async def evaluate_model_with_cache(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    cluster_id: Optional[int] = None,
) -> ModelEvaluationResult:
    cache_key = make_hashable_key(model, workers)
    if cache_key in evaluate_cache:
        logger.trace(
            f"Evaluation cache hit for model: {model.name or model.readable_source}"
        )
        return evaluate_cache[cache_key]

    try:
        async with evaluate_model_limiter:
            result = await evaluate_model(
                config, session, model, workers, model_instances, cluster_id=cluster_id
            )
            evaluate_cache[cache_key] = result
    except Exception as e:
        logger.exception(
            f"Error evaluating model {model.name or model.readable_source}: {e}"
        )
        result = ModelEvaluationResult(
            compatible=False, error=True, error_message=str(e)
        )

    return result


@time_decorator
async def evaluate_model(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    cluster_id: Optional[int] = None,
) -> ModelEvaluationResult:
    result = ModelEvaluationResult()

    if await set_default_spec(session, model):
        result.default_spec = model.model_copy()

    await set_gguf_model_file_path(config, model)

    evaluations = [
        (evaluate_model_input, (session, model, cluster_id)),
        (evaluate_model_metadata, (config, model, workers)),
        (evaluate_environment, (model, workers)),
    ]
    for evaluation, args in evaluations:
        compatible, messages = await evaluation(*args)
        if not compatible:
            result.compatible = False
            result.compatibility_messages = messages
            return result

    workers_by_cluster: Dict[int, List[Worker]] = defaultdict(list)
    for worker in workers:
        workers_by_cluster[worker.cluster_id].append(worker)

    overcommit_clusters = []
    result.resource_claim_by_cluster_id = {}
    role_claims_by_cluster_id: Dict[int, List[RoleResourceClaim]] = {}

    for cluster_id, cluster_workers in workers_by_cluster.items():
        cluster_model_instances = [
            inst for inst in model_instances if inst.cluster_id == cluster_id
        ]

        # A role-bearing deployment is a group, and a group is placed all at
        # once. Asking `find_candidate` instead answers a question the
        # deployment never asks -- "where would ONE instance of the model-level
        # spec go" -- and its answer is wrong in three directions at the same
        # time: it ignores the role overrides (a decode's tensor parallelism is
        # not the model's), it counts one member where the group has x+y+1, and
        # it never checks that prefill and decode fit *together*.
        if model.roles:
            group_claim, role_claims, group_messages = await evaluate_group(
                config,
                session,
                model,
                cluster_workers,
                cluster_model_instances,
                cluster_id,
            )
            if group_claim is None:
                result.scheduling_messages.extend(group_messages)
                continue
            result.resource_claim_by_cluster_id[cluster_id] = group_claim
            role_claims_by_cluster_id[cluster_id] = role_claims
            continue

        candidate, schedule_messages = await scheduler.find_candidate(
            session, config, model, cluster_workers, cluster_model_instances
        )
        if not candidate:
            result.scheduling_messages.extend(schedule_messages)
            continue
        if candidate.overcommit:
            overcommit_clusters.append(cluster_id)
            result.scheduling_messages.extend(schedule_messages)
            continue
        result.resource_claim_by_cluster_id[cluster_id] = (
            summarize_candidate_resource_claim(candidate)
        )

    if result.resource_claim_by_cluster_id:
        first_cluster_id = next(iter(result.resource_claim_by_cluster_id))
        result.resource_claim = result.resource_claim_by_cluster_id[first_cluster_id]
        if role_claims_by_cluster_id:
            # Kept in step with `resource_claim` above rather than derived
            # separately: the two describe the same cluster's placement, and a
            # reader that took the total from one cluster and the breakdown
            # from another would show a breakdown that does not add up.
            result.role_resource_claims_by_cluster_id = role_claims_by_cluster_id
            result.role_resource_claims = role_claims_by_cluster_id.get(
                first_cluster_id
            )
    else:
        result.resource_claim = None
        result.compatible = False
        result.compatibility_messages.append(
            "Unable to find a schedulable worker for the model."
        )
    return result


async def evaluate_group(
    config: Config,
    session: AsyncSession,
    model: ModelSpec,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    cluster_id: int,
) -> Tuple[Optional[ResourceClaim], List[RoleResourceClaim], List[str]]:
    """What the whole group would claim in this cluster, or why it cannot land.

    Returns ``(total, per_role, messages)``. A ``None`` total means the group
    does not fit, and ``messages`` then carries the solver's own refusal --
    which names the shortfall and the roomiest domain -- rather than a generic
    "no suitable worker".

    🔴 **The same pass the scheduler runs, not a second estimate of it.** A
    feasibility answer computed a different way can be right and still
    disagree with what happens at deploy time, and the one thing an evaluation
    must not do is promise a placement the scheduler then refuses. So this goes
    through `solve_group_placement` over the cluster's own topology, with the
    group's gather requirement applied, and reads the claim off the candidates
    `GroupCapacity.commit` produces -- the very objects the group scheduler
    writes onto instance rows.

    ⚠️ **The router is priced here and checked, but still not placed.** It
    stays out of `role_demands` -- counting it among the gang would make a 4P4D
    need nine placements in one domain -- so this function adds its container
    memory to the total afterwards. What changed is that it now travels as an
    `attendant`: the solver verifies a worker can host it before calling the
    group placeable. Until then a cluster with room for the GPU members and
    none for the router evaluated as compatible, in both places for the same
    reason, and the deployment it promised then sat with a router that could
    not be scheduled -- which is a group that serves nothing, since a router
    answers every request.
    """
    # 🔴 A `ModelSpec` is not quite a `Model`, and the role projection
    # revalidates it through `ModelBase` — where two of these fields are not
    # optional. Left as they arrive, `role_effective_model` raises and the
    # whole evaluation reports "invalid role specification" for a spec that is
    # perfectly valid:
    #
    # - `name` is optional on a spec, because the deployment form evaluates
    #   while it is still being filled in;
    # - `owner_principal_id` is stamped by the route from the caller's context
    #   and is legitimately None there — that is the Platform-only view (an
    #   admin in "All" mode) — while `ModelBase` declares it a plain int.
    #
    # `cluster_id` is stamped for a different reason: `ClusterFilter` reads it,
    # and this call is already scoped to the cluster whose workers we hold.
    group_model = model.model_copy(
        update={
            "cluster_id": cluster_id,
            "name": model.name or "model-evaluation",
            "owner_principal_id": (
                model.owner_principal_id or _platform_principal_id()
            ),
        }
    )

    # Ordered like the deployment declares them, so the breakdown reads
    # prefill, decode, router rather than in whatever order the solver found
    # convenient (it sorts by weight) or a dict happened to keep.
    order = {spec.name: index for index, spec in enumerate(group_model.roles or [])}

    try:
        demands = [RoleDemand(**d) for d in role_demands(group_model)]
    except Exception as e:
        return None, [], [f"Invalid role specification: {e}"]

    free_claims = _accelerator_free_claims(group_model)
    if not demands:
        # Nothing in this group occupies an accelerator. There is no placement
        # to solve, but the group still has an honest footprint -- a router's
        # container memory -- and reporting it beats reporting nothing.
        if not free_claims:
            return None, [], ["The group has no member that occupies an accelerator."]
        return _total_claim(free_claims), free_claims, []

    cluster = await Cluster.one_by_id(session, cluster_id) if cluster_id else None
    try:
        view = build_view(cluster.topology if cluster else None, workers)
    except TopologyError as e:
        return None, [], [f"Cluster topology is invalid: {e}"]

    cache_instances = await cache_instances_in(session, cluster_id)
    capacity = GroupCapacity(
        config, group_model, workers, model_instances, cache_instances
    )
    placement = await solve_group_placement(
        view.root,
        demands,
        capacity,
        view.scopes(),
        gather_request(group_model),
        attendants=[RoleDemand(**d) for d in attendant_demands(group_model)],
    )
    if not isinstance(placement, GroupPlacement):
        reason = getattr(placement, "reason", "The group does not fit.")
        notes = capacity.notes_for(getattr(placement, "role", None))
        return None, [], [reason] + notes

    # `already` accumulates across roles for the same reason the capacity count
    # does: the second role has to see what the first one took, or both are
    # priced against the same free cards.
    already: List[object] = []
    claims: List[RoleResourceClaim] = []
    for role, worker_ids in placement.assignments.items():
        candidates = await capacity.commit(role, worker_ids, already)
        if len(candidates) != len(worker_ids):
            return (
                None,
                [],
                [
                    "The group's placement could not be turned into GPU "
                    f"assignments for role '{role}'."
                ],
            )
        claims.append(
            _role_claim(
                role, [summarize_candidate_resource_claim(c) for c in candidates]
            )
        )
        already.extend(stand_in(c) for c in candidates)

    claims.extend(free_claims)
    claims.sort(key=lambda claim: order.get(claim.role, len(order)))
    return _total_claim(claims), claims, []


def _accelerator_free_claims(model) -> List[RoleResourceClaim]:
    """The roles the solver does not place, priced from what they declare.

    Today that is the router alone, and its claim is a declared floor rather
    than an estimate -- it holds no weights, so there is nothing to size.
    """
    claims: List[RoleResourceClaim] = []
    for spec in model.roles or []:
        if not role_takes_no_accelerator(model, spec.name):
            continue
        replicas = max(int(spec.replicas or 0), 0)
        ram = role_container_resources(model, spec.name).memory or 0
        claims.append(
            RoleResourceClaim(
                role=spec.name,
                replicas=replicas,
                ram=ram * replicas,
                vram=0,
                per_replica=ResourceClaim(ram=ram, vram=0),
            )
        )
    return claims


def _role_claim(role: str, per_member: List[ResourceClaim]) -> RoleResourceClaim:
    first = per_member[0] if per_member else None
    uniform = first is not None and all(
        claim.ram == first.ram and claim.vram == first.vram for claim in per_member
    )
    return RoleResourceClaim(
        role=role,
        replicas=len(per_member),
        ram=sum(claim.ram for claim in per_member),
        vram=sum(claim.vram for claim in per_member),
        # None rather than the first member's numbers when the members
        # disagree: a heterogeneous role has no "per replica" figure, and
        # showing one member's as if it were every member's is how a 2P4D on
        # mixed cards would read as half its real size.
        per_replica=first if uniform else None,
    )


def _total_claim(claims: List[RoleResourceClaim]) -> ResourceClaim:
    return ResourceClaim(
        ram=sum(claim.ram for claim in claims),
        vram=sum(claim.vram for claim in claims),
    )


def summarize_candidate_resource_claim(
    candidate: ModelInstanceScheduleCandidate,
) -> ResourceClaim:
    """
    Summarize the computed resource claim for a schedule candidate.
    """
    computed_resource_claims = [candidate.computed_resource_claim]

    if candidate.subordinate_workers:
        computed_resource_claims.extend(
            sw.computed_resource_claim
            for sw in candidate.subordinate_workers
            if sw.computed_resource_claim is not None
        )

    ram, vram = 0, 0
    for computed_resource_claim in computed_resource_claims:
        ram += computed_resource_claim.ram or 0
        if computed_resource_claim.vram:
            vram += sum(
                v for v in computed_resource_claim.vram.values() if v is not None
            )

    return ResourceClaim(ram=ram, vram=vram)


async def set_gguf_model_file_path(config: Config, model: ModelSpec):
    if (
        model.source == SourceEnum.HUGGING_FACE
        and "gguf" in model.huggingface_repo_id.lower()
        and not model.huggingface_filename
    ):
        model.huggingface_filename = await run_in_thread(
            get_hugging_face_model_min_gguf_path,
            timeout=15,
            model_id=model.huggingface_repo_id,
            token=config.huggingface_token,
        )
    elif (
        model.source == SourceEnum.MODEL_SCOPE
        and "gguf" in model.model_scope_model_id.lower()
        and not model.model_scope_file_path
    ):
        model.model_scope_file_path = await run_in_thread(
            get_model_scope_model_min_gguf_path,
            timeout=15,
            model_id=model.model_scope_model_id,
        )


async def evaluate_environment(
    model: ModelSpec,
    workers: List[Worker],
) -> Tuple[bool, List[str]]:
    backend = get_backend(model)

    if backend == BackendEnum.ASCEND_MINDIE and not any_gpu_match(
        workers, lambda gpu: gpu.vendor == ManufacturerEnum.ASCEND.value
    ):
        return False, [
            "The Ascend MindIE backend requires Ascend NPUs but none are available."
        ]

    if (
        backend == BackendEnum.SGLANG
        and all_gpu_match(
            workers, lambda gpu: gpu.vendor == ManufacturerEnum.NVIDIA.value
        )
        and not any_gpu_match(
            workers,
            lambda gpu: compare_compute_capability(gpu.compute_capability, "8.0") >= 0,
        )
    ):
        # Ref: https://github.com/sgl-project/sglang/issues/6006
        gpu = find_one_gpu(workers)
        return False, [
            "The SGLang backend requires NVIDIA GPUs with compute capability 8.0 or higher "
            "(e.g., A100/SM80, H100/SM90, RTX 3090/SM86). "
            + (
                f"Available GPU: {gpu.name} (compute capability: {gpu.compute_capability})"
                if gpu
                else ""
            )
        ]

    return True, []


async def evaluate_model_metadata(
    config: Config,
    model: ModelSpec,
    workers: List[Worker],
) -> Tuple[bool, List[str]]:
    try:
        if model.source == SourceEnum.LOCAL_PATH:
            # Check if local path exists on server
            path_exists_on_server = os.path.exists(model.local_path)

            if not path_exists_on_server:
                # Try to check if path exists on any worker
                try:
                    async with WorkerFilesystemClient() as filesystem_client:
                        selector = WorkerSelector(filesystem_client)

                        found_worker = await selector.find_worker_with_path(
                            workers, path=model.local_path
                        )

                        if found_worker:
                            logger.info(
                                f"Found path {model.local_path} on worker {found_worker.id}"
                            )
                        else:
                            # Path not found on any worker
                            return False, [
                                "The model file path you specified does not exist."
                                "Please ensure the model file is accessible from at least one node."
                            ]
                except Exception as e:
                    logger.warning(
                        f"Failed to check path on workers: {e}, falling back to local check"
                    )
                    # Fallback to original warning
                    return False, [
                        "Failed to get model metadata. The model file path you specified does not exist."
                    ]

        if model.source in [
            SourceEnum.HUGGING_FACE,
            SourceEnum.MODEL_SCOPE,
        ]:
            repo_id = model.huggingface_repo_id
            if model.source == SourceEnum.MODEL_SCOPE:
                repo_id = model.model_scope_model_id
            if not is_repo_cached(repo_id, model.source):
                await run_in_thread(
                    auth_check,
                    timeout=15,
                    model=model,
                    huggingface_token=config.huggingface_token,
                )

        if is_gguf_model(model):
            await scheduler.evaluate_gguf_model(model, workers=workers)
        elif not is_audio_model(model):
            await scheduler.evaluate_pretrained_config(model, workers=workers)
    except Exception as e:
        if model.env and model.env.get("GPUSTACK_SKIP_MODEL_EVALUATION"):
            logger.warning(f"Ignore model evaluation error for model {model.name}: {e}")
            return True, []

        return False, [str(e)]

    return True, []


async def evaluate_model_input(
    session: AsyncSession,
    model: ModelSpec,
    cluster_id: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    try:
        await validate_model_in(session, model, cluster_id=cluster_id)
    except HTTPException as e:
        return False, [e.message]
    except Exception as e:
        return False, [str(e)]

    return True, []


async def set_default_spec(session: AsyncSession, model: ModelSpec) -> bool:
    """
    Set the default spec for the model if it matches the catalog spec.
    """
    model_spec_in_catalog = await get_catalog_spec_by_source_key(
        session, model.model_source_key
    )

    modified = False
    if model_spec_in_catalog:
        if (
            model_spec_in_catalog.backend_parameters
            and model.backend_parameters is None
        ):
            model.backend_parameters = model_spec_in_catalog.backend_parameters
            modified = True

        if model_spec_in_catalog.env and model.env is None:
            model.env = model_spec_in_catalog.env
            modified = True

        if model_spec_in_catalog.categories and not model.categories:
            model.categories = model_spec_in_catalog.categories
            modified = True

    gpus_per_replica_modified = scheduler.set_model_gpus_per_replica(model)
    return modified or gpus_per_replica_modified
