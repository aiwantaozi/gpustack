import os
from typing import List, Optional, Union
from gpustack_runtime.envs import (
    GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE,
    GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE,
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT,
)
from gpustack_runtime.deployer.docker import DockerWorkloadPlan
from gpustack_runtime.deployer import (
    ContainerMount,
    WorkloadPlan,
    DockerDeployer,
    WorkloadStatus,
)

from gpustack.config.config import Config
from gpustack.utils.config import apply_registry_override_to_image


def transform_workload_plan(
    config: Config,
    workload: WorkloadPlan,
    fallback_registry: Optional[str] = None,
) -> Union[DockerWorkloadPlan, WorkloadPlan]:
    """
    If the deployer is docker, transform the generic WorkloadPlan to DockerWorkloadPlan,
    and fill the pause image and restart image with registry override.
    """
    if not DockerDeployer().is_supported():
        return workload
    pause_image = apply_registry_override_to_image(
        config, GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE, fallback_registry
    )
    restart_image = apply_registry_override_to_image(
        config, GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE, fallback_registry
    )
    docker_workload = DockerWorkloadPlan(
        pause_image=pause_image,
        unhealthy_restart_image=restart_image,
        **workload.__dict__,
    )
    return docker_workload


def get_configured_mounts(
    model_path: Optional[str],
    extra_paths: Optional[List[Optional[str]]] = None,
) -> List[ContainerMount]:
    """
    Build workload mounts for the model directory and optional extra directories.
    If runtime mirrored deployment is enabled, no mounts will be set up.
    """
    if not model_path or GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
        return []

    mount_paths = [os.path.dirname(model_path)]
    if extra_paths:
        mount_paths.extend(path for path in extra_paths if path)

    mounts: List[ContainerMount] = []
    seen = set()
    for path in mount_paths:
        normalized_path = os.path.normpath(path)
        if normalized_path in seen:
            continue
        seen.add(normalized_path)
        mounts.append(ContainerMount(path=normalized_path))

    return mounts


def is_benchmark_workload(status: WorkloadStatus) -> bool:
    """
    Check if a workload is a benchmark workload.

    A workload is considered a benchmark workload if it has the 'type' label
    set to 'benchmark'.

    Args:
        status: The workload status to check.

    Returns:
        True if the workload is a benchmark workload, False otherwise.
    """
    if not status.labels:
        return False
    return status.labels.get("type") == "benchmark"
