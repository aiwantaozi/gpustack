import logging
import os
from typing import List, Optional

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config, set_global_config
from gpustack.logging import setup_logging
from gpustack.routes.evaluation_suites import load_builtin_evaluation_suites
from gpustack.schemas.evaluations import Evaluation
from gpustack.schemas.models import BackendEnum
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.utils.envs import filter_env_vars, sanitize_env
from gpustack.worker.evaluation.command_builder import build_lm_eval_command
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    ContainerRestartPolicyEnum,
    WorkloadPlan,
    create_workload,
)

from gpustack.utils.runtime import get_configured_mounts, transform_workload_plan

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        clientset: ClientSet,
        evaluation: Evaluation,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        setup_logging(debug=cfg.debug)
        set_global_config(cfg)

        self._clientset = clientset
        self._evaluation = evaluation
        self._config = cfg
        self._fallback_registry = fallback_registry

        snapshot = evaluation.snapshot
        if (
            snapshot is None
            or snapshot.instances is None
            or evaluation.model_instance_name not in snapshot.instances
        ):
            raise ValueError(
                f"Evaluation {evaluation.name}(id={evaluation.id}) has no model instance snapshot"
            )

        self._instance_snapshot = snapshot.instances[evaluation.model_instance_name]
        if self._instance_snapshot.worker_ip is None:
            raise ValueError("Evaluation snapshot missing worker_ip")
        if not self._instance_snapshot.ports:
            raise ValueError("Evaluation snapshot missing instance ports")

    def start(self):
        command = self._build_command()
        self._create_workload(command_args=command[1:])

    def _build_command(self) -> List[str]:
        base_url = f"http://{self._instance_snapshot.worker_ip}:{self._instance_snapshot.ports[0]}"
        backend = self._instance_snapshot.backend
        if backend == BackendEnum.SGLANG:
            base_url = f"{base_url}/generate"
            pretrained = self._evaluation.model_name
        else:
            base_url = f"{base_url}/v1/completions"
            pretrained = self._evaluation.model_name

        suite = next(
            (
                item
                for item in load_builtin_evaluation_suites().items
                if item.id == self._evaluation.suite_id
            ),
            None,
        )
        if suite is None:
            raise ValueError(f"Unknown evaluation suite {self._evaluation.suite_id}")

        command = build_lm_eval_command(
            backend=backend,
            pretrained=pretrained,
            base_url=base_url,
            tokenizer=self._instance_snapshot.resolved_path or "",
            tasks=suite.tasks,
            batch_size=1,
        )
        command.extend(
            [
                "--output_path",
                os.path.join(
                    self._config.evaluation_dir, f"{self._evaluation.id}.json"
                ),
            ]
        )
        return command

    def _create_workload(self, command_args: List[str]):
        image = apply_registry_override_to_image(
            self._config, self._config.evaluation_image_repo, self._fallback_registry
        )
        if not image:
            raise ValueError("Failed to get image for evaluation runner workload")

        env = {}
        if not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            env = filter_env_vars(os.environ)

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                args=command_args,
            ),
            envs=[ContainerEnv(name=name, value=value) for name, value in env.items()],
            mounts=get_configured_mounts(
                self._instance_snapshot.resolved_path,
                extra_paths=[self._config.evaluation_dir],
            ),
        )

        deployment_metadata = self._evaluation.get_deployment_metadata()
        logger.info(
            f"Creating evaluation container workload: {deployment_metadata.name}"
        )
        logger.info(
            f"With image: {image}, arguments: [{' '.join(str(arg) for arg in command_args)}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=10 * 1 << 30,
            containers=[run_container],
            labels=deployment_metadata.labels,
        )
        create_workload(
            transform_workload_plan(
                self._config, workload_plan, self._fallback_registry
            )
        )
