import json
import logging
import os
import sys
from typing import Dict, List, Optional

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config, set_global_config
from gpustack.config.registration import read_worker_token
from gpustack.envs import BENCHMARK_DATASET_SHAREGPT_PATH, BENCHMARK_REQUEST_TIMEOUT
from gpustack.logging import setup_logging
from gpustack.schemas.benchmark import (
    DATASET_CUSTOM,
    DATASET_RANDOM,
    DATASET_SHAREGPT,
    Benchmark,
    BenchmarkDeploymentMetadata,
    BenchmarkStateEnum,
    ModelInstanceSnapshot,
)
from gpustack.utils.command import find_bool_parameter
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.utils.envs import filter_env_vars, sanitize_env
from gpustack_runtime.logging import setup_logging as setup_runtime_logging
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer import ContainerMount

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
    ContainerRestartPolicyEnum,
)

from gpustack.utils.profiling import time_decorator
from gpustack.utils.runtime import transform_workload_plan

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    _clientset: ClientSet
    _config: Config
    _benchmark: Benchmark
    _model_path: str
    _model_endpoint: str
    _model_backend_parameters: Optional[List[str]]
    _api_url: str
    _api_key: str
    _benchmark_dir: Optional[str]
    _fallback_registry: Optional[str] = None
    """The fallback container registry to use if needed."""

    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        benchmark: Benchmark,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        setup_logging(debug=cfg.debug)
        setup_runtime_logging()
        set_global_config(cfg)

        try:
            self._clientset = clientset
            self._benchmark = benchmark
            self._config = cfg
            self._fallback_registry = fallback_registry

            if (
                benchmark.snapshot is None
                or benchmark.snapshot.instances is None
                or len(benchmark.snapshot.instances) == 0
                or benchmark.snapshot.instances.get(benchmark.model_instance_name)
                is None
            ):
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) has no snapshot for model instance {benchmark.model_instance_name}"
                )

            instance_snapshot: ModelInstanceSnapshot = benchmark.snapshot.instances.get(
                benchmark.model_instance_name
            )
            if instance_snapshot.resolved_path is None:
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) snapshot for model instance {benchmark.model_instance_name} has no resolved path"
                )

            if instance_snapshot.worker_ip is None:
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) snapshot for model instance {benchmark.model_instance_name} has no worker IP"
                )

            if instance_snapshot.ports is None or len(instance_snapshot.ports) == 0:
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) snapshot for model instance {benchmark.model_instance_name} has no ports"
                )

            self._benchmark_dir = self._config.benchmark_dir
            self._model_path = instance_snapshot.resolved_path
            self._model_endpoint = f"http://{instance_snapshot.worker_ip}:{instance_snapshot.ports[0] if instance_snapshot.ports else ''}"
            self._model_backend_parameters = instance_snapshot.backend_parameters

            _api_key = read_worker_token(self._config.data_dir)
            if _api_key is None:
                raise ValueError(
                    f"Worker token not found for benchmark {benchmark.name}(id={benchmark.id}) progress reporting"
                )
            self._api_key = _api_key

            _server_url = self._clientset.base_url
            if not _server_url:
                raise ValueError(
                    f"Server URL not configured for benchmark {benchmark.name}(id={benchmark.id}) progress reporting"
                )
            self._api_url = (
                f"{_server_url.rstrip('/')}/v2/benchmarks/{self._benchmark.id}/state"
            )

            # Resolve the custom Dataset resource (path + column mapping) so the
            # command builder / mounts can point --data at the on-worker files.
            self._dataset = None
            if (
                benchmark.dataset_name == DATASET_CUSTOM
                and benchmark.dataset_id is not None
            ):
                self._dataset = self._clientset.datasets.get(id=benchmark.dataset_id)
                if self._dataset is None or not self._dataset.resolved_paths:
                    raise ValueError(
                        f"Benchmark {benchmark.name}(id={benchmark.id}) references "
                        f"dataset {benchmark.dataset_id} which is not ready"
                    )

        except Exception as e:
            error_message = f"Failed to initialize: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": BenchmarkStateEnum.ERROR,
                }
                self._update_benchmark_state(benchmark.id, **patch_dict)
            except Exception as ue:
                logger.error(
                    f"Failed to update benchmark {benchmark.name}(id={benchmark.id}) state: {ue}"
                )
            sys.exit(1)

    def start(self):
        deployment_metadata = self._benchmark.get_deployment_metadata()

        env = {}
        if not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            env = filter_env_vars(os.environ)

        command_args = self._build_command_args()
        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=["benchmark-runner"],
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: BenchmarkDeploymentMetadata,
        command: Optional[List[str]],
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = apply_registry_override_to_image(
            self._config, self._config.benchmark_image_repo, self._fallback_registry
        )
        if not image:
            raise ValueError("Failed to get image for benchmark runner workload")

        mounts = self._get_configured_mounts()

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=False,
                args=command_args,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            mounts=mounts,
        )

        logger.info(
            f"Creating benchmark container workload: {deployment_metadata.name}"
        )
        logger.info(
            f"With image: {image}, "
            f"command: [{' '.join(command) if command else ''}], "
            f"arguments: [{' '.join(str(arg) for arg in command_args)}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
            labels=deployment_metadata.labels,
        )
        create_workload(
            transform_workload_plan(
                self._config, workload_plan, self._fallback_registry
            )
        )

        logger.info(f"Created benchmark container workload: {deployment_metadata.name}")

    def _build_command_args(self) -> List[str]:  # noqa: C901
        # guidellm 0.7.1: OpenAIHTTPBackend takes `response_handlers` keyed by
        # request TYPE (e.g. "chat_completions"), not API path; the value is a
        # handler name resolved by benchmark-runner. (0.6.0 used `request_handlers`
        # keyed by API path — benchmark-runner still accepts that legacy form via a
        # shim, but this is the native 0.7.1 shape.)
        backend_kwargs = {
            "timeout": BENCHMARK_REQUEST_TIMEOUT,
            "response_handlers": {
                "chat_completions": "chat_completions_with_reasoning"
            },
        }

        # Load selection. Three mutually-exclusive shapes:
        #   1. auto_tune  -> benchmark-runner's adaptive ramp engine (geometric
        #      bracket + binary search) over the load axis. Replaces the old
        #      guidellm `sweep` profile. Target derived: sla_* set -> SLA boundary,
        #      else throughput saturation.
        #   2. stages     -> one single-rate guidellm run per stage (Custom manual
        #      mode; each stage carries its own max_requests / max_seconds).
        #   3. legacy     -> single `constant`/`concurrent` run (old single-rate
        #      records via request_rate).
        b = self._benchmark
        if b.auto_tune:
            # axis: fixed_rate -> ramp request rate (open-loop constant);
            #       concurrency -> ramp streams (closed-loop concurrent).
            axis = "concurrency" if b.load_type == "concurrency" else "rate"
            profile_args = ["--auto-tune", "--axis", axis]
            for attr, flag in (
                ("lower_bound", "--lower-bound"),
                ("upper_bound", "--upper-bound"),
                ("max_points", "--max-points"),
                ("max_total_seconds", "--max-total-seconds"),
            ):
                value = getattr(b, attr, None)
                if value is not None:
                    profile_args += [flag, str(value)]
            # SLA targets ("<=" ms). Any one set -> target is the SLA boundary;
            # a point meets the SLA when every set threshold holds (AND).
            for attr, flag in (
                ("sla_avg_ttft_ms", "--sla-avg-ttft-ms"),
                ("sla_avg_tpot_ms", "--sla-avg-tpot-ms"),
                ("sla_p99_ttft_ms", "--sla-p99-ttft-ms"),
                ("sla_p99_tpot_ms", "--sla-p99-tpot-ms"),
                ("sla_avg_latency_ms", "--sla-avg-latency-ms"),
                ("sla_p99_latency_ms", "--sla-p99-latency-ms"),
            ):
                value = getattr(b, attr, None)
                if value is not None:
                    profile_args += [flag, str(value)]
        elif b.stages:
            # Custom manual mode: per-stage independent constraints. benchmark-runner
            # loops one single-rate run per stage (each carries its own
            # max_requests / max_seconds), so no top-level profile/rate here.
            profile_args = ["--stages", json.dumps(b.stages)]
        else:
            # concurrency -> guidellm `concurrent` (closed-loop N in-flight);
            # fixed_rate -> `constant` (open-loop fixed req/s).
            kind = "concurrent" if b.load_type == "concurrency" else "constant"
            profile_args = ["--profile", kind, "--rate", str(b.request_rate)]

        command_args = [
            "benchmark",
            "run",
            "--target",
            self._model_endpoint,
            *profile_args,
            "--sample-requests",
            "0",
            "--processor",
            self._model_path,
            "--output-dir",
            f"{self._benchmark_dir}",
            "--outputs",
            f"{self._benchmark.id}.dual_json",
            "--progress-url",
            self._api_url,
            "--progress-auth",
            self._api_key,
            "--backend-kwargs",
            json.dumps(backend_kwargs),
            "--backend",
            "openai_http_error_detail",
        ]

        # Multi-stage seed policy (auto-tune ramp / manual stages): increment the
        # seed per stage unless the user pinned it fixed. Only affects the Random
        # synthetic dataset (file datasets read in file order regardless of seed).
        command_args.append(
            "--seed-increment"
            if self._benchmark.dataset_seed_increment is not False
            else "--no-seed-increment"
        )

        if find_bool_parameter(self._model_backend_parameters, ["trust-remote-code"]):
            command_args.extend(
                [
                    "--processor-args",
                    json.dumps({"trust_remote_code": True}),
                ]
            )

        if self._benchmark.dataset_name == DATASET_SHAREGPT:
            data = BENCHMARK_DATASET_SHAREGPT_PATH
            command_args.extend(["--data", data])
        elif self._benchmark.dataset_name == DATASET_CUSTOM and self._dataset:
            # Custom Dataset resource: --data points at the on-worker path
            # (a dir for whole-dataset, a file for single-file mode). guidellm /
            # benchmark-runner detects the kind by extension / directory layout.
            command_args.extend(["--data", self._dataset.resolved_paths[0]])
            if self._dataset.column_mapping:
                command_args.extend(
                    [
                        "--data-column-mapper",
                        json.dumps({"column_mappings": self._dataset.column_mapping}),
                    ]
                )
            if self._benchmark.dataset_seed is not None:
                command_args.extend(
                    ["--random-seed", f"{self._benchmark.dataset_seed}"]
                )
        elif (
            self._benchmark.dataset_name == DATASET_RANDOM
            and self._benchmark.dataset_input_tokens is not None
            and self._benchmark.dataset_output_tokens is not None
        ):
            data = f"prompt_tokens={self._benchmark.dataset_input_tokens},output_tokens={self._benchmark.dataset_output_tokens}"
            # P1: data distribution — spread token lengths around the mean
            # (guidellm prompt_tokens_stdev/min/max + output_tokens_stdev/min/max).
            for attr, key in (
                ("dataset_input_stdev", "prompt_tokens_stdev"),
                ("dataset_input_min", "prompt_tokens_min"),
                ("dataset_input_max", "prompt_tokens_max"),
                ("dataset_output_stdev", "output_tokens_stdev"),
                ("dataset_output_min", "output_tokens_min"),
                ("dataset_output_max", "output_tokens_max"),
            ):
                value = getattr(self._benchmark, attr, None)
                if value is not None:
                    data += f",{key}={value}"
            # P1-10: multi-turn synthetic conversations
            if self._benchmark.turns and self._benchmark.turns > 1:
                data += f",turns={self._benchmark.turns}"
            # Shared prefix: prefix_buckets is a nested list, so re-encode the whole
            # data config as JSON (guidellm parses --data starting with "{" as JSON).
            prefix_buckets = self._benchmark.prefix_buckets
            if prefix_buckets:
                data_dict: dict = {}
                for pair in data.split(","):
                    k, _, v = pair.partition("=")
                    k = k.strip()
                    v = v.strip()
                    data_dict[k] = int(v) if v.lstrip("-").isdigit() else v
                data_dict["prefix_buckets"] = prefix_buckets
                data = json.dumps(data_dict)
            command_args.extend(["--data", data])

            if self._benchmark.dataset_seed is not None:
                command_args.extend(
                    [
                        "--random-seed",
                        f"{self._benchmark.dataset_seed}",
                    ]
                )

        # auto_tune computes per-point request counts itself (knob * multiplier)
        # and caps the whole run via --max-total-seconds, so it takes neither the
        # global --max-requests nor --max-seconds.
        if not b.auto_tune:
            if (
                self._benchmark.total_requests is not None
                and self._benchmark.total_requests > 0
            ):
                command_args.extend(
                    [
                        "--max-requests",
                        f"{self._benchmark.total_requests}",
                    ]
                )

            # global duration cap for non-stage runs (stages carry their own).
            if b.max_seconds is not None and not b.stages:
                command_args.extend(["--max-seconds", str(b.max_seconds)])

        # P1-12: warmup / cooldown / constraints (透传给 benchmark-runner / guidellm)
        if b.warmup is not None:
            command_args.extend(["--warmup", str(b.warmup)])
        if b.cooldown is not None:
            command_args.extend(["--cooldown", str(b.cooldown)])
        if b.max_errors is not None:
            command_args.extend(["--max-errors", str(b.max_errors)])
        if b.max_error_rate is not None:
            command_args.extend(["--max-error-rate", str(b.max_error_rate)])
        if b.stop_on_saturation:
            command_args.append("--detect-saturation")

        return command_args

    def _update_benchmark_state(self, id: int, **kwargs):
        resp = self._clientset.http_client.get_httpx_client().patch(
            "/benchmarks/{id}/state".format(id=id), json=kwargs
        )
        resp.raise_for_status()

    def _get_configured_mounts(self) -> List[ContainerMount]:
        """
        Get the volume mounts for the model instance.
        If runtime mirrored deployment is enabled, no mounts will be set up.

        Returns:
            A list of ContainerMount objects for the model instance.
        """
        mounts: List[ContainerMount] = []
        if (
            self._model_path
            and self._benchmark_dir
            and not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
        ):
            model_dir = os.path.dirname(self._model_path)
            mounts.extend(
                [
                    ContainerMount(
                        path=model_dir,
                    ),
                    ContainerMount(
                        path=self._benchmark_dir,
                    ),
                ]
            )
            # Mount the custom dataset so --data resolves inside the container.
            # Mount the containing dir for a single file, or the dir itself.
            if self._dataset is not None and self._dataset.resolved_paths:
                ds_path = self._dataset.resolved_paths[0]
                ds_mount = (
                    ds_path if os.path.isdir(ds_path) else os.path.dirname(ds_path)
                )
                if ds_mount:
                    mounts.append(ContainerMount(path=ds_mount))
        return mounts
