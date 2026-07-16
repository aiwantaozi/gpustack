import asyncio
import multiprocessing
import setproctitle
import os
import re
import time
from typing import Dict, Optional, Callable, List, Tuple
import logging
from collections import Counter, deque

from gpustack_runtime.deployer import (
    delete_workload,
    get_workload,
    WorkloadStatusStateEnum,
)
from gpustack.api.exceptions import raise_if_response_error
from gpustack.config.config import Config
from gpustack.config import registration
from gpustack.logging import RedirectStdoutStderr
from gpustack.schemas.benchmark import (
    DATASET_CUSTOM,
    Benchmark,
    BenchmarkStateEnum,
)
from gpustack.schemas.datasets import DatasetStateEnum
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.benchmark.runner import BenchmarkRunner
from gpustack.client import ClientSet
from gpustack.server.bus import Event, EventType
from gpustack.worker.schemas.benchmark_runner import (
    GenerativeBenchmarksReport,
    GenerativeRequestStats,
)
from gpustack_runtime.deployer import logs_workload


logger = logging.getLogger(__name__)

HTTP_ERROR_PATTERN = re.compile(
    r"^HTTP\s+(?P<status>\d+):\s+(?P<msg>.*)\s+\(type=(?P<type>[^,]+),\s*code=(?P<code>[^)]+)\)$"
)
TRUNCATION_SUFFIX = "..."
BENCHMARK_STATE_MESSAGE_MAX_LEN = 1024
BENCHMARK_FAILURE_REASON_MAX_LEN = 220
# Snapshot the running container's logs to disk at most this often, so logs are
# preserved even if the container is garbage-collected before we poll a terminal
# state (see _maybe_snapshot_logs).
BENCHMARK_LOG_SNAPSHOT_INTERVAL_SECONDS = 30


class BenchmarkManager:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    """
    The ID of current worker.
    """
    _config: Config
    """
    Global configuration.
    """
    _benchmark_log_dir: str
    """
    The directory to store logs of benchmarks(in subprocess).
    """
    _benchmark_dir: str
    """
    The directory to store results of benchmarks(in subprocess).
    """

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """

    _provisioning_processes: Dict[int, multiprocessing.Process]
    """
    The mapping of benchmark ID to provisioning (sub)process.
    When the (sub)process is alive, the benchmark is provisioning.
    If the (sub)process exited, the benchmark is either running or failed.
    """
    _benchmark_by_id: Dict[int, Benchmark]
    _benchmark_queue: deque
    _queue_lock: asyncio.Lock
    _worker_task: Optional[asyncio.Task]
    _active_benchmark_id: Optional[int]
    _active_benchmark_started_at: Optional[float]

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._benchmark_log_dir = f"{cfg.log_dir}/benchmarks"
        self._benchmark_dir = f"{cfg.benchmark_dir}"
        self._clientset_getter = clientset_getter

        self._provisioning_processes = {}
        self._benchmark_by_id = {}
        self._benchmark_queue = deque()
        self._queue_lock = asyncio.Lock()
        self._worker_task = None
        self._active_benchmark_id = None
        self._active_benchmark_started_at = None
        # Per-benchmark: byte offset where the container logs begin (after the
        # provisioning logs the subprocess wrote), and the last snapshot time.
        self._container_log_offset: Dict[int, int] = {}
        self._last_log_snapshot_at: Dict[int, float] = {}

        os.makedirs(self._benchmark_log_dir, exist_ok=True)
        os.makedirs(self._benchmark_dir, exist_ok=True)

    async def watch_benchmarks_event(self):
        """
        Loop to watch benchmarks' event and handle.
        """
        logger.info("Watching benchmarks event.")
        if not self._worker_task or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._benchmark_queue_worker())
        while True:
            try:
                await self._clientset.benchmarks.awatch(
                    callback=self._handle_benchmark_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching benchmarks: {e}")
                await asyncio.sleep(5)

    def _handle_benchmark_event(self, event: Event):
        """
        Handle benchmark events.
        Args:
            event: The benchmark event to handle.
        """
        benchmark = Benchmark.model_validate(event.data)
        logger.trace(
            f"Received event: {str(event.type)}, id: {benchmark.id}, name: {benchmark.name}, state: {str(benchmark.state)}"
        )
        is_pending = benchmark.state == BenchmarkStateEnum.PENDING
        is_stopped = benchmark.state == BenchmarkStateEnum.STOPPED

        is_current_worker = benchmark.worker_id == self._worker_id
        if not is_current_worker:
            return

        if event.type == EventType.DELETED:
            self._stop_benchmark(benchmark)
            logger.trace(
                f"DELETED event: stopped deleted benchmark {benchmark.name}(id={benchmark.id})."
            )
            return

        if is_pending:
            asyncio.create_task(self._enqueue_benchmark(benchmark))
            return

        if is_stopped:
            asyncio.create_task(self._handle_stop_benchmark_event(benchmark))

    async def _handle_stop_benchmark_event(self, benchmark: Benchmark):
        try:
            self._dump_benchmark_logs_to_file(benchmark)
            self._stop_benchmark(benchmark)
            self._clear_active_benchmark(benchmark.id)
        except Exception as e:
            logger.error(f"Failed to stop benchmark {benchmark.name}: {e}")

    async def _enqueue_benchmark(self, benchmark: Benchmark):
        async with self._queue_lock:
            if benchmark.id not in [b.id for b in self._benchmark_queue]:
                self._benchmark_queue.append(benchmark)

                patch_dict = {"state": BenchmarkStateEnum.QUEUED}
                await self._update_benchmark_state(benchmark.id, **patch_dict)
                logger.info(
                    f"Enqueued benchmark {benchmark.name}(id={benchmark.id}) and set to QUEUED."
                )

    async def _benchmark_queue_worker(self):
        """
        Process benchmarks in the queue.
        """
        while True:
            benchmark = None
            async with self._queue_lock:
                if self._active_benchmark_id is not None:
                    benchmark = None
                elif self._benchmark_queue:
                    benchmark = self._benchmark_queue.popleft()
            if benchmark:
                # Lifecycle gate: a benchmark referencing a custom Dataset
                # can be created before the dataset finishes downloading. Hold it in
                # the queue until READY; fail it if the dataset errors.
                gate = await self._check_dataset_ready(benchmark)
                if gate == "wait":
                    async with self._queue_lock:
                        self._benchmark_queue.append(benchmark)
                    await asyncio.sleep(3)
                    continue
                if gate == "error":
                    continue
                try:
                    await self._start_benchmark(benchmark)
                except Exception as e:
                    logger.error(
                        f"Failed to start benchmark {benchmark.name}(id={benchmark.id}): {e}"
                    )
            else:
                await asyncio.sleep(1)

    async def _check_dataset_ready(self, benchmark: Benchmark) -> str:
        """Return 'ready' / 'wait' / 'error' for the benchmark's custom dataset.

        Non-custom benchmarks are always 'ready'. On 'error' the benchmark has
        already been moved to ERROR.
        """
        if benchmark.dataset_name != DATASET_CUSTOM or benchmark.dataset_id is None:
            return "ready"
        try:
            dataset = await asyncio.to_thread(
                self._clientset.datasets.get, id=benchmark.dataset_id
            )
        except Exception as e:
            await self._update_benchmark_state(
                benchmark.id,
                state=BenchmarkStateEnum.ERROR,
                state_message=f"Dataset {benchmark.dataset_id} not found: {e}",
            )
            return "error"

        if dataset is None:
            await self._update_benchmark_state(
                benchmark.id,
                state=BenchmarkStateEnum.ERROR,
                state_message=f"Dataset {benchmark.dataset_id} not found",
            )
            return "error"
        if dataset.state == DatasetStateEnum.READY:
            return "ready"
        if dataset.state == DatasetStateEnum.ERROR:
            await self._update_benchmark_state(
                benchmark.id,
                state=BenchmarkStateEnum.ERROR,
                state_message=(
                    f"Dataset '{dataset.readable_source}' download failed: "
                    f"{dataset.state_message}"
                ),
            )
            return "error"
        return "wait"

    async def _start_benchmark(self, benchmark: Benchmark):
        """
        Start benchmark through a subprocess.
        Args:
            benchmark: The benchmark to start.
        """
        if benchmark.id in self._provisioning_processes:
            logger.warning(
                f"Benchmark {benchmark.name}(id={benchmark.id}) is provisioning. Skipping start."
            )
            return

        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        try:
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
        except Exception as e:
            logger.warning(f"Failed to remove old log file {log_file_path}: {e}")

        try:
            fallback_registry = registration.determine_default_registry(
                self._config.system_default_container_registry
            )
            process = multiprocessing.Process(
                target=BenchmarkManager._launch_benchmark,
                args=(
                    benchmark,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    fallback_registry,
                ),
            )
            process.daemon = False
            process.start()

            self._provisioning_processes[benchmark.id] = process
            self._set_active_benchmark(benchmark.id)
            patch_dict = {
                "state": BenchmarkStateEnum.RUNNING,
                "pid": process.pid,
            }
            await self._update_benchmark_state(benchmark.id, **patch_dict)
            logger.info(f"Started benchmark {benchmark.name}(id={benchmark.id})")

        except Exception as e:
            # Clean up provisioning process if started.
            if benchmark.id in self._provisioning_processes:
                self._stop_benchmark(benchmark)
            patch_dict = {
                "state": BenchmarkStateEnum.ERROR,
                "state_message": f"Failed to start benchmark: {e}",
            }
            await self._update_benchmark_state(benchmark.id, **patch_dict)
            logger.error(
                f"Failed to start benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )

    @staticmethod
    def _launch_benchmark(
        benchmark: Benchmark,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        """
        Serve benchmark in a subprocess.
        Exits the subprocess when serving ends.

        Args:
            benchmark: The benchmark to serve.
            client_headers: The headers for the clientset.
            log_file_path: The path to the log file.
            cfg: The configuration.
            fallback_registry: The fallback container registry to use if needed.
        """

        setproctitle.setproctitle(f"gpustack_benchmark_{benchmark.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.get_server_url(),
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_ins = BenchmarkRunner(
                        clientset,
                        benchmark,
                        cfg,
                        fallback_registry,
                    )
                    logger.info(
                        f"Provisioning benchmark {benchmark.name}(id={benchmark.id})"
                    )
                    server_ins.start()
                    logger.info(
                        f"Finished provisioning benchmark {benchmark.name}(id={benchmark.id})"
                    )
                except Exception as e:
                    logger.exception(
                        f"Error provisioning benchmark {benchmark.name}(id={benchmark.id}): {e}"
                    )
                    raise e

    async def _update_benchmark_state(self, id: int, **kwargs):
        client = self._clientset.http_client.get_async_httpx_client()
        resp = await client.patch(f"/benchmarks/{id}/state", json=kwargs)
        resp.raise_for_status()

    def _update_benchmark_state_sync(self, id: int, **kwargs):
        client = self._clientset.http_client.get_httpx_client()
        resp = client.patch(f"/benchmarks/{id}/state", json=kwargs)
        resp.raise_for_status()

    def _stop_benchmark(self, benchmark: Benchmark):
        """
        Stop benchmark and clean up.

        Args:
            benchmark: The benchmark to stop.
        """

        # Teardown provisioning process if still alive.
        if self._is_provisioning(benchmark):
            terminate_process_tree(self._provisioning_processes[benchmark.id].pid)

        # Delete workload.
        delete_workload(benchmark.name)

        # Cleanup internal states.
        self._provisioning_processes.pop(benchmark.id, None)
        self._benchmark_by_id.pop(benchmark.id, None)
        self._container_log_offset.pop(benchmark.id, None)
        self._last_log_snapshot_at.pop(benchmark.id, None)
        self._clear_active_benchmark(benchmark.id)

        logger.info(f"Stopped benchmark {benchmark.name}(id={benchmark.id})")

    def _is_provisioning(self, benchmark: Benchmark) -> bool:
        """
        Check if the benchmark is still provisioning.

        Args:
            benchmark: The benchmark to check.
        """
        if process := self._provisioning_processes.get(benchmark.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False

    def sync_benchmark_state(self):
        """
        Synchronize benchmarks' state.
        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, failed, update the benchmark state to ERROR.
        - If the workload is inactive, update the benchmark state to COMPLETED.
        """
        benchmarks_page = self._clientset.benchmarks.list(
            params={"worker_id": self._worker_id, "state": BenchmarkStateEnum.RUNNING}
        )
        if not benchmarks_page.items:
            return

        for benchmark in benchmarks_page.items:
            self._sync_single_benchmark_state(benchmark)

    def _sync_single_benchmark_state(self, benchmark: Benchmark):
        """Synchronize a single benchmark's state."""
        # Check for timeout
        if self._is_benchmark_timed_out(benchmark):
            self._handle_benchmark_timeout(benchmark)
            return

        # Skip if still provisioning
        if self._is_provisioning(benchmark):
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) is provisioning. Skipping sync."
            )
            return

        # Get workload and handle based on state
        workload = get_workload(benchmark.name)

        # Snapshot container logs while running, so we still have them if the
        # container is garbage-collected before we observe a terminal state.
        if workload and workload.state == WorkloadStatusStateEnum.RUNNING:
            self._maybe_snapshot_logs(benchmark)

        if self._should_skip_workload(benchmark, workload):
            return

        if self._is_workload_completed(workload):
            self._handle_benchmark_completion(benchmark)
            return

        if self._is_workload_failed(workload):
            self._handle_benchmark_failure(benchmark)
            return

    def _should_skip_workload(self, benchmark: Benchmark, workload) -> bool:
        """Check if workload should be skipped (still launching or running)."""
        if not workload:
            return False

        if workload.state in [
            WorkloadStatusStateEnum.PENDING,
            WorkloadStatusStateEnum.INITIALIZING,
        ]:
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) workload is still launching. Skipping sync."
            )
            return True

        if workload.state == WorkloadStatusStateEnum.RUNNING:
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) workload is running. Skipping sync."
            )
            return True

        return False

    def _is_workload_completed(self, workload) -> bool:
        """Check if workload has completed successfully."""
        return workload and workload.state == WorkloadStatusStateEnum.INACTIVE

    def _is_workload_failed(self, workload) -> bool:
        """Check if workload has failed or is unhealthy."""
        if not workload:
            return True
        return workload.state in [
            WorkloadStatusStateEnum.UNKNOWN,
            WorkloadStatusStateEnum.UNHEALTHY,
            WorkloadStatusStateEnum.FAILED,
        ]

    def _handle_benchmark_timeout(self, benchmark: Benchmark):
        """Handle benchmark timeout."""
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": "Benchmark timed out.",
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _handle_benchmark_completion(self, benchmark: Benchmark):
        """Handle successful benchmark completion."""
        patch_dict = {
            "state": BenchmarkStateEnum.COMPLETED,
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        logger.info(f"Benchmark {benchmark.name} finished.")

        self._dump_benchmark_logs_to_file(benchmark)
        self._sync_benchmark_metrics(benchmark)
        self._stop_benchmark(benchmark)

    def _handle_benchmark_failure(self, benchmark: Benchmark):
        """Handle benchmark failure."""
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": "Benchmark exited or unhealthy.",
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _sync_benchmark_metrics(self, benchmark):  # noqa: C901
        """
        Synchronize benchmarks' metrics.
        """
        metrics = None
        results = []
        report = None
        try:
            if benchmark.auto_tune:
                # Adaptive ramp: benchmark-runner writes one file per measured
                # point ({id}__p{index}.json), the point count decided at runtime.
                # Aggregate all points; representative = global throughput peak.
                # Missing/corrupt files are skipped (logged) so a bad point
                # doesn't drop the whole run.
                best = None
                prefix = f"{benchmark.id}__p"
                try:
                    names = [
                        n
                        for n in os.listdir(self._benchmark_dir)
                        if n.startswith(prefix)
                        and n.endswith(".json")
                        and not n.endswith(".full.json")
                    ]
                except Exception:
                    names = []

                def _point_index(name: str) -> int:
                    m = re.search(r"__p(\d+)\.json$", name)
                    return int(m.group(1)) if m else 0

                for name in sorted(names, key=_point_index):
                    path = f"{self._benchmark_dir}/{name}"
                    try:
                        rep = GenerativeBenchmarksReport.load_file(path)
                    except Exception as e:
                        logger.warning(
                            f"Skipping auto-tune point {name} of benchmark "
                            f"{benchmark.name}(id={benchmark.id}); result file "
                            f"unavailable: {e}"
                        )
                        continue
                    if report is None:
                        report = rep  # primary report for error samples
                    results.extend(
                        rep.to_results(input_tokens=benchmark.dataset_input_tokens)
                    )
                    m = rep.to_metrics()
                    if m and (
                        best is None
                        or (m.tokens_per_second_mean or 0)
                        > (best.tokens_per_second_mean or 0)
                    ):
                        best = m
                metrics = best
            elif benchmark.stages:
                # v2.1 stages: aggregate per-stage result files
                # ({id}__stage{i}.json), one single-rate run each.
                # Representative = global throughput peak. A missing/corrupt
                # stage file is skipped (logged) so the other stages still
                # report — a failure in one stage shouldn't drop the whole run.
                best = None
                for i in range(len(benchmark.stages)):
                    path = f"{self._benchmark_dir}/{benchmark.id}__stage{i}.json"
                    try:
                        rep = GenerativeBenchmarksReport.load_file(path)
                    except Exception as e:
                        logger.warning(
                            f"Skipping stage {i} of benchmark "
                            f"{benchmark.name}(id={benchmark.id}); result file "
                            f"unavailable: {e}"
                        )
                        continue
                    if report is None:
                        report = rep  # primary report for error samples
                    results.extend(
                        rep.to_results(input_tokens=benchmark.dataset_input_tokens)
                    )
                    m = rep.to_metrics()
                    if m and (
                        best is None
                        or (m.tokens_per_second_mean or 0)
                        > (best.tokens_per_second_mean or 0)
                    ):
                        best = m
                metrics = best
            else:
                metrics_file_path = f"{self._benchmark_dir}/{benchmark.id}.json"
                report = GenerativeBenchmarksReport.load_file(metrics_file_path)
                metrics = report.to_metrics()
                results = report.to_results(input_tokens=benchmark.dataset_input_tokens)
        except Exception as e:
            logger.error(
                f"Failed to load metrics for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
            return

        if not metrics:
            logger.error(
                f"No metrics found for benchmark {benchmark.name}(id={benchmark.id})."
            )
            return

        # Failure counts aggregate across ALL stages/points — a failure in any
        # stage should surface, not only the representative peak point. (For a
        # single-run benchmark `results` has one point, so this matches the old
        # behavior.)
        if results:
            total = sum(r.get("request_total") or 0 for r in results)
            successful = sum(r.get("request_successful") or 0 for r in results)
            errored = sum(r.get("request_errored") or 0 for r in results)
            incomplete = sum(r.get("request_incomplete") or 0 for r in results)
        else:
            total = metrics.request_total or 0
            successful = metrics.request_successful or 0
            errored = metrics.request_errored or 0
            incomplete = metrics.request_incomplete or 0

        try:
            errored_samples, incomplete_samples = self._load_request_samples(
                report, limit=None
            )
        except Exception as e:
            logger.error(
                "Failed to read request error samples for benchmark "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )
            errored_samples, incomplete_samples = [], []

        self._log_request_failures_if_any(
            benchmark=benchmark,
            total=total,
            successful=successful,
            errored=errored,
            incomplete=incomplete,
            errored_samples=errored_samples,
            incomplete_samples=incomplete_samples,
        )

        partial_failure_message = self._build_partial_failure_state_message(
            errored=errored,
            incomplete=incomplete,
            errored_samples=errored_samples,
            incomplete_samples=incomplete_samples,
        )

        resp = self._clientset.http_client.get_httpx_client().post(
            f"/benchmarks/{benchmark.id}/metrics", json=metrics.model_dump()
        )
        raise_if_response_error(resp)

        # Upload per-point results (one row per (input_tokens, rate) grid cell).
        # The parent metrics above hold the representative (throughput-peak) point.
        try:
            resp = self._clientset.http_client.get_httpx_client().post(
                f"/benchmarks/{benchmark.id}/results", json=results
            )
            raise_if_response_error(resp)
        except Exception as e:
            logger.error(
                "Failed to upload benchmark results for "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )

        # v2.1 best operating points: peak throughput / latency-throughput knee /
        # max rate meeting the SLA. Computed from the per-point grid and persisted
        # on the parent row for the detail page's "Best Operating Points" cards.
        best_points = self._compute_best_points(benchmark, results)
        # v2.1 test-coverage validity: whether the sweep explored enough to trust
        # the result (single source of truth on the parent; the UI just renders
        # the warning codes). See _compute_validity.
        validity = self._compute_validity(benchmark, results, best_points)
        patch = {**best_points, "validity": validity}
        try:
            self._update_benchmark_state_sync(benchmark.id, **patch)
        except Exception as e:
            logger.error(
                "Failed to update best operating points / validity for "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )

        # Surface partial failures (errored / incomplete requests) on the
        # benchmark's state_message so the UI shows why a run partly failed.
        if partial_failure_message:
            self._update_benchmark_state_sync(
                benchmark.id,
                state_message=partial_failure_message,
            )

    # SLA thresholds ("<=" ms). Each: (benchmark attr, point metric key, scale) —
    # a point's (metric * scale) must be <= the threshold. request_latency is
    # stored in seconds, so it scales to ms (x1000); TTFT/TPOT are already ms.
    # A point meets the SLA when EVERY set threshold holds (AND) + success ok.
    _SLA_CHECKS = [
        ("sla_avg_ttft_ms", "time_to_first_token_mean", 1.0),
        ("sla_p99_ttft_ms", "time_to_first_token_p99", 1.0),
        ("sla_avg_tpot_ms", "time_per_output_token_mean", 1.0),
        ("sla_p99_tpot_ms", "time_per_output_token_p99", 1.0),
        ("sla_avg_latency_ms", "request_latency_mean", 1000.0),
        ("sla_p99_latency_ms", "request_latency_p99", 1000.0),
    ]

    @staticmethod
    def _has_sla(benchmark) -> bool:
        return any(
            getattr(benchmark, attr, None)
            for attr, _, _ in BenchmarkManager._SLA_CHECKS
        )

    @staticmethod
    def _meets_sla(benchmark, r: dict) -> bool:
        """True iff every SET SLA threshold holds for this point (AND)."""
        for attr, key, scale in BenchmarkManager._SLA_CHECKS:
            thr = getattr(benchmark, attr, None)
            if thr is None:
                continue
            val = r.get(key)
            if val is None or val * scale > thr:
                return False
        return True

    @staticmethod
    def _success_ok(r: dict) -> bool:
        total = r.get("request_total") or 0
        if total <= 0:
            return False
        return (
            r.get("request_successful") or 0
        ) / total >= BenchmarkManager._MIN_SUCCESS_RATE

    @staticmethod
    def _compute_best_points(benchmark, results: list) -> dict:
        """Derive best operating points from the per-rate result grid.

        - peak_rate: rate at the global throughput peak.
        - knee_rate: rate at the latency-throughput knee (best balance).
        - sla_met_rate / recommended_rate: when SLA targets are set, the max rate
          whose TTFT/TPOT both stay within the SLA thresholds (ms).
        """
        points = [
            r
            for r in results
            if r.get("rate") is not None and r.get("tokens_per_second_mean") is not None
        ]
        if not points:
            return {}
        points = sorted(points, key=lambda r: r["rate"])
        out: dict = {}

        peak = max(points, key=lambda r: r.get("tokens_per_second_mean") or 0)
        out["peak_rate"] = float(peak["rate"])

        knee = BenchmarkManager._find_knee(points)
        if knee is not None:
            out["knee_rate"] = float(knee["rate"])

        if BenchmarkManager._has_sla(benchmark):
            met = [
                r
                for r in points
                if BenchmarkManager._meets_sla(benchmark, r)
                and BenchmarkManager._success_ok(r)
            ]
            if met:
                sla_rate = float(max(met, key=lambda r: r["rate"])["rate"])
                out["sla_met_rate"] = sla_rate
                out["recommended_rate"] = sla_rate
        else:
            # No SLA => the user asked for maximum throughput and stated no latency
            # budget, so recommend the throughput peak (matches the Max Throughput
            # profile and the runner's argmax search). knee_rate is still computed
            # above and surfaced as an informational "balanced" point on the chart;
            # it is deliberately NOT the recommendation. Points past the peak are
            # already excluded by the overload / throughput-drop guards, so no extra
            # latency guard is needed here.
            out["recommended_rate"] = out.get("peak_rate")

        return out

    # A sampled point's success rate below this = overloaded / not trustworthy.
    _MIN_SUCCESS_RATE = 0.95

    @staticmethod
    def _compute_validity(benchmark, results: list, best_points: dict) -> dict:
        """Judge whether the adaptive ramp explored enough to trust the result.

        Returns ``{"sufficient": bool, "warnings": [{"code", "params"}]}``. Codes
        (rendered/localized by the UI), inferred from the measured point grid:
        - ``sla_never_met``: SLA targets set but no measured point meets them ->
          the server is too slow for this SLA; no usable capacity.
        - ``not_saturated``: recommended == the highest measured knob and no point
          overloaded -> the true optimum may be higher; extend bounds / budget.
        - ``point_high_error``: some point's success rate < 95% -> overloaded /
          unreliable at that load (already flagged red in the table).
        - ``few_points``: too few measured points (< 3, no SLA) to trust the curve.
        """
        warnings: list = []

        rate_points = [r for r in results if r.get("rate") is not None]
        has_sla = BenchmarkManager._has_sla(benchmark)

        # Any measured point that overloaded (low success rate).
        overloaded_any = False
        worst_ok = None
        for r in rate_points:
            total = r.get("request_total") or 0
            if total <= 0:
                continue
            ok = (r.get("request_successful") or 0) / total
            worst_ok = ok if worst_ok is None else min(worst_ok, ok)
            if ok < BenchmarkManager._MIN_SUCCESS_RATE:
                overloaded_any = True
        if overloaded_any and worst_ok is not None:
            warnings.append(
                {"code": "point_high_error", "params": {"rate": round(worst_ok * 100)}}
            )

        if has_sla and best_points.get("sla_met_rate") is None:
            # SLA set but nothing met it — even the lowest load is too slow.
            warnings.append({"code": "sla_never_met", "params": {}})

        rec = best_points.get("recommended_rate")
        # "Could go higher" only makes sense when nothing overloaded — if a point
        # already overloaded we hit the ceiling, so don't also say "test higher".
        if not overloaded_any and rec is not None and rate_points:
            max_rate = max(r["rate"] for r in rate_points)
            if rec == max_rate:
                warnings.append({"code": "not_saturated", "params": {}})

        if not has_sla and 0 < len(rate_points) < 3:
            warnings.append({"code": "few_points", "params": {}})

        return {"sufficient": len(warnings) == 0, "warnings": warnings}

    @staticmethod
    def _find_knee(points: list) -> Optional[dict]:
        """Latency-throughput knee: the last point before latency starts rising
        sharply relative to throughput gains. Walks rate ascending and returns
        the point right before Δlatency/Δthroughput first exceeds 2× the mean.

        Latency here is TPOT (time per output token), matching the "Throughput vs
        Latency" decision chart's x-axis: for a max-throughput sweep the decode
        speed (TPOT) is the throughput-relevant latency, whereas TTFT mostly
        reflects prefill queueing."""
        if len(points) < 3:
            return None
        ratios = []
        for a, b in zip(points, points[1:]):
            dl = (b.get("time_per_output_token_mean") or 0) - (
                a.get("time_per_output_token_mean") or 0
            )
            dt = (b.get("tokens_per_second_mean") or 0) - (
                a.get("tokens_per_second_mean") or 0
            )
            ratios.append(dl / dt if dt > 0 else float("inf"))
        finite = [r for r in ratios if r != float("inf")]
        if not finite:
            return None
        threshold = (sum(finite) / len(finite)) * 2.0
        for idx, r in enumerate(ratios):
            if r > threshold:
                return points[idx]  # point before the sharp rise
        return None

    def _log_request_failures_if_any(
        self,
        benchmark: Benchmark,
        total: int,
        successful: int,
        errored: int,
        incomplete: int,
        errored_samples: List[GenerativeRequestStats],
        incomplete_samples: List[GenerativeRequestStats],
        limit: int = 5,
    ) -> None:
        if errored <= 0 and incomplete <= 0:
            return

        errored_samples_to_show = errored_samples[:limit]
        incomplete_samples_to_show = incomplete_samples[:limit]

        if not errored_samples_to_show and not incomplete_samples_to_show:
            return

        lines: List[str] = [
            "",
            "=== BENCHMARK REQUEST FAILURES ===",
            "SUMMARY: "
            f"benchmark={benchmark.name}(id={benchmark.id}) "
            f"total={total} successful={successful} "
            f"errored={errored} incomplete={incomplete} "
            f"showing_up_to={limit}",
        ]

        if errored_samples_to_show:
            lines.append("")
            lines.append(f"---- ERRORED REQUESTS (SHOWING UP TO {limit}) ----")
            lines.extend(self._format_request_samples(errored_samples_to_show))

        if incomplete_samples_to_show:
            lines.append("")
            lines.append(f"---- INCOMPLETE REQUESTS (SHOWING UP TO {limit}) ----")
            lines.extend(self._format_request_samples(incomplete_samples_to_show))

        message = "\n".join(lines)
        self._append_benchmark_log(benchmark, message)

    def _load_request_samples(
        self, report: GenerativeBenchmarksReport, limit: Optional[int] = 5
    ) -> Tuple[List[GenerativeRequestStats], List[GenerativeRequestStats]]:
        if (
            not report.benchmarks
            or len(report.benchmarks) == 0
            or report.benchmarks[0] is None
            or report.benchmarks[0].requests_truncated is None
        ):
            return [], []

        requests = report.benchmarks[0].requests_truncated
        errored = requests.errored or []
        incomplete = requests.incomplete or []

        if limit is None:
            return errored, incomplete

        return errored[:limit], incomplete[:limit]

    def _format_request_samples(
        self, samples: List[GenerativeRequestStats]
    ) -> List[str]:
        lines: List[str] = []
        for idx, sample in enumerate(samples, start=1):
            request_id = sample.request_id or "unknown"
            request_type = sample.request_type or "unknown"
            status = sample.info.status or "unknown"
            error = sample.info.error
            traceback = sample.info.traceback

            base = (
                f"- [{idx}] request_id={request_id} type={request_type} "
                f"status={status}"
            )
            lines.append(base)

            if error:
                lines.append(f"  ERROR: {error}")
            if traceback:
                lines.append("  TRACEBACK:")
                indented = "\n".join(f"    {line}" for line in traceback.splitlines())
                lines.append(indented)
            lines.append("")
        return lines

    def _build_partial_failure_state_message(
        self,
        errored: int,
        incomplete: int,
        errored_samples: List[GenerativeRequestStats],
        incomplete_samples: List[GenerativeRequestStats],
        top_n: int = 3,
    ) -> Optional[str]:
        if errored <= 0 and incomplete <= 0:
            return None

        summary = (
            "Completed with partial success: "
            f"errored={errored}, incomplete={incomplete}."
        )

        errored_reasons = self._collect_failure_reasons(
            errored_samples, fallback="Errored"
        )
        incomplete_reasons = self._collect_failure_reasons(
            incomplete_samples, fallback="Incomplete"
        )

        reason_parts: List[str] = []
        if errored_reasons:
            top_errored = ", ".join(
                f"{reason} (x{count})"
                for reason, count in errored_reasons.most_common(top_n)
            )
            reason_parts.append(f"Top errored reasons: {top_errored}")

        if incomplete_reasons:
            top_incomplete = ", ".join(
                f"{reason} (x{count})"
                for reason, count in incomplete_reasons.most_common(top_n)
            )
            reason_parts.append(f"Top incomplete reasons: {top_incomplete}")

        if reason_parts:
            summary = f"{summary} {'; '.join(reason_parts)}"
        else:
            summary = f"{summary} See benchmark logs for details."

        return self._truncate_state_message(summary)

    def _collect_failure_reasons(
        self, samples: List[GenerativeRequestStats], fallback: str
    ) -> Counter[str]:
        reasons: Counter[str] = Counter()
        for sample in samples:
            error = sample.info.error
            if error:
                reason = self._normalize_error_message(error)
            else:
                status = sample.info.status or "unknown"
                reason = f"{fallback} request (status={status})"
            reasons[reason] += 1
        return reasons

    def _normalize_error_message(self, error: str) -> str:
        stripped = error.strip()
        if not stripped:
            return "Unknown error"

        first_line = stripped.splitlines()[0]
        match = HTTP_ERROR_PATTERN.match(first_line)
        if not match:
            return first_line

        status = match.group("status")
        msg = " ".join(match.group("msg").split())
        error_type = match.group("type").strip()
        code = match.group("code").strip()

        if code and code.lower() != "none":
            normalized = f"HTTP {status} {error_type}/{code}: {msg}"
        else:
            normalized = f"HTTP {status} {error_type}: {msg}"

        return self._truncate_with_ellipsis(
            normalized, BENCHMARK_FAILURE_REASON_MAX_LEN
        )

    def _truncate_state_message(self, message: str) -> str:
        return self._truncate_with_ellipsis(message, BENCHMARK_STATE_MESSAGE_MAX_LEN)

    def _truncate_with_ellipsis(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        if max_len <= len(TRUNCATION_SUFFIX):
            return TRUNCATION_SUFFIX[:max_len]
        return text[: max_len - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX

    def _append_benchmark_log(self, benchmark: Benchmark, message: str) -> None:
        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(message)
                if not message.endswith("\n"):
                    f.write("\n")
        except Exception as e:
            logger.error(
                f"Failed to append benchmark log for {benchmark.name}(id={benchmark.id}): {e}"
            )

    def _set_active_benchmark(self, benchmark_id: int):
        self._active_benchmark_id = benchmark_id
        self._active_benchmark_started_at = time.time()

    def _clear_active_benchmark(self, benchmark_id: int):
        if self._active_benchmark_id == benchmark_id:
            self._active_benchmark_id = None
            self._active_benchmark_started_at = None

    def _is_benchmark_timed_out(self, benchmark: Benchmark) -> bool:
        limit = self._config.benchmark_max_duration_seconds
        if not limit:
            return False
        if self._active_benchmark_id != benchmark.id:
            return False
        if self._active_benchmark_started_at is None:
            return False
        return (time.time() - self._active_benchmark_started_at) > limit

    def _maybe_snapshot_logs(self, benchmark: Benchmark):
        """Throttled log snapshot for a running benchmark (see
        BENCHMARK_LOG_SNAPSHOT_INTERVAL_SECONDS)."""
        last = self._last_log_snapshot_at.get(benchmark.id, 0.0)
        now = time.time()
        if now - last < BENCHMARK_LOG_SNAPSHOT_INTERVAL_SECONDS:
            return
        self._last_log_snapshot_at[benchmark.id] = now
        self._dump_benchmark_logs_to_file(benchmark)

    def _dump_benchmark_logs_to_file(
        self,
        benchmark: Benchmark,
    ):
        """Write the container's (full) logs to the benchmark log file.

        The provisioning subprocess already wrote its own logs to the same file;
        the container logs are (re)written after that boundary. `logs_workload`
        returns the full log each call, so we truncate back to the recorded
        boundary and rewrite — making repeated snapshots idempotent while
        preserving the provisioning logs.
        """
        try:
            logs = logs_workload(name=benchmark.name)
        except Exception as e:
            logger.error(
                f"Failed to fetch workload logs for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
            return
        if logs is None:
            return

        log_str = logs
        if isinstance(log_str, (bytes, bytearray)):
            log_str = log_str.decode("utf-8", errors="replace")
        log_str = str(log_str)

        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        try:
            size = (
                os.path.getsize(log_file_path) if os.path.exists(log_file_path) else 0
            )
            # Boundary = end of the provisioning logs, captured on first snapshot.
            offset = self._container_log_offset.get(benchmark.id)
            if offset is None:
                offset = size
                self._container_log_offset[benchmark.id] = offset
            offset = min(offset, size)  # guard against a shrunk/recreated file

            mode = "r+" if os.path.exists(log_file_path) else "w"
            with open(log_file_path, mode, encoding="utf-8") as f:
                f.seek(offset)
                f.truncate()
                if offset > 0:
                    f.write("\n---- Benchmark container logs ----\n")
                f.write(log_str)
                if not log_str.endswith("\n"):
                    f.write("\n")
        except Exception as e:
            logger.error(
                f"Failed to write workload logs for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
