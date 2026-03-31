import asyncio
import json
import logging
import multiprocessing
import os
import time
from collections import deque
from typing import Callable, Dict, Optional

import httpx
import setproctitle

from gpustack.client import ClientSet
from gpustack.config import registration
from gpustack.config.config import Config
from gpustack.logging import RedirectStdoutStderr
from gpustack.schemas.evaluations import (
    Evaluation,
    EvaluationResult,
    EvaluationStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.utils.process import add_signal_handlers, terminate_process_tree
from gpustack.worker.evaluation.result_parser import parse_evaluation_results
from gpustack.worker.evaluation.runner import EvaluationRunner
from gpustack_runtime.deployer import (
    WorkloadStatusStateEnum,
    delete_workload,
    get_workload,
)

logger = logging.getLogger(__name__)


class EvaluationManager:
    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._clientset_getter = clientset_getter
        self._config = cfg
        self._evaluation_log_dir = f"{cfg.log_dir}/evaluations"
        self._evaluation_dir = f"{cfg.evaluation_dir}"

        self._provisioning_processes: Dict[int, multiprocessing.Process] = {}
        self._evaluation_queue = deque()
        self._queue_lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._active_evaluation_id: Optional[int] = None
        self._active_evaluation_started_at: Optional[float] = None

        os.makedirs(self._evaluation_log_dir, exist_ok=True)
        os.makedirs(self._evaluation_dir, exist_ok=True)

    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    async def watch_evaluations_event(self):
        logger.info("Watching evaluations event.")
        if not self._worker_task or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._evaluation_queue_worker())

        while True:
            try:
                await self._awatch(callback=self._handle_evaluation_event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching evaluations: {e}")
                await asyncio.sleep(5)

    async def _awatch(self, callback):
        client = self._clientset.http_client.get_async_httpx_client()
        async with client.stream(
            "GET",
            "/evaluations",
            params={"watch": "true", "worker_id": self._worker_id},
            timeout=httpx.Timeout(connect=10, read=None, write=10, pool=10),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = Event(**json.loads(line))
                await callback(event)

    async def _handle_evaluation_event(self, event: Event):
        evaluation = Evaluation.model_validate(event.data)
        if evaluation.worker_id != self._worker_id:
            return

        if event.type == EventType.DELETED:
            self._stop_evaluation(evaluation)
            return

        if evaluation.state == EvaluationStateEnum.PENDING:
            await self._enqueue_evaluation(evaluation)
            return

        if evaluation.state == EvaluationStateEnum.STOPPED:
            self._stop_evaluation(evaluation)
            self._clear_active_evaluation(evaluation.id)

    async def _enqueue_evaluation(self, evaluation: Evaluation):
        async with self._queue_lock:
            if evaluation.id not in [item.id for item in self._evaluation_queue]:
                self._evaluation_queue.append(evaluation)
                await self._update_evaluation_state(
                    evaluation.id, state=EvaluationStateEnum.QUEUED
                )

    async def _evaluation_queue_worker(self):
        while True:
            evaluation = None
            async with self._queue_lock:
                if self._active_evaluation_id is None and self._evaluation_queue:
                    evaluation = self._evaluation_queue.popleft()
            if evaluation is None:
                await asyncio.sleep(1)
                continue
            await self._start_evaluation(evaluation)

    async def _start_evaluation(self, evaluation: Evaluation):
        log_file_path = f"{self._evaluation_log_dir}/{evaluation.id}.log"
        try:
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
        except Exception:
            pass

        fallback_registry = registration.determine_default_registry(
            self._config.system_default_container_registry
        )
        process = multiprocessing.Process(
            target=EvaluationManager._launch_evaluation,
            args=(
                evaluation,
                self._clientset.headers,
                log_file_path,
                self._config,
                fallback_registry,
            ),
        )
        process.daemon = False
        process.start()
        self._provisioning_processes[evaluation.id] = process
        self._set_active_evaluation(evaluation.id)
        await self._update_evaluation_state(
            evaluation.id,
            state=EvaluationStateEnum.RUNNING,
            pid=process.pid,
        )

    @staticmethod
    def _launch_evaluation(
        evaluation: Evaluation,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        setproctitle.setproctitle(f"gpustack_evaluation_{evaluation.id}")
        add_signal_handlers()

        clientset = ClientSet(base_url=cfg.get_server_url(), headers=client_headers)
        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                EvaluationRunner(
                    clientset,
                    evaluation,
                    cfg,
                    fallback_registry,
                ).start()

    async def _update_evaluation_state(self, id: int, **kwargs):
        client = self._clientset.http_client.get_async_httpx_client()
        response = await client.patch(f"/evaluations/{id}/state", json=kwargs)
        response.raise_for_status()

    def _update_evaluation_state_sync(self, id: int, **kwargs):
        client = self._clientset.http_client.get_httpx_client()
        response = client.patch(f"/evaluations/{id}/state", json=kwargs)
        response.raise_for_status()

    def _post_evaluation_result_sync(self, id: int, payload: EvaluationResult):
        client = self._clientset.http_client.get_httpx_client()
        response = client.post(f"/evaluations/{id}/result", json=payload.model_dump())
        response.raise_for_status()

    def sync_evaluation_state(self):
        client = self._clientset.http_client.get_httpx_client()
        response = client.get(
            "/evaluations",
            params={"worker_id": self._worker_id, "state": EvaluationStateEnum.RUNNING},
        )
        response.raise_for_status()
        items = response.json().get("items") or []
        for item in items:
            self._sync_single_evaluation_state(Evaluation.model_validate(item))

    def _sync_single_evaluation_state(self, evaluation: Evaluation):
        if self._is_evaluation_timed_out(evaluation):
            self._handle_evaluation_timeout(evaluation)
            return

        if self._is_provisioning(evaluation):
            return

        workload = get_workload(evaluation.name)
        if workload and workload.state in [
            WorkloadStatusStateEnum.PENDING,
            WorkloadStatusStateEnum.INITIALIZING,
            WorkloadStatusStateEnum.RUNNING,
        ]:
            return

        if workload and workload.state == WorkloadStatusStateEnum.INACTIVE:
            self._handle_evaluation_completion(evaluation)
            return

        if workload is None or workload.state in [
            WorkloadStatusStateEnum.UNKNOWN,
            WorkloadStatusStateEnum.UNHEALTHY,
            WorkloadStatusStateEnum.FAILED,
        ]:
            self._handle_evaluation_failure(evaluation)

    def _handle_evaluation_timeout(self, evaluation: Evaluation):
        self._update_evaluation_state_sync(
            evaluation.id,
            state=EvaluationStateEnum.ERROR,
            state_message="Evaluation timed out.",
        )
        self._stop_evaluation(evaluation)

    def _handle_evaluation_completion(self, evaluation: Evaluation):
        self._sync_evaluation_result(evaluation)
        self._update_evaluation_state_sync(
            evaluation.id,
            state=EvaluationStateEnum.COMPLETED,
            progress=1.0,
        )
        self._stop_evaluation(evaluation)

    def _handle_evaluation_failure(self, evaluation: Evaluation):
        self._update_evaluation_state_sync(
            evaluation.id,
            state=EvaluationStateEnum.ERROR,
            state_message="Evaluation exited or unhealthy.",
        )
        self._stop_evaluation(evaluation)

    def _sync_evaluation_result(self, evaluation: Evaluation):
        result_file_path = f"{self._evaluation_dir}/{evaluation.id}.json"
        with open(result_file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        result = parse_evaluation_results(payload)
        self._post_evaluation_result_sync(evaluation.id, result)

    def _is_provisioning(self, evaluation: Evaluation) -> bool:
        if process := self._provisioning_processes.get(evaluation.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False

    def _stop_evaluation(self, evaluation: Evaluation):
        if self._is_provisioning(evaluation):
            terminate_process_tree(self._provisioning_processes[evaluation.id].pid)
        delete_workload(evaluation.name)
        self._provisioning_processes.pop(evaluation.id, None)
        self._clear_active_evaluation(evaluation.id)

    def _set_active_evaluation(self, evaluation_id: int):
        self._active_evaluation_id = evaluation_id
        self._active_evaluation_started_at = time.time()

    def _clear_active_evaluation(self, evaluation_id: Optional[int]):
        if self._active_evaluation_id == evaluation_id:
            self._active_evaluation_id = None
            self._active_evaluation_started_at = None

    def _is_evaluation_timed_out(self, evaluation: Evaluation) -> bool:
        timeout = self._config.evaluation_max_duration_seconds
        if timeout is None or self._active_evaluation_id != evaluation.id:
            return False
        if self._active_evaluation_started_at is None:
            return False
        return time.time() - self._active_evaluation_started_at > timeout
