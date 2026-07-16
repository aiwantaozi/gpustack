import asyncio
import csv
import glob
import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from multiprocessing import Manager, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gpustack.api.exceptions import NotFoundException
from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.config.registration import read_worker_token
from gpustack.logging import setup_logging
from gpustack.schemas.datasets import Dataset, DatasetStateEnum, DatasetUpdate
from gpustack.server.bus import Event, EventType
from gpustack.utils.file import delete_path
from gpustack.worker import downloaders

logger = logging.getLogger(__name__)

max_concurrent_downloads = 3

# Preview knobs: keep the stored preview small (it rides in the DB row
# and the API response).
_SAMPLE_ROWS = 5
_CELL_MAX_CHARS = 500
_FILE_EXT_KIND = {
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".txt": "text",
    ".text": "text",
    ".parquet": "parquet",
    ".arrow": "arrow",
}


class DatasetManager:
    """Worker-side download orchestrator for benchmark datasets.

    Mirrors ``ModelFileManager`` (watch events → download in a process pool →
    report state) but is deliberately lighter: no per-file tqdm log rendering,
    plus a schema-probe step on completion that fills ``columns``/``sample_rows``
    for the column-mapping UI.
    """

    def __init__(self, worker_id: int, clientset: ClientSet, cfg: Config):
        self._worker_id = worker_id
        self._config = cfg
        self._clientset = clientset
        self._active_downloads: Dict[int, Tuple] = {}
        self._download_pool: Optional[ProcessPoolExecutor] = None

    async def watch_datasets(self):
        self._prerun()
        while True:
            try:
                logger.debug("Started watching datasets.")
                await self._clientset.datasets.awatch(
                    callback=self._handle_dataset_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed to watch datasets: {e}")
                await asyncio.sleep(5)

    def _prerun(self):
        self._mp_manager = Manager()
        self._download_pool = ProcessPoolExecutor(
            max_workers=min(max_concurrent_downloads, cpu_count()),
        )

    def _handle_dataset_event(self, event: Event):
        dataset = Dataset.model_validate(event.data)

        if dataset.worker_id != self._worker_id:
            return

        logger.trace(
            f"Received dataset event: {event.type} {dataset.id} {dataset.state}"
        )

        if event.type == EventType.DELETED:
            asyncio.create_task(self._handle_deletion(dataset))
        elif event.type in {EventType.CREATED, EventType.UPDATED}:
            if dataset.state != DatasetStateEnum.DOWNLOADING:
                return
            self._create_download_task(dataset)

    def _update_dataset(self, id: int, **kwargs):
        dataset_public = self._clientset.datasets.get(id=id)
        dataset_update = DatasetUpdate(**dataset_public.model_dump())
        for key, value in kwargs.items():
            setattr(dataset_update, key, value)
        self._clientset.datasets.update(id=id, model_update=dataset_update)

    async def _handle_deletion(self, dataset: Dataset):
        entry = self._active_downloads.pop(dataset.id, None)
        if entry:
            future, cancel_flag = entry
            cancel_flag.set()
            future.cancel()
            try:
                await asyncio.wrap_future(future)
            except (asyncio.CancelledError, NotFoundException):
                pass
            except Exception as e:
                logger.error(
                    f"Error while cancelling download for "
                    f"{dataset.readable_source}(id: {dataset.id}): {e}"
                )
            finally:
                logger.info(
                    f"Cancelled download for deleted dataset "
                    f"{dataset.readable_source}(id: {dataset.id})"
                )

        # local_path datasets are user-managed; never delete them on our behalf.
        if dataset.cleanup_on_delete and dataset.source != "local_path":
            await self._delete_dataset_files(dataset)

    async def _delete_dataset_files(self, dataset: Dataset):
        try:
            if dataset.resolved_paths:
                paths = chain.from_iterable(
                    glob.glob(p) if '*' in p else [p] for p in dataset.resolved_paths
                )
                for path in paths:
                    delete_path(path)
            logger.info(
                f"Deleted dataset {dataset.readable_source}(id: {dataset.id}) from disk"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete dataset "
                f"{dataset.readable_source}(id: {dataset.id}): {e}"
            )

    def _create_download_task(self, dataset: Dataset):
        if dataset.id in self._active_downloads:
            return

        cancel_flag = self._mp_manager.Event()
        download_task = DatasetDownloadTask(dataset, self._config, cancel_flag)
        future = self._download_pool.submit(download_task.run)
        self._active_downloads[dataset.id] = (future, cancel_flag)

        logger.debug(f"Created download task for {dataset.readable_source}")

        async def _check_completion():
            try:
                await asyncio.wrap_future(future)
            except NotFoundException:
                logger.info(
                    f"Dataset {dataset.readable_source} not found. Maybe cancelled."
                )
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                self._update_dataset(
                    dataset.id,
                    state=DatasetStateEnum.ERROR,
                    state_message=str(e),
                )
            finally:
                self._active_downloads.pop(dataset.id, None)
            logger.debug(f"Download completed for {dataset.readable_source}")

        asyncio.create_task(_check_completion())


class DatasetDownloadTask:
    def __init__(self, dataset: Dataset, cfg: Config, cancel_flag):
        self._dataset = dataset
        self._config = cfg
        self._cancel_flag = cancel_flag

    def prerun(self):
        setup_logging(self._config.debug)
        self._clientset = ClientSet(
            base_url=self._config.get_server_url(),
            api_key=read_worker_token(self._config.data_dir),
        )

    def run(self):
        try:
            self.prerun()
            logger.info(f"Downloading dataset: {self._dataset.readable_source}")
            path = downloaders.download_dataset(
                self._dataset,
                local_dir=self._dataset.local_dir,
                cache_dir=self._config.cache_dir,
                huggingface_token=self._config.huggingface_token,
            )
            if self._cancel_flag.is_set():
                raise asyncio.CancelledError("Download cancelled")

            columns, sample_rows, inspect_error = _inspect_dataset(path)

            self._update_dataset(
                self._dataset.id,
                state=DatasetStateEnum.READY,
                download_progress=100,
                resolved_paths=[path],
                columns=columns,
                sample_rows=sample_rows,
                inspect_error=inspect_error,
            )
            logger.info(f"Successfully downloaded {self._dataset.readable_source}")
        except asyncio.CancelledError:
            logger.info(f"Download task cancelled: {self._dataset.readable_source}")
        except Exception as e:
            logger.error(f"Download task failed: {self._dataset.readable_source} - {e}")
            self._update_dataset(
                self._dataset.id,
                state=DatasetStateEnum.ERROR,
                state_message=str(e),
            )

    def _update_dataset(self, id: int, **kwargs):
        dataset_public = self._clientset.datasets.get(id=id)
        dataset_update = DatasetUpdate(**dataset_public.model_dump())
        for key, value in kwargs.items():
            setattr(dataset_update, key, value)
        self._clientset.datasets.update(id=id, model_update=dataset_update)


def _truncate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str) and len(v) > _CELL_MAX_CHARS:
            out[k] = v[:_CELL_MAX_CHARS] + "…"
        elif isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
            out[k] = s[:_CELL_MAX_CHARS] + ("…" if len(s) > _CELL_MAX_CHARS else "")
        else:
            out[k] = v
    return out


def _inspect_dataset(
    path: str,
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[str]]:
    """Best-effort schema probe: (columns, sample_rows, error).

    Reads only the first rows — never materializes the whole dataset. On failure
    returns ``(None, None, error)``; the dataset is still usable (guidellm auto-
    detects columns, the UI falls back to manual column entry).
    """
    try:
        p = Path(path)
        if p.is_file():
            return _inspect_file(p)
        return _inspect_hf_dir(path)
    except Exception as e:  # noqa: BLE001 - probing must never be fatal
        return None, None, str(e)


def _inspect_file(
    p: Path,
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[str]]:
    ext = p.suffix.lower()
    kind = _FILE_EXT_KIND.get(ext)

    if kind == "json":
        rows: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            head = f.read(2048).lstrip()
            f.seek(0)
            if head.startswith("["):
                data = json.load(f)
                rows = [r for r in data[:_SAMPLE_ROWS] if isinstance(r, dict)]
            else:  # jsonl
                for line in itertools.islice(f, _SAMPLE_ROWS):
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(obj)
        columns = list(rows[0].keys()) if rows else []
        return columns, [_truncate_row(r) for r in rows], None

    if kind == "csv":
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(itertools.islice(reader, _SAMPLE_ROWS))
        return columns, [_truncate_row(dict(r)) for r in rows], None

    if kind == "text":
        with open(p, "r", encoding="utf-8") as f:
            rows = [
                {"text": line.rstrip("\n")}
                for line in itertools.islice(f, _SAMPLE_ROWS)
            ]
        return ["text"], rows, None

    if kind in ("parquet", "arrow"):
        import pyarrow.parquet as pq  # lazy import

        if kind == "parquet":
            table = pq.read_table(p)
        else:
            import pyarrow as pa

            with pa.memory_map(str(p), "r") as source:
                table = pa.ipc.open_file(source).read_all()
        columns = list(table.column_names)
        sample = table.slice(0, _SAMPLE_ROWS).to_pylist()
        return columns, [_truncate_row(r) for r in sample], None

    return None, None, f"Unsupported dataset file type: {ext or p.name}"


def _inspect_hf_dir(
    path: str,
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[str]]:
    from datasets import load_dataset  # lazy import

    # Whole-dataset mode: default config, no split (a DatasetDict is handled below).
    ds = load_dataset(path, streaming=True)
    # A DatasetDict (no split given) is not directly iterable; pick the first split.
    if hasattr(ds, "keys") and not hasattr(ds, "features"):
        first_key = next(iter(ds.keys()))
        ds = ds[first_key]

    sample = list(itertools.islice(ds, _SAMPLE_ROWS))
    columns: List[str] = []
    features = getattr(ds, "features", None)
    if features:
        columns = list(features.keys())
    elif sample:
        columns = list(sample[0].keys())
    return columns, [_truncate_row(r) for r in sample], None
