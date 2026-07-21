"""The MVP's only controlled background worker thread."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Settings
from .db import sqlite_connection
from .operations import IngestionService, prepare_ingestion_job
from .repositories import JobRepository

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._parser_pool = ThreadPoolExecutor(
            max_workers=settings.parse_workers,
            thread_name_prefix="document-parser",
        )
        self._thread = threading.Thread(
            target=self._run,
            name="document-ingestion-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._thread.join(timeout=10)
        self._parser_pool.shutdown(wait=True, cancel_futures=True)

    def _run(self) -> None:
        leases_recovered = False
        while not self._stop_event.is_set():
            try:
                if not leases_recovered:
                    self._recover_stale_leases()
                    leases_recovered = True
                worked = self._process_one_job()
            except Exception:
                logger.exception("ingestion worker iteration failed; retrying")
                self._wake_event.wait(timeout=2)
                self._wake_event.clear()
                continue
            if not worked:
                self._wake_event.wait(timeout=2)
                self._wake_event.clear()

    def _recover_stale_leases(self) -> int:
        """Return jobs abandoned by a previous worker process to the queue."""
        with sqlite_connection(self.settings) as connection:
            return JobRepository(connection).release_leases()

    def _process_one_job(self) -> bool:
        with sqlite_connection(self.settings) as connection:
            jobs = JobRepository(connection)
            jobs.release_leases()
            runnable = jobs.list_runnable()
            if not runnable:
                return False
            if runnable[0].operation.lower() not in {"add", "update", "reindex"}:
                service = IngestionService(connection=connection, settings=self.settings)
                try:
                    service.process_job(job_id=runnable[0].id)
                finally:
                    service.close()
                return True
            leased = [
                jobs.lease(job_id=job.id, lease_owner="document-ingestion-writer")
                for job in runnable[: self.settings.ingestion_batch_size]
            ]

        futures = {
            self._parser_pool.submit(prepare_ingestion_job, job=job, settings=self.settings): job
            for job in leased
        }
        with sqlite_connection(self.settings) as connection:
            service = IngestionService(connection=connection, settings=self.settings)
            try:
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        prepared = future.result()
                    except Exception as exc:
                        service.fail_prepared_job(job=job, error=_safe_error(exc))
                        continue
                    service.write_prepared_jobs([prepared])
            finally:
                service.close()
        return True


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]
