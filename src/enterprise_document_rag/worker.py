"""The MVP's only controlled background worker thread."""

from __future__ import annotations

import threading

from .config import Settings
from .db import sqlite_connection
from .operations import IngestionService
from .repositories import JobRepository


class IngestionWorker:
    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
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

    def _run(self) -> None:
        while not self._stop_event.is_set():
            worked = self._process_one_job()
            if not worked:
                self._wake_event.wait(timeout=2)
                self._wake_event.clear()

    def _process_one_job(self) -> bool:
        with sqlite_connection(self.settings) as connection:
            jobs = JobRepository(connection)
            jobs.release_leases()
            runnable = jobs.list_runnable()
            if not runnable:
                return False
            service = IngestionService(connection=connection, settings=self.settings)
            try:
                service.process_job(job_id=runnable[0].id)
            finally:
                service.close()
        return True
