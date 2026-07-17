"""First-run management for the Transformers Qwen answer model."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .config import Settings


class QwenModelManager:
    """Download Qwen3 into the configured Hugging Face cache when needed."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = settings.model_download_repository
        self.cache_dir = (
            settings.huggingface_home or settings.application_data_path / "models" / "huggingface"
        ).expanduser().resolve()
        self.model_page = f"https://huggingface.co/{self.repository}"
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "state": "missing",
            "model_id": self.repository,
            "download_url": self.model_page,
            "storage_path": str(self.cache_dir),
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "error": None,
        }
        self._refresh_ready_state()

    def start_auto_download(self) -> dict[str, Any]:
        if not self._uses_transformers_model() or not self.settings.model_auto_download:
            return self.status()
        if self._is_ready():
            return self.status()
        return self.start_download()

    def start_download(self) -> dict[str, Any]:
        if not self._uses_transformers_model():
            return self.status()
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status()
            self._state.update(
                {
                    "state": "downloading",
                    "error": None,
                    "downloaded_bytes": self._cache_bytes(),
                }
            )
            self._worker = threading.Thread(
                target=self._download_worker,
                name="qwen-model-download",
                daemon=True,
            )
            self._worker.start()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._state["state"] != "downloading" and self._is_ready():
                self._state["state"] = "ready"
                self._state["error"] = None
            return dict(self._state)

    def is_ready(self) -> bool:
        return self._is_ready()

    def _uses_transformers_model(self) -> bool:
        model_id = str(self.settings.llm_model_id)
        return self.settings.llm_backend == "qwen_transformers" and not Path(model_id).exists()

    def _refresh_ready_state(self) -> None:
        if self._is_ready():
            self._state["state"] = "ready"

    def _is_ready(self) -> bool:
        model_id = str(self.settings.llm_model_id)
        model_path = Path(model_id).expanduser()
        if model_path.is_dir():
            return (model_path / "config.json").is_file() and any(
                (model_path / name).is_file()
                for name in ("model.safetensors", "pytorch_model.bin")
            )
        if not self._uses_transformers_model():
            return True
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=self.repository,
                    cache_dir=str(self.cache_dir / "hub"),
                    local_files_only=True,
                )
            )
        except Exception:
            return False
        return (snapshot / "config.json").is_file() and any(
            (snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
        )

    def _download_worker(self) -> None:
        result: dict[str, object] = {"path": None, "error": None}
        worker = threading.Thread(
            target=self._snapshot_download,
            args=(result,),
            name="qwen-model-snapshot",
            daemon=True,
        )
        worker.start()
        while worker.is_alive():
            with self._lock:
                self._state["downloaded_bytes"] = self._cache_bytes()
            worker.join(timeout=0.5)
        worker.join()
        with self._lock:
            self._state["downloaded_bytes"] = self._cache_bytes()
            error = result.get("error")
            if error is not None:
                self._state.update({"state": "error", "error": str(error)})
                return
            if not self._is_ready():
                self._state.update(
                    {
                        "state": "error",
                        "error": "Qwen3 model download completed but required files are missing",
                    }
                )
                return
            self._state.update({"state": "ready", "error": None})

    def _snapshot_download(self, result: dict[str, object]) -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.repository,
                cache_dir=str(self.cache_dir / "hub"),
                local_files_only=False,
            )
        except Exception as exc:  # surfaced in the model status endpoint
            result["error"] = exc

    def _cache_bytes(self) -> int:
        cache_root = self.cache_dir / "hub" / (
            "models--" + self.repository.replace("/", "--")
        )
        if not cache_root.is_dir():
            return 0
        total = 0
        for path in cache_root.rglob("*"):
            if path.is_file() and ".locks" not in path.parts:
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total
