"""Windows desktop launcher for the local web application."""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from contextlib import suppress
from pathlib import Path
from tkinter import LEFT, Button, Frame, Label, StringVar, Tk, X, messagebox

APP_NAME = "DocQA"
LEGACY_APP_NAME = "EnterpriseDocumentRAG"


def _application_home() -> Path:
    bundle = _bundle_home()
    if (bundle / "portable.mode").is_file():
        home = bundle / "user-data"
    else:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        preferred_home = base / APP_NAME
        legacy_home = base / LEGACY_APP_NAME
        home = (
            legacy_home
            if legacy_home.is_dir() and not preferred_home.exists()
            else preferred_home
        )
    for child in ("data", "knowledge", "models/huggingface"):
        (home / child).mkdir(parents=True, exist_ok=True)
    return home


def _bundle_home() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _configure_environment(home: Path) -> None:
    bundle = _bundle_home()
    bundled_models = bundle / "models" / "huggingface"
    model_home = bundled_models if bundled_models.is_dir() else home / "models" / "huggingface"
    os.environ.setdefault("APP_ENV", "production")
    os.environ.setdefault("APP_DATA_DIR", str(home))
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{(home / 'data' / 'agent.db').as_posix()}")
    os.environ.setdefault("QDRANT_PATH", str(home / "data" / "qdrant"))
    os.environ.setdefault("AUTHORIZED_ROOTS", str(home / "knowledge"))
    os.environ.setdefault("HUGGINGFACE_HOME", str(model_home))
    offline_marker = bundle / "offline.mode"
    online_models_marker = bundle / "online-models.mode"
    embedding_model = bundle / "models" / "embedding-bge-small-zh-v1.5"
    reranker_model = bundle / "models" / "reranker-bge-base-int8"
    online_reranker_model = bundle / "models" / "reranker-bge-base"
    qwen_model = bundle / "models" / "qwen3" / "Qwen3-0.6B-Q8_0.gguf"
    llama_cli = bundle / "tools" / "llama.cpp" / "llama-cli.exe"
    bsdtar = bundle / "tools" / "libarchive" / "bsdtar.exe"
    # Prefer the current Transformers package when an older GGUF installation
    # still has a stale offline.mode marker after an update.
    if online_models_marker.is_file():
        required = (embedding_model,)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "Online model package is missing bundled model assets:\n" + "\n".join(missing)
            )
        os.environ.setdefault("EMBEDDING_MODEL", str(embedding_model))
        os.environ["RERANKER_ENABLED"] = "false"
        os.environ.setdefault("LLM_BACKEND", "qwen_transformers")
        os.environ.setdefault("LLM_MODEL_ID", "Qwen/Qwen3-0.6B")
        os.environ.setdefault("MODEL_AUTO_DOWNLOAD", "true")
    elif offline_marker.is_file():
        required = (embedding_model, reranker_model, qwen_model, llama_cli, bsdtar)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError("离线完整版缺少必要资产：\n" + "\n".join(missing))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("EMBEDDING_MODEL", str(embedding_model))
        os.environ["RERANKER_ENABLED"] = "false"
        os.environ.setdefault("LLM_BACKEND", "qwen_gguf_cli")
        os.environ.setdefault("LLM_MODEL_ID", str(qwen_model))
        os.environ.setdefault("LLAMA_CLI_PATH", str(llama_cli))
        os.environ.setdefault("BSDTAR_PATH", str(bsdtar))
    os.chdir(home)


def _available_port(preferred: int = 8765) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("没有可用的本地端口（已检查 8765-8784）")


class DesktopLauncher:
    def __init__(self) -> None:
        self.home = _application_home()
        self.log_file = (self.home / "launcher.log").open(
            "a", encoding="utf-8", buffering=1
        )
        if sys.stdout is None:
            sys.stdout = self.log_file
        if sys.stderr is None:
            sys.stderr = self.log_file
        _configure_environment(self.home)
        self.port = _available_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = None
        self.error: BaseException | None = None

        self.root = Tk()
        self.root.title("Document RAG")
        icon_path = _bundle_home() / "docqa.ico"
        if icon_path.is_file():
            with suppress(Exception):
                self.root.iconbitmap(default=str(icon_path))
        self.root.geometry("520x230")
        self.root.minsize(480, 210)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.status = StringVar(value="正在启动本地服务，请稍候……")
        Label(self.root, text="Document RAG", font=("Microsoft YaHei UI", 16, "bold")).pack(
            pady=(24, 10)
        )
        Label(self.root, textvariable=self.status, font=("Microsoft YaHei UI", 10)).pack(
            padx=20, pady=8
        )
        Label(
            self.root,
            text=f"数据保存在：{self.home}",
            font=("Microsoft YaHei UI", 9),
            wraplength=470,
        ).pack(padx=20, pady=4)

        buttons = Frame(self.root)
        buttons.pack(fill=X, padx=40, pady=18)
        self.open_button = Button(
            buttons, text="打开应用", command=self.open_browser, state="disabled", width=14
        )
        self.open_button.pack(side=LEFT, expand=True)
        Button(buttons, text="打开数据目录", command=self.open_data_folder, width=14).pack(
            side=LEFT, expand=True
        )
        Button(buttons, text="退出", command=self.stop, width=10).pack(side=LEFT, expand=True)

    def start(self) -> None:
        threading.Thread(target=self._run_server, name="local-web-server", daemon=True).start()
        self.root.after(200, self._check_server)
        self.root.mainloop()

    def _run_server(self) -> None:
        try:
            import uvicorn

            from enterprise_document_rag.main import create_app

            app = create_app()
            app.state.shutdown_callback = self._shutdown_for_update
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                log_config=None,
                access_log=False,
            )
            self.server = uvicorn.Server(config)
            self.server.run()
        except BaseException as exc:  # surfaced in the launcher window
            self.error = exc

    def _check_server(self) -> None:
        if self.error is not None:
            self.status.set("启动失败")
            messagebox.showerror("启动失败", f"本地服务无法启动：\n\n{self.error}")
            return
        if self.server is not None and self.server.started:
            self.status.set("本地服务已启动。关闭此窗口将退出应用。")
            self.open_button.config(state="normal")
            self.open_browser()
            return
        self.root.after(200, self._check_server)

    def open_browser(self) -> None:
        webbrowser.open(self.url)

    def open_data_folder(self) -> None:
        os.startfile(self.home)  # type: ignore[attr-defined]

    def stop(self) -> None:
        if self.server is not None:
            self.status.set("正在退出……")
            self.server.should_exit = True
        self.root.after(250, self.root.destroy)

    def _shutdown_for_update(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        self.root.after(0, self.root.destroy)


def main() -> None:
    DesktopLauncher().start()


if __name__ == "__main__":
    main()
