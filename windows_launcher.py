"""Windows desktop launcher for the local web application."""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import LEFT, Button, Frame, Label, StringVar, Tk, X, messagebox

APP_NAME = "EnterpriseDocumentRAG"


def _application_home() -> Path:
    bundle = _bundle_home()
    if (bundle / "portable.mode").is_file():
        home = bundle / "user-data"
    else:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        home = base / APP_NAME
    for child in ("data", "knowledge", "models/huggingface"):
        (home / child).mkdir(parents=True, exist_ok=True)
    return home


def _bundle_home() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _configure_environment(home: Path) -> None:
    bundled_models = _bundle_home() / "models" / "huggingface"
    model_home = bundled_models if bundled_models.is_dir() else home / "models" / "huggingface"
    os.environ.setdefault("APP_ENV", "production")
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{(home / 'data' / 'agent.db').as_posix()}")
    os.environ.setdefault("QDRANT_PATH", str(home / "data" / "qdrant"))
    os.environ.setdefault("AUTHORIZED_ROOTS", str(home / "knowledge"))
    os.environ.setdefault("HUGGINGFACE_HOME", str(model_home))
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
        self.root.title("企业文档智能检索")
        self.root.geometry("520x230")
        self.root.minsize(480, 210)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.status = StringVar(value="正在启动本地服务，请稍候……")
        Label(self.root, text="企业文档智能检索", font=("Microsoft YaHei UI", 16, "bold")).pack(
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

            config = uvicorn.Config(
                create_app(),
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


def main() -> None:
    DesktopLauncher().start()


if __name__ == "__main__":
    main()
