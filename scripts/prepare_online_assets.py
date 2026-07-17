"""Prepare the bundled Transformers embedding and reranker models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

EMBEDDING_REPOSITORY = "BAAI/bge-small-zh-v1.5"
RERANKER_REPOSITORY = "BAAI/bge-reranker-base"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".online-assets"))
    parser.add_argument("--huggingface-cache", type=Path)
    parser.add_argument("--proxy", help="Optional HTTP/HTTPS proxy, for example http://127.0.0.1:7899")
    return parser.parse_args()


def _snapshot(repository: str, cache_dir: Path | None, allow_patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repository,
            cache_dir=str(cache_dir / "hub") if cache_dir else None,
            allow_patterns=allow_patterns,
            local_files_only=cache_dir is not None,
        )
    )


def _copy_model(source: Path, target: Path, *, skip_onnx: bool = False) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in source.rglob("*"):
        if not item.is_file() or ".locks" in item.parts:
            continue
        relative = item.relative_to(source)
        if skip_onnx and relative.parts and relative.parts[0].lower() == "onnx":
            continue
        if item.name == "pytorch_model.bin" and (source / "model.safetensors").is_file():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"no model files found in {source}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _arguments()
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.huggingface_cache.resolve() if args.huggingface_cache else None

    embedding_source = _snapshot(
        EMBEDDING_REPOSITORY,
        cache_dir,
        ["*.json", "*.txt", "*.model", "*.safetensors", "1_Pooling/*"],
    )
    reranker_source = _snapshot(
        RERANKER_REPOSITORY,
        cache_dir,
        ["*.json", "*.txt", "*.model", "*.safetensors", "tokenizer*", "special*"],
    )
    _copy_model(
        embedding_source,
        output / "models" / "embedding-bge-small-zh-v1.5",
    )
    _copy_model(
        reranker_source,
        output / "models" / "reranker-bge-base",
        skip_onnx=True,
    )

    licenses = output / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (embedding_source / "LICENSE", "bge-small-zh-v1.5-MIT.txt"),
        (reranker_source / "LICENSE", "bge-reranker-base-MIT.txt"),
    ):
        if source.is_file():
            shutil.copy2(source, licenses / name)

    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (output / "MODEL_MANIFEST.json").write_text(
        json.dumps(
            {
                "models": [
                    {"name": EMBEDDING_REPOSITORY, "format": "Transformers"},
                    {"name": RERANKER_REPOSITORY, "format": "Transformers"},
                ],
                "answer_model": {
                    "name": "Qwen/Qwen3-0.6B",
                    "packaged": False,
                    "download_url": "https://huggingface.co/Qwen/Qwen3-0.6B",
                },
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Online model assets prepared at {output}")


if __name__ == "__main__":
    main()
