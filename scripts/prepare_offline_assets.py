"""Prepare redistributable models and native tools for the complete offline package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

QWEN_REPOSITORY = "Qwen/Qwen3-0.6B-GGUF"
QWEN_FILE = "Qwen3-0.6B-Q8_0.gguf"
EMBEDDING_REPOSITORY = "BAAI/bge-small-zh-v1.5"
RERANKER_REPOSITORY = "BAAI/bge-reranker-base"
LLAMA_CPP_VERSION = "b10050"
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_VERSION}/llama-{LLAMA_CPP_VERSION}-bin-win-cpu-x64.zip"
)
LIBARCHIVE_VERSION = "3.7.4"
LIBARCHIVE_RUNTIME_FILES = (
    "bsdtar.exe",
    "archive.dll",
    "zlib.dll",
    "libbz2.dll",
    "liblzma.dll",
    "liblz4.dll",
    "zstd.dll",
    "libcrypto-3-x64.dll",
    "iconv.dll",
    "charset.dll",
    "libxml2.dll",
)
LIBARCHIVE_CONDA_PACKAGES = (
    "libarchive",
    "bzip2",
    "libiconv",
    "libxml2",
    "lz4-c",
    "openssl",
    "xz",
    "zlib",
    "zstd",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".offline-assets"))
    parser.add_argument(
        "--huggingface-cache",
        type=Path,
        help="Existing Hugging Face hub cache; downloads missing files when omitted.",
    )
    parser.add_argument(
        "--libarchive-bin",
        type=Path,
        help="Directory containing a conda bsdtar.exe and its runtime DLLs.",
    )
    parser.add_argument("--proxy", help="Optional HTTP/HTTPS proxy, for example http://127.0.0.1:7899")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
            destination = (target / Path(*path.parts)).resolve()
            if destination != target_resolved and target_resolved not in destination.parents:
                raise RuntimeError(f"path escapes extraction directory: {member.filename}")
        source.extractall(target)


def _download(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DocQA-offline-builder/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    os.replace(partial, destination)
    return destination


def _snapshot(
    *,
    repository: str,
    cache_dir: Path | None,
    allow_patterns: list[str],
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repository,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=allow_patterns,
            local_files_only=False,
        )
    )


def _copy_files(source: Path, target: Path, patterns: list[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in patterns:
        for item in source.glob(pattern):
            if not item.is_file():
                continue
            destination = target / item.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            copied += 1
    if copied == 0:
        raise RuntimeError(f"no matching model files found in {source}")


def _prepare_models(output: Path, cache_dir: Path | None) -> None:
    embedding_source = _snapshot(
        repository=EMBEDDING_REPOSITORY,
        cache_dir=cache_dir,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "1_Pooling/*",
        ],
    )
    _copy_files(
        embedding_source,
        output / "models" / "embedding-bge-small-zh-v1.5",
        ["*.json", "*.txt", "*.model", "*.safetensors", "1_Pooling/*"],
    )

    reranker_source = _snapshot(
        repository=RERANKER_REPOSITORY,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
            "onnx/model.onnx",
        ],
    )
    reranker_target = output / "models" / "reranker-bge-base-int8"
    _copy_files(
        reranker_source,
        reranker_target,
        [
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
        ],
    )
    quantized_model = reranker_target / "model.int8.onnx"
    if not quantized_model.is_file():
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise RuntimeError(
                "Preparing the INT8 reranker requires: python -m pip install onnx"
            ) from exc
        quantize_dynamic(
            str(reranker_source / "onnx" / "model.onnx"),
            str(quantized_model),
            weight_type=QuantType.QInt8,
        )

    qwen_source = _snapshot(
        repository=QWEN_REPOSITORY,
        cache_dir=cache_dir,
        allow_patterns=[QWEN_FILE, "LICENSE", "README.md"],
    )
    _copy_files(qwen_source, output / "models" / "qwen3", [QWEN_FILE])
    licenses = output / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    if (qwen_source / "LICENSE").is_file():
        shutil.copy2(qwen_source / "LICENSE", licenses / "Qwen3-Apache-2.0.txt")


def _conda_prefix_from_bin(bin_dir: Path) -> Path | None:
    if bin_dir.name.lower() == "bin" and bin_dir.parent.name.lower() == "library":
        return bin_dir.parent.parent
    return None


def _copy_conda_licenses(prefix: Path, output: Path) -> None:
    license_target = output / "licenses" / "libarchive-dependencies"
    license_target.mkdir(parents=True, exist_ok=True)
    packages: list[dict[str, object]] = []
    for package_name in LIBARCHIVE_CONDA_PACKAGES:
        metadata_files = sorted((prefix / "conda-meta").glob(f"{package_name}-*.json"))
        metadata_candidates = [
            json.loads(path.read_text(encoding="utf-8")) for path in metadata_files
        ]
        matching_metadata = [
            item for item in metadata_candidates if item.get("name") == package_name
        ]
        if not matching_metadata:
            continue
        metadata = matching_metadata[-1]
        extracted = Path(str(metadata.get("extracted_package_dir") or ""))
        source_licenses = extracted / "info" / "licenses"
        if source_licenses.is_dir():
            for source in source_licenses.rglob("*"):
                if source.is_file():
                    destination = license_target / f"{package_name}-{source.name}"
                    shutil.copy2(source, destination)
        packages.append(
            {
                key: metadata.get(key)
                for key in ("name", "version", "build", "license", "url", "sha256")
            }
        )
    (license_target / "CONDA_PACKAGES.json").write_text(
        json.dumps(packages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_tools(output: Path, downloads: Path, libarchive_bin: Path | None) -> None:
    llama_archive = _download(LLAMA_CPP_URL, downloads / Path(LLAMA_CPP_URL).name)
    llama_target = output / "tools" / "llama.cpp"
    shutil.rmtree(llama_target, ignore_errors=True)
    llama_target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        extracted = Path(temporary)
        _safe_extract(llama_archive, extracted)
        for source in extracted.rglob("*"):
            if source.is_file() and (
                source.name == "llama-cli.exe" or source.suffix.lower() == ".dll"
            ):
                shutil.copy2(source, llama_target / source.name)
    if not (llama_target / "llama-cli.exe").is_file():
        raise RuntimeError("llama.cpp package does not contain llama-cli.exe")

    if libarchive_bin is None:
        located = shutil.which("bsdtar")
        if located is None:
            raise RuntimeError(
                "bsdtar was not found; pass --libarchive-bin pointing to a conda Library/bin"
            )
        libarchive_bin = Path(located).resolve().parent
    else:
        libarchive_bin = libarchive_bin.resolve()
    libarchive_target = output / "tools" / "libarchive"
    shutil.rmtree(libarchive_target, ignore_errors=True)
    libarchive_target.mkdir(parents=True, exist_ok=True)
    for name in LIBARCHIVE_RUNTIME_FILES:
        source = libarchive_bin / name
        if not source.is_file():
            raise RuntimeError(f"libarchive runtime file is missing: {source}")
        shutil.copy2(source, libarchive_target / name)
    conda_prefix = _conda_prefix_from_bin(libarchive_bin)
    if conda_prefix is not None:
        _copy_conda_licenses(conda_prefix, output)


def _prepare_licenses(output: Path) -> None:
    licenses = output / "licenses"
    sources = {
        "llama.cpp-MIT.txt": "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE",
        "libarchive-BSD.txt": "https://raw.githubusercontent.com/libarchive/libarchive/master/COPYING",
        "FlagEmbedding-MIT.txt": (
            "https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/LICENSE"
        ),
        "lz4-BSD-2-Clause.txt": "https://raw.githubusercontent.com/lz4/lz4/dev/LICENSE",
        "zstd-BSD-GPL.txt": "https://raw.githubusercontent.com/facebook/zstd/dev/LICENSE",
        "zlib-Zlib.txt": "https://raw.githubusercontent.com/madler/zlib/master/LICENSE",
    }
    for name, url in sources.items():
        _download(url, licenses / name)


def _write_manifest(output: Path) -> None:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "MODEL_MANIFEST.json" or "downloads" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "models": [
            {
                "name": EMBEDDING_REPOSITORY,
                "license": "MIT",
                "purpose": "Chinese dense embedding",
            },
            {
                "name": RERANKER_REPOSITORY,
                "license": "MIT",
                "format": "dynamic INT8 ONNX converted from the official ONNX model",
                "purpose": "cross-encoder reranking",
            },
            {
                "name": QWEN_REPOSITORY,
                "license": "Apache-2.0",
                "file": QWEN_FILE,
                "purpose": "local answer generation",
            },
        ],
        "tools": [
            {
                "name": "llama.cpp",
                "version": LLAMA_CPP_VERSION,
                "license": "MIT",
                "source": LLAMA_CPP_URL,
            },
            {
                "name": "libarchive bsdtar",
                "version": LIBARCHIVE_VERSION,
                "license": "BSD-2-Clause and component notices",
                "source": "https://repo.anaconda.com/pkgs/main/win-64/",
            },
        ],
        "files": files,
    }
    (output / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = _arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    cache_dir = args.huggingface_cache.resolve() if args.huggingface_cache else None
    downloads = output / "downloads"
    _prepare_models(output, cache_dir)
    _prepare_tools(output, downloads, args.libarchive_bin)
    _prepare_licenses(output)
    _write_manifest(output)
    shutil.rmtree(downloads, ignore_errors=True)
    print(f"Offline assets prepared at {output}")


if __name__ == "__main__":
    main()
