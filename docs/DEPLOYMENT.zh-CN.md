# 构建与部署

## 源码运行

```powershell
git clone https://github.com/ykykk000009/enterprise-rag.git
cd enterprise-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn enterprise_document_rag.main:app --host 127.0.0.1 --port 8000 --workers 1
```

保持 `--workers 1`，避免多个进程同时打开 Qdrant Local。

## 准备离线资产

准备机需要约 8 GB 临时空间。已有 Hugging Face 缓存时可复用，脚本只复制指定模型，
不会把其他缓存打入发布包。动态量化 Reranker 需要额外安装 `onnx`：

```powershell
python -m pip install onnx
$env:HTTP_PROXY="http://127.0.0.1:7899"
$env:HTTPS_PROXY="http://127.0.0.1:7899"
python .\scripts\prepare_offline_assets.py `
  --output .\.offline-assets `
  --huggingface-cache E:\models\huggingface\hub `
  --libarchive-bin D:\Anaconda\Library\bin `
  --proxy http://127.0.0.1:7899
```

输出目录包含：

```text
.offline-assets\
  models\embedding-bge-small-zh-v1.5\
  models\reranker-bge-base-int8\
  models\qwen3\Qwen3-0.6B-Q8_0.gguf
  tools\llama.cpp\
  tools\libarchive\
  licenses\
  MODEL_MANIFEST.json
```

`.offline-assets` 被 Git 忽略，只用于本地打包。

## 一次构建两个 Windows 完整包

构建机需要 Windows 10/11 x64、Python 3.11/3.12 和至少 15 GB 可用空间：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\windows\build_windows.ps1 `
  -OfflineAssetsPath .\.offline-assets
```

脚本创建隔离构建环境、安装依赖、执行测试、构建 `DocQA.exe` 和 `Updater.exe`，然后
依次输出：

```text
dist\windows\DocQA-vX.Y.Z-win-x64.zip
dist\windows\DocQA-vX.Y.Z-win-x64.zip.sha256
dist\windows\DocQA-vX.Y.Z-win-x64-offline.zip
dist\windows\DocQA-vX.Y.Z-win-x64-offline.zip.sha256
```

仅调试在线包时可添加 `-OnlineOnly`。正式发布不得使用该选项。

构建脚本如果发现 `dist/windows/DocQA/user-data` 会立即停止。模型、索引、业务文档、
`.env`、测试目录和凭据不进入源码提交；离线模型只作为 Release 二进制附件发布。

## 发布检查

1. `pytest` 与 `ruff` 通过；
2. 两个 ZIP 均包含 `DocQA.exe`、`Updater.exe`、`version.json` 和 `docqa.ico`；
3. 离线 ZIP 额外包含 `offline.mode`、三套模型、工具、许可证和模型清单；
4. ZIP 不含 `user-data`、`.env`、`tests`、业务文档、数据库或凭据；
5. 在无网络环境运行离线包，完成健康检查、建库、扫描、检索和一次问答；
6. 校验两个 `.sha256`；
7. 推送源码，再创建同版本 Git tag 和 GitHub Release，上传四个附件。

完整更新规则见 [UPDATES.zh-CN.md](UPDATES.zh-CN.md)。
