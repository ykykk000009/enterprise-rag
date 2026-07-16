# 构建与部署

## 源码部署

```powershell
git clone https://github.com/ykykk000009/enterprise-rag.git
cd enterprise-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
Copy-Item .env.example .env
New-Item -ItemType Directory -Force knowledge, data, models\huggingface
uvicorn enterprise_document_rag.main:app --host 0.0.0.0 --port 8000 --workers 1
```

保持 `--workers 1`，避免多个进程同时打开 Qdrant Local。源码部署应使用权限受限的系统账号；当前 `AUTHORIZED_ROOTS` 尚未由 API 强制校验。

## Docker Compose

```bash
git clone https://github.com/ykykk000009/enterprise-rag.git
cd enterprise-rag
cp .env.example .env
mkdir -p data knowledge models
docker compose up -d --build
```

映射目录：`data/` 保存索引，`knowledge/` 只读挂载文档，`models/` 保存模型缓存。更新使用 `git pull --ff-only && docker compose up -d --build`。

## Windows 便携包

构建机需要 Windows x64、Python 3.11/3.12 和至少 15 GB 可用空间。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\build_windows.ps1
```

脚本会创建独立环境、运行测试并生成 `dist/windows/DocQA-vX.Y.Z-win-x64.zip`。携带模型的离线包可使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\windows\build_windows.ps1 `
  -IncludeModels `
  -ModelCachePath E:\models\huggingface
```

PyInstaller 应在 Windows 上构建 Windows EXE。项目采用 `--onedir`，避免单文件模式启动慢和临时解压问题。

## Python wheel

```powershell
python -m pip install build
python -m build
```

wheel 位于 `dist/`，不包含模型、索引或业务文档。

## 发布与生产检查

- 源码推送到 Git；Windows ZIP 作为 GitHub Release 附件，不提交进 Git 历史。
- 发布前运行 `pytest`、敏感信息扫描和干净 Windows 启动测试。
- 构建脚本在 `dist/windows/DocQA` 中发现 `user-data` 时会停止，防止把本地知识库或企业文档打入发布包。
- 公网部署必须增加 HTTPS、身份认证、防火墙和服务端目录白名单。
- 定期备份 `data/`，监控磁盘、内存、失败任务和 `/health/ready`。
- 多实例部署前将 Qdrant Local 替换为 Qdrant Server，并增加任务协调。

## Bug 修复与 Windows 用户升级

GitHub 仓库管理的是源码，不会直接修改用户电脑中已经安装的 EXE。一次完整更新应按以下顺序进行：

```powershell
git pull --ff-only
# 修改代码
pytest
git status --short
git add <本次修改的文件>
git commit -m "fix: describe the bug"
git push origin main

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\build_windows.ps1
```

为新版本创建 Git 标签和 GitHub Release，将生成的 ZIP 与 `SHA256SUMS.txt` 上传为附件。用户下载新版本后，按照 Windows 安装说明保留原来的 `user-data` 完成升级。

如需让用户端自动检测和安装新版本，需要另外实现带签名校验、失败回滚和 `user-data` 保护的更新器；当前便携版采用 GitHub Release 手动升级。
