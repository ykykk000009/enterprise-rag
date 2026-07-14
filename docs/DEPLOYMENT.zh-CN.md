# GitHub 发布、打包与部署指南

面向不懂 Python 的 Windows 个人用户，优先使用 [Windows 应用打包与交付](WINDOWS_APP_PACKAGING.zh-CN.md) 中的 EXE 便携版。本文的源码和 Docker 方式更适合开发者或服务器管理员。

## 1. 发布前整理

仓库应提交源码、测试、文档和部署配置，不提交运行数据、客户文件、模型缓存或密钥。本项目的 `.gitignore` 已排除 `.env`、`data/` 和 `models/`；发布前仍应执行：

```powershell
git status --short
git ls-files | Select-String -Pattern "(^|/)(\.env|data|models|knowledge)/"
```

确认输出中没有敏感文件后运行测试：

```powershell
python -m pip install -e ".[dev]"
pytest
```

## 2. 推送到 GitHub

当前工作区的上级 `.git` 目录为空，不构成有效 Git 仓库。若要把 `codex_agent_mvp` 单独发布，最直接的方式是在本目录初始化：

```powershell
Set-Location E:\findfileagent\codex_agent_mvp
git init
git add .
git commit -m "docs: add packaging and deployment guide"
git branch -M main
git remote add origin https://github.com/<你的账号>/<仓库名>.git
git push -u origin main
```

也可以先把 `codex_agent_mvp` 复制到新的干净目录再初始化。GitHub 网页上先创建一个空仓库，不要预生成 README，以减少首次推送冲突。首次 `git add` 后务必再次运行 `git status --short`，确认没有模型、索引或业务文档。

建议在仓库页面补充许可证（如 MIT、Apache-2.0 或企业内部许可证）。没有许可证时，其他人默认无权复制、修改和分发代码。

## 3. 源码方式部署（推荐用于 Windows 办公电脑）

接收方执行：

```powershell
git clone https://github.com/<账号>/<仓库名>.git
Set-Location <仓库名>\codex_agent_mvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
Copy-Item .env.example .env
New-Item -ItemType Directory -Force knowledge, data, models\huggingface
uvicorn enterprise_document_rag.main:app --host 0.0.0.0 --port 8000 --workers 1
```

接着把文件复制到 `knowledge/`。当前版本尚未强制执行 `.env` 中 `AUTHORIZED_ROOTS` 的白名单，因此不要只依赖该配置隔离文件；应使用专用 Windows 账号运行服务，并只授予它读取业务文档目录的权限。Windows 防火墙若询问，仅在可信的专用网络放行。局域网访问地址为 `http://<部署机IP>:8000`。

首次进行索引或问答时会下载对应模型。要完全离线部署，可先在联网机器完成一次索引和问答，使三个模型写入 `models/huggingface`（重排模型仅在启用时下载），然后连同该缓存通过移动硬盘交付；不要把数 GB 模型文件放入普通 Git 仓库。

## 4. Docker Compose 部署（推荐用于 Linux 服务器）

部署机安装 Docker Engine 与 Compose 插件后：

```bash
git clone https://github.com/<账号>/<仓库名>.git
cd <仓库名>/codex_agent_mvp
cp .env.example .env
mkdir -p data knowledge models
docker compose up -d --build
docker compose logs -f rag-agent
```

访问 `http://<服务器IP>:8000`。Compose 已做以下映射：

- `./data` → 数据库与向量索引，可持久化；
- `./knowledge` → 容器内只读文档目录；
- `./models` → Hugging Face 模型缓存；
- 宿主机 `8000` → 应用端口。

更新版本：

```bash
git pull --ff-only
docker compose up -d --build
```

停止服务不会删除数据：

```bash
docker compose down
```

不要执行 `docker compose down -v` 或手工删除 `data/`，除非明确要清空索引。

## 5. 构建 Python 安装包

若要交付 wheel 而不是源码：

```powershell
python -m pip install build
python -m build
```

产物位于 `dist/`。接收方可以执行：

```powershell
python -m pip install .\dist\enterprise_document_local_rag_agent-0.1.0-py3-none-any.whl
uvicorn enterprise_document_rag.main:app --host 0.0.0.0 --port 8000 --workers 1
```

wheel 只包含应用代码和静态页面，不包含模型、索引、业务文档，也不会把第三方依赖打入单个文件；安装时仍需从 Python 软件源下载依赖。严格离线交付时，需要额外建立与目标操作系统和 Python 版本一致的 wheelhouse。

## 6. 生产部署检查单

- 保持单进程 `--workers 1`，避免多个进程同时打开 Qdrant Local；
- 只挂载确有授权的文档目录，并尽量使用只读挂载；
- 在正式对外服务前，实现并测试 `AUTHORIZED_ROOTS` 的服务端强制校验；
- 不把 8000 端口直接暴露到公网，在 Nginx/Caddy 后配置 HTTPS 与身份认证；
- 限制安全组/防火墙来源地址；
- 定期备份 `data/`，并演练恢复；
- 监控磁盘、内存、任务失败数和 `/health/ready`；
- 升级前备份数据，并先在副本环境运行测试；
- 若不同用户必须看到不同文档，应由可信认证层计算并传入 `allowed_document_ids`，不能信任客户端自行声明权限。
