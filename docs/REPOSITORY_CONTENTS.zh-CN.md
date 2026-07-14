# GitHub 仓库内容说明

## 上传内容

以下内容适合公开，便于学习、二次开发和自行构建：

| 路径 | 内容 |
|---|---|
| `src/enterprise_document_rag/` | FastAPI 后端、解析、OCR、切块、检索、问答及前端页面 |
| `tests/` | 单元测试和 API 测试 |
| `docs/` | 技术架构、安装、部署、评测和打包说明 |
| `evaluation/` | RAG 评测脚本、数据结构定义和评测说明（不含真实业务数据） |
| `scripts/` | 备份、修复、重嵌入和基准测试工具 |
| `packaging/windows/` | Windows EXE/ZIP 构建脚本和用户说明 |
| `Dockerfile`、`docker-compose.yml` | Linux/Docker 部署配置 |
| `pyproject.toml` | Python 依赖、打包、测试与代码规范配置 |
| `.env.example` | 不含密钥的配置示例 |
| `SPEC.md`、`spec.json`、`tasks.json`、`acceptance.json` | 项目规格、任务拆分和验收标准 |
| `README.md`、`LICENSE` | 项目入口与开源许可 |

## 不上传内容

以下内容可能很大、包含本机状态或业务数据，已通过 `.gitignore` 排除：

| 路径/文件 | 排除原因 |
|---|---|
| `.env` | 可能包含本机路径或后续添加的敏感配置 |
| `data/`、`user-data/` | SQLite、Qdrant 索引和运行状态 |
| `knowledge/` | 用户业务文档 |
| `models/` | Hugging Face 模型缓存，体积很大 |
| `dist/`、`build/` | EXE、ZIP 和临时构建产物 |
| `.venv*` | Python 虚拟环境 |
| `.pytest_cache/`、`.ruff_cache/`、`__pycache__/` | 工具缓存 |
| `.tmp-package/`、`.pyinstaller-cache/` | Windows 打包缓存 |
| `evaluation/dataset.jsonl`、`evaluation/rag_eval_set.json` | 从真实资料生成的评测问题和证据文本 |
| `evaluation/results/`、`evaluation/*report*.md`、`reports/` | 可能包含真实路径、文档片段或内部评测结果 |

Windows ZIP 应上传到 GitHub Release，而不是提交进 Git 历史。普通 GitHub 文件超过 100 MB 会被拒绝，且二进制发行包会让源码仓库快速膨胀。

## 发布前检查

```powershell
git status --short
git ls-files | Select-String -Pattern "(^|/)(\.env|data|models|knowledge|user-data|dist|build)/"
pytest
```

确认没有真实文档、模型、数据库、索引或密钥后再推送。
